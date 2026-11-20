"""Resumable account-wide indexing; each conversation keeps its own checkpoint."""
from __future__ import annotations

import json
import time

from .chats import MAX_MESSAGES, digest
from .weflow import SyncCancelled, WeFlowError, sync_session
from .weflow_protocol import ISSUE_LABELS
from .weflow_scope import validate_scope


DONE = {"done", "empty", "skipped"}


def batch_key(client) -> str:
    return "weflow_batch:" + digest(client.base_url + "\0" + (client.account_id or "legacy"))


def load_batch(db, client) -> dict | None:
    try:
        job = json.loads(db.get_setting(batch_key(client), "null"))
        if isinstance(job, dict) and job.get("version") in {1, 2} and isinstance(job.get("sessions"), list):
            return job
    except (TypeError, ValueError):
        pass
    return None


def batch_summary(job: dict) -> dict:
    rows = job["sessions"]
    return {"state": job["state"], "start": job["start"], "end": job["end"], "total_sessions": len(rows),
            "new_sessions": job.get("new_sessions", 0),
            "completed_sessions": sum(row["status"] in DONE or row["status"] in {"failed", "partial"} for row in rows),
            "succeeded_sessions": sum(row["status"] in {"done", "empty"} for row in rows),
            "failed_sessions": sum(row["status"] == "failed" for row in rows),
            "partial_sessions": sum(row["status"] == "partial" for row in rows),
            "skipped_sessions": sum(row["status"] == "skipped" for row in rows),
            **{key: sum(row.get(key, 0) for row in rows) for key in ("received", "updated_units", "reused_units", "failed_messages")}}


def sync_all_sessions(db, client, *, start: int, end: int | None = None, incremental=True,
                      max_messages=5000, embedding_service=None, should_cancel=lambda: False,
                      progress=lambda _: None, on_update=lambda _: None, scope: dict | None = None) -> dict:
    end = int(time.time()) if end is None else end
    if not 0 <= start <= end or not 1 <= max_messages <= MAX_MESSAGES:
        raise WeFlowError("同步范围或消息上限无效")
    # Every invocation checks the live account and discovers new conversations.
    sessions = client.sessions(should_cancel=should_cancel, progress=progress)
    available = {row["id"]: row for row in sessions}
    scope_id = None
    if scope is not None:
        try:
            validate_scope(scope, client.base_url, client.account_id)
        except ValueError as exc:
            raise WeFlowError(str(exc), code="scope_changed") from None
        scope_id = scope["fingerprint"]
        sessions = [available.get(identity, {"id": identity, "name": identity, "category": scope["types"][identity]})
                    for identity in scope["ids"]]
    previous = load_batch(db, client)
    resume = bool(incremental and client.protocol == "alpha" and previous and previous.get("start") == start
                  and previous.get("scope_id") == scope_id
                  and previous.get("embeddings") == bool(embedding_service)
                  and previous.get("state") in {"running", "paused", "partial"})
    if resume:
        job = previous
        end = job["end"]  # Keep a resumed snapshot fixed; a later run catches new messages.
        if scope is None:
            known = {row["id"] for row in job["sessions"]}
            job["sessions"].extend({**row, "status": "pending"} for row in sessions if row["id"] not in known)
    else:
        job = {"version": 2, "start": start, "end": end, "embeddings": bool(embedding_service), "scope_id": scope_id,
               "owner": None, "sessions": [{**row, "status": "pending"} for row in sessions]}
    job["new_sessions"] = len(set(available) - set(scope["known_ids"])) if scope is not None else 0
    job.update(state="running", max_messages=max_messages)
    key = batch_key(client)

    def save():
        db.set_setting(key, json.dumps(job, ensure_ascii=False))

    def emit(index=None):
        event = {**batch_summary(job), "resumed": resume}
        if index is None:
            event["sessions"] = [dict(row) for row in job["sessions"]]
        else:
            event.update(index=index, row=dict(job["sessions"][index]))
        on_update(event)

    save()
    emit()
    changed = False
    for index, row in enumerate(job["sessions"]):
        if should_cancel():
            job["state"] = "paused"
            break
        if row["status"] in DONE:
            continue
        current_type = available.get(row["id"], {}).get("category", "unknown")
        scope_unchanged = scope is None or (row["id"] in available and current_type == scope["types"][row["id"]])
        if scope_unchanged and row.get("last_activity") and row["last_activity"] < start:
            row.update(status="skipped", error="早于起始日期", seconds=0)
            save()
            emit(index)
            continue
        row.update(status="running", error="", received=0, updated_units=0, reused_units=0, failed_messages=0)
        save()
        emit(index)
        started = time.perf_counter()
        stop = False
        try:
            if should_cancel():
                raise SyncCancelled("同步已取消")
            if scope is not None:
                live = available.get(row["id"])
                if live is None or live.get("category", "unknown") != scope["types"][row["id"]]:
                    raise WeFlowError("会话已不可见或类型改变，请重新选择范围", code="scope_changed")
            identity = client.identity(row["id"], end)
            if job["owner"] and job["owner"] != identity["owner"]:
                raise WeFlowError("同步期间微信账号发生变化，已停止批量任务", code="account_mismatch")
            job["owner"] = identity["owner"]
            value = sync_session(db, client, row["id"], start=start, end=end, incremental=incremental,
                                 max_messages=max_messages, embedding_service=embedding_service,
                                 should_cancel=should_cancel, progress=progress, _identity=identity)
            row.update({name: value[name] for name in ("received", "updated_units", "reused_units")})
            row["failed_messages"] = value.get("failed_messages", 0)
            row["status"] = "partial" if row["failed_messages"] else "done" if value["received"] else "empty"
            if row["failed_messages"]:
                row["error"] = "；".join(f"{ISSUE_LABELS.get(issue['code'], '消息异常')} {issue['count']} 条"
                                       for issue in value["issue_counts"])
            changed = True
        except Exception as exc:
            if isinstance(exc, SyncCancelled) or should_cancel():
                row.update(status="pending", error="已暂停，等待继续")
                job["state"], stop = "paused", True
            else:
                message = str(exc) if isinstance(exc, (WeFlowError, ValueError)) else f"处理失败（{type(exc).__name__}）"
                row.update(status="failed", error=message[:300])
                stop = isinstance(exc, WeFlowError) and (exc.status in {401, 403, 423, 429, 502, 503, 504}
                        or exc.code in {"transport_unavailable", "account_mismatch"})
        row["seconds"] = round(time.perf_counter() - started, 3)
        save()
        emit(index)
        if stop:
            break
    if job["state"] != "paused":
        job["state"] = "completed" if all(row["status"] in DONE for row in job["sessions"]) else "partial"
    save()
    emit()
    return {**batch_summary(job), "changed": changed, "resumed": resume, "job": job}
