"""Opt-in live contract probe. Credentials and message contents never enter the report."""
from __future__ import annotations

import argparse
import html
import importlib
import importlib.util
import json
import os
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit

from localmind.chats import ChatStore
from localmind.config import default_db_path
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.models import SearchQuery
from localmind.reranker import RerankerService
from localmind.searcher import SearchService
from localmind.weflow import WeFlowClient, WeFlowError, decode_messages, sync_session
from localmind.weflow_protocol import integer


def shape(value, depth=0):
    if depth > 7:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(key): shape(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return {"count": len(value), "item_shapes": [shape(item, depth + 1) for item in value[:2]]}
    return type(value).__name__


def recent_sample(client, session):
    # Alpha defaults to historical ascending rows, not the most recent page.
    payload = client.get("/api/v1/messages", talker=session, limit=200, offset=0, media="0", order="desc")
    client._check_response_account(payload)
    rows = payload.get("messages")
    if payload.get("talker") != session or not isinstance(rows, list) or len(rows) > 200 or type(payload.get("hasMore")) is not bool:
        raise WeFlowError("最近消息样本结构不兼容")
    if any(not isinstance(row, dict) or client.protocol == "alpha" and row.get("conversation_id") != session for row in rows):
        raise WeFlowError("最近消息样本包含不同会话")
    timestamps = [integer(row.get("sent_at_unix_ms"), "sent_at_unix_ms", minimum=1) // 1000
                  if "sent_at_unix_ms" in row else integer(row.get("createTime"), "createTime", minimum=1) for row in rows]
    if any(a < b for a, b in zip(timestamps, timestamps[1:])):
        raise WeFlowError("接口未按最近消息倒序返回，不能将默认第一页当作最近消息验证")
    return payload, timestamps


def verify_search(client, session, report):
    """Bounded, isolated smoke test, not a relevance benchmark or persistent import."""
    for key in ("e2e", "sync", "retrieval", "attachments", "failure"):
        report.pop(key, None)
    started = time.perf_counter()
    try:
        sessions = client.sessions()
    except WeFlowError as exc:
        report["connection_error"] = {"http_status": exc.status, "code": exc.code}
        print(json.dumps({"connection_error": report["connection_error"]}), flush=True)
        raise
    report.pop("connection_error", None)
    report["session_listing"] = {"count": len(sessions), "selected_present": any(row["id"] == session for row in sessions),
                                 "seconds": round(time.perf_counter() - started, 3)}
    print(json.dumps({"session_listing": report["session_listing"]}), flush=True)
    print("live_phase: sample", flush=True)
    started = time.perf_counter()
    sample, timestamps = recent_sample(client, session)
    rows = sample.get("messages", [])
    if not rows:
        report["e2e"] = {"status": "no_messages"}
        return
    start, end = min(timestamps), max(timestamps)
    if sample.get("hasMore"):
        start += 1  # Exclude the potentially partial oldest second.
    report["sample"] = {"count": len(rows), "order": "desc", "seconds": round(time.perf_counter() - started, 3),
                        "file_label_rows": sum(bool(row.get("text", row.get("content", "")).startswith("[文件]")) for row in rows),
                        "nonempty_refs_shapes": [shape(row["refs"]) for row in rows if row.get("refs")][:2]}
    report["attachment_diagnostics"] = attachment_diagnostics(rows)
    report["media_contract"] = inspect_media_rows(rows)
    print(json.dumps({"attachment_diagnostics": report["attachment_diagnostics"]}, ensure_ascii=False), flush=True)
    print(json.dumps({"sample": report["sample"]}, ensure_ascii=False), flush=True)
    if start > end:
        raise WeFlowError("样本集中在同一秒，无法建立有完整时间边界的小样本，请换一个会话")
    cache = None
    if default_db_path().exists():
        with sqlite3.connect(default_db_path().as_uri() + "?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='model_cache_dir'").fetchone()
            cache = row[0] if row else None
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    embedding = EmbeddingService(cache_dir=cache, local_files_only=True)
    print("live_phase: local_model", flush=True)
    model = embedding.active_model_name
    report["model"] = {"name": model, "device": embedding.device, "load_source": embedding.load_source,
                       "load_seconds": round(embedding.load_seconds, 3)}
    if model == "localmind-hash-embedding":
        raise WeFlowError("本地 BGE 模型不可用，不能把哈希回退当作语义验证通过")
    with tempfile.TemporaryDirectory(prefix="localmind-alpha-verify-") as temp:
        db = LocalMindDB(Path(temp) / "index.db")
        try:
            phases = {}
            for phase in ("initial", "repeat", "incremental"):
                print("live_phase: " + phase, flush=True)
                started = time.perf_counter()
                value = sync_session(db, client, session, start=start, end=end, max_messages=200,
                                     incremental=phase == "incremental", embedding_service=embedding)
                phases[phase] = {key: value[key] for key in ("received", "total", "updated_units", "reused_units")}
                phases[phase]["seconds"] = round(time.perf_counter() - started, 3)
                if phase != "initial" and value["updated_units"] != 0:
                    raise AssertionError("unchanged_sync_reembedded")
            report["sync"] = phases
            db_rows = db.conn.execute("SELECT id, body FROM chat_messages ORDER BY id").fetchall()
            checks = []
            for row in db_rows:
                words = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{6,}", row["body"])
                if words:
                    checks.append((int(row["id"]), max(words, key=len)[:14]))
                if len(checks) == 10:
                    break
            report["retrieval"] = {}
            for mode in ("keyword", "semantic", "hybrid"):
                service = SearchService(db, embedding_service=embedding, reranker_service=RerankerService(enabled=False), retrieval_mode=mode)
                hits, durations = 0, []
                for identity, query in checks:
                    started = time.perf_counter()
                    results = service.search(SearchQuery(query, scope="messages", limit=5))
                    durations.append((time.perf_counter() - started) * 1000)
                    hits += any(result.chat_message_id == identity for result in results)
                report["retrieval"][mode] = {"excerpt_smoke_queries": len(checks), "source_message_top5_hits": hits,
                                             "mean_ms": round(sum(durations) / max(1, len(durations)), 2)}
            attachments = ChatStore(db).attachments()
            report["attachments"] = {"count": len(attachments), "filename_top5_hits": 0, "physical_file_link_tested": False,
                                      "sample_verdict": "recognized" if attachments else "not_recognized_in_sample"}
            for attachment in attachments[:10]:
                found = service.search(SearchQuery(attachment["name"], scope="chat_files", limit=5))
                report["attachments"]["filename_top5_hits"] += any(r.attachment_id == attachment["id"] for r in found)
            report["attachments"]["queries"] = min(10, len(attachments))
            if report["attachments"]["filename_top5_hits"] != report["attachments"]["queries"]:
                raise AssertionError("attachment_filename_retrieval_failed")
            report["e2e"] = {"status": "completed", "messages": len(db_rows), "temporary_index_removed": False,
                             "not_a_fuzzy_relevance_evaluation": True}
        finally:
            db.close()
    report["e2e"]["temporary_index_removed"] = True


def inspect_media(client, session):
    rows = recent_sample(client, session)[0]["messages"]
    return inspect_media_rows(rows)


def inspect_media_rows(rows):
    summary = {}
    for row in rows:
        kind = row.get("kind", "legacy")
        kind = kind if isinstance(kind, str) and re.fullmatch(r"[a-zA-Z_0-9.-]{1,40}", kind) else "other"
        group = summary.setdefault(kind, {"count": 0, "text_shapes": {}, "refs": []})
        group["count"] += 1
        content = row.get("text", row.get("content", ""))
        text_shape = "empty" if not content else "xml" if content.lstrip().startswith("<") else "file_label" if content.startswith("[文件]") else "plain"
        extension = Path(content).suffix.lower() if len(content) < 240 and "\n" not in content else ""
        if re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
            text_shape += ":filename_extension"
        group["text_shapes"][text_shape] = group["text_shapes"].get(text_shape, 0) + 1
        for ref in row.get("refs", []):
            if not isinstance(ref, dict):
                continue
            uri = urllib.parse.urlsplit(str(ref.get("uri", "")))
            entry = {"kind": ref.get("kind"), "mime": ref.get("mime_type"), "uri_scheme": uri.scheme,
                     "uri_has_filename_extension": bool(re.fullmatch(r"\.[a-zA-Z0-9]{1,8}", Path(uri.path).suffix))}
            if entry not in group["refs"] and len(group["refs"]) < 8:
                group["refs"].append(entry)
    return summary


def attachment_diagnostics(rows):
    """Only structural counters; never emit filenames, IDs, XML text or URIs."""
    groups, timestamps = {}, []
    for row in rows:
        source_type = row.get("source_type", row.get("localType"))
        source_type = str(source_type) if type(source_type) is int or isinstance(source_type, str) and re.fullmatch(r"[0-9]{1,20}", source_type) else "other"
        group = groups.setdefault(source_type, Counter())
        group["rows"] += 1
        content = row.get("text", row.get("content", "")) or ""
        raw = row.get("rawContent") or content
        if isinstance(raw, str) and len(raw) <= 128000:
            raw = html.unescape(raw) if "<" not in raw else raw
            start = raw.find("<")
            if start >= 0 and "<!DOCTYPE" not in raw.upper() and "<!ENTITY" not in raw.upper():
                try:
                    root = ET.fromstring(raw[start:])
                    group["valid_xml"] += 1
                    apps = [root] if root.tag == "appmsg" else root.findall(".//appmsg")
                    for app in apps:
                        app_type = (app.findtext("type") or "").strip()
                        app_type = app_type if re.fullmatch(r"[0-9]{1,8}", app_type) else "other"
                        group["appmsg_type_" + app_type] += 1
                        group["appmsg_title_present"] += bool((app.findtext("title") or "").strip())
                        group["appattach_present"] += app.find("appattach") is not None
                        group["recorditem_present"] += app.find("recorditem") is not None
                    group["nested_file_items"] += len(root.findall(".//dataitem[@datatype='8']"))
                except ET.ParseError:
                    group["malformed_xml"] += 1
        for ref in row.get("refs", []):
            if isinstance(ref, dict):
                kind = ref.get("kind")
                kind = kind if kind in {"image", "voice", "video", "file", "document"} else "other"
                group["ref_kind_" + kind] += 1
                group["ref_file_name_present"] += any(isinstance(ref.get(key), str) and bool(ref[key]) for key in ("file_name", "filename", "name", "title"))
        try:
            decoded = decode_messages([row])[0]
            group["recognized_attachments"] += len(decoded.attachments)
            group["recognized_explicit"] += sum(att.kind == "explicit" for att in decoded.attachments)
            timestamps.append(datetime.fromisoformat(decoded.sent_at.replace("Z", "+00:00")))
        except (WeFlowError, ValueError, TypeError):
            group["decode_errors"] += 1
    return {"source_types": groups, "date_range_utc": [min(timestamps).date().isoformat(), max(timestamps).date().isoformat()] if timestamps else [],
            "descending_timestamps": all(a >= b for a, b in zip(timestamps, timestamps[1:]))}


def load_checks():
    spec = importlib.util.spec_from_file_location("localmind_live_checks", Path(__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5031")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--control", type=Path, help="Opt-in developer loop; JSON commands run/stop, never credentials")
    args = parser.parse_args()
    if args.control and args.control.exists():
        raise ValueError("control path must not exist")
    app = QApplication.instance() or QApplication([])
    token, accepted = QInputDialog.getText(None, "LocalMind 本地接口验证", "请输入 Alpha API Token（仅保留在内存中，不写入报告）", QLineEdit.EchoMode.Password)
    if not accepted or not token.strip():
        print("cancelled", flush=True)
        return
    client = WeFlowClient(args.url, token, timeout=150)
    report = {"kind": "live-contract-probe", "contains_chat_text": False, "endpoints": {}}

    def probe(path, **params):
        url = client.base_url + path + ("?" + urllib.parse.urlencode(params) if params else "")
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token.strip(), "Accept": "application/json"})
        started = time.perf_counter()
        try:
            with client.opener.open(request, timeout=150) as response:
                status, data = response.status, response.read(8 * 1024 * 1024)
        except urllib.error.HTTPError as exc:
            status, data = exc.code, exc.read(65536)
        payload = json.loads(data)
        entry = {"status": status, "seconds": round(time.perf_counter() - started, 3), "schema": shape(payload)}
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            code = error["code"]
            if len(code) < 100 and all(c.isalnum() or c in "_.-" for c in code):
                entry["error_code"] = code
        report_path = "/api/v1/sessions/:id/messages" if path.startswith("/api/v1/sessions/") else path
        report["endpoints"][report_path] = entry
        print(json.dumps({"path": report_path, **entry}, ensure_ascii=False), flush=True)
        return payload

    try:
        probe("/health")
        accounts = probe("/api/v1/accounts")
        rows = accounts.get("accounts", [])
        params = {}
        if rows:
            labels = [str(row.get("display_name") or row.get("displayName") or row.get("wxid") or row.get("id") or row.get("account_id")) for row in rows]
            label, accepted = QInputDialog.getItem(None, "LocalMind 本地接口验证", "选择验证账号", labels, 0, False)
            if not accepted:
                return
            row = rows[labels.index(label)]
            account = row.get("id") or row.get("account_id") or row.get("accountId") or row.get("wxid")
            if account:
                params["account"] = account
                client.account_id = account
        sessions = probe("/api/v1/sessions", limit=30, **params)
        rows = sessions.get("sessions", sessions.get("items", []))
        if rows:
            labels = [f"{i + 1}. " + str(row.get("displayName") or row.get("display_name") or row.get("name") or row.get("username") or row.get("id")) for i, row in enumerate(rows)]
            prompt = "选择含文件消息的会话（读取最近最多 200 条，临时索引测试后删除）" if args.e2e else "选择一个会话（仅读取最多 20 条消息做结构验证）"
            label, accepted = QInputDialog.getItem(None, "LocalMind 本地接口验证", prompt, labels, 0, False)
            if not accepted:
                return
            row = rows[labels.index(label)]
            session = row.get("username") or row.get("id")
            if session:
                probe("/api/v1/messages", talker=session, limit=20, offset=0, media="0", **params)
                probe("/api/v1/sessions/" + urllib.parse.quote(session, safe="") + "/messages", limit=1, **params)
                if args.e2e:
                    report["media_contract"] = inspect_media(client, session)
                    print(json.dumps({"media_contract": report["media_contract"]}, ensure_ascii=False), flush=True)
                    while True:
                        if args.control:
                            print("live_phase: waiting_for_local_run_or_stop", flush=True)
                            while not args.control.exists():
                                app.processEvents()
                                time.sleep(0.2)
                            command = json.loads(args.control.read_text(encoding="utf-8"))["command"]
                            args.control.unlink()
                            if command == "stop":
                                break
                            if command not in {"run", "diagnose"}:
                                continue
                            from localmind import weflow, weflow_protocol
                            importlib.reload(weflow_protocol)
                            module = importlib.reload(weflow)
                            globals()["sync_session"] = module.sync_session
                            globals()["WeFlowError"] = module.WeFlowError
                            client = module.WeFlowClient(args.url, token, account=params.get("account"))
                        checks = load_checks()
                        calls = []
                        get = client.get

                        def timed_get(path, **query):
                            started = time.perf_counter()
                            try:
                                return get(path, **query)
                            finally:
                                route = "/api/v1/sessions/:id/messages" if path.startswith("/api/v1/sessions/") else path
                                calls.append({"route": route, "seconds": round(time.perf_counter() - started, 3)})

                        client.get = timed_get
                        try:
                            if args.control and command == "diagnose":
                                rows = checks.recent_sample(client, session)[0]["messages"]
                                report["attachment_diagnostics"] = checks.attachment_diagnostics(rows)
                            else:
                                checks.verify_search(client, session, report)
                            report.pop("failure", None)
                        except Exception as exc:
                            report["failure"] = {"type": type(exc).__name__}
                            print("live_failure_type: " + type(exc).__name__, flush=True)
                        finally:
                            client.get = get
                            report["request_timings"] = calls
                        print(json.dumps({key: value for key, value in report.items() if key != "endpoints"}, ensure_ascii=False), flush=True)
                        if not args.control:
                            break
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__}
        if isinstance(exc, WeFlowError):
            report["failure"]["message"] = str(exc)
        print(json.dumps(report["failure"], ensure_ascii=False), flush=True)
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        token = ""
        client._token = ""
    return 1 if report.get("failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())
