from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Callable

from .chats import ChatMessage, ChatStore, MAX_IMPORT_BYTES, MAX_MESSAGES, digest
from .models import utc_now_iso
from .weflow_protocol import (
    ALPHA_SESSION_PAGE_SIZE, MAX_SESSION_PAGES, MESSAGE_PAGE_SIZE,
    WeFlowError, _api_error, _xml_attachment, decode_messages, integer, session_page,
)

REFERENCE_REVISION = "2f23e72cebf7f309a0f4d129307d16d7b0cd41c3"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class SyncCancelled(WeFlowError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise WeFlowError("WeFlow 返回重定向，已阻止向其它地址发送 Token")


def local_base_url(value: str) -> str:
    try:
        url = urllib.parse.urlsplit(value.strip())
        host = url.hostname
        if host == "localhost":
            host = "127.0.0.1"
        if url.scheme != "http" or not host or not ipaddress.ip_address(host).is_loopback:
            raise ValueError()
        if url.username or url.password or url.query or url.fragment or url.path.rstrip("/") not in {"", "/api/v1"}:
            raise ValueError()
        port = 5031 if url.port is None else url.port
        if not 1 <= port <= 65535:
            raise ValueError()
        return f"http://{'[' + host + ']' if ':' in host else host}:{port}"
    except ValueError as exc:
        raise WeFlowError("仅允许本机 HTTP 地址，例如 http://127.0.0.1:5031；Token 请单独填写") from exc


class WeFlowClient:
    def __init__(self, base_url="http://127.0.0.1:5031", token="", *, timeout=150.0, account: str | None = None):
        self.base_url = local_base_url(base_url)
        if any(not 33 <= ord(char) <= 126 for char in token.strip()):
            raise WeFlowError("Token 必须是可打印 ASCII 字符，不能包含空白或换行")
        self._token = token.strip()
        self.timeout = timeout
        self.account_id = account or None
        self.available_accounts: list[dict] = []
        self._accounts_checked = False
        self.protocol = "weflow"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def get(self, path: str, **params) -> dict:
        routes = {"/health", "/api/v1/health", "/api/v1/accounts", "/api/v1/sessions", "/api/v1/messages", "/api/v1/group-members"}
        if path not in routes and not re.fullmatch(r"/api/v1/sessions/[^/?#]+/messages", path):
            raise WeFlowError("仅允许已支持的只读接口路径")
        if "access_token" in params or "account_id" in params:
            raise WeFlowError("Token 只能通过请求头传递，账号仅使用 account 参数")
        url = self.base_url + path
        if self.account_id and path not in {"/health", "/api/v1/health", "/api/v1/accounts"}:
            if "account" in params and params["account"] != self.account_id:
                raise WeFlowError("请求账号与当前选择不一致，已停止读取")
            params.setdefault("account", self.account_id)
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = urllib.request.Request(url, headers=headers, method="GET")
        return self._request_json(request, timeout=min(self.timeout, 8) if path in {"/health", "/api/v1/health"} else self.timeout)

    def _request_json(self, request, *, timeout=None) -> dict:
        try:
            with self.opener.open(request, timeout=self.timeout if timeout is None else timeout) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise WeFlowError("单页响应超过 8 MB，请缩小每页消息数", code="response_limit")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise WeFlowError("WeFlow 返回的 JSON 结构不兼容")
            if value.get("success") is False or value.get("status") in ("error", "failed") or isinstance(value.get("error"), dict):
                raise _api_error(200, value)
            return value
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read(65536))
            except (ValueError, OSError):
                payload = None
            raise _api_error(exc.code, payload) from None
        except http.client.HTTPException:
            raise WeFlowError("API 响应传输中断，未接受不完整数据；请重试") from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError):
            raise WeFlowError("无法连接本机 WeFlow，或请求超时；请开启 API 服务后重试", code="transport_unavailable") from None
        except (json.JSONDecodeError, UnicodeError):
            raise WeFlowError("WeFlow 返回的内容不是有效 JSON") from None

    def accounts(self) -> list[dict]:
        if self._accounts_checked:
            return self.available_accounts
        try:
            payload = self.get("/api/v1/accounts")
        except WeFlowError as exc:
            if exc.status != 404:
                raise
            if self.account_id:
                raise WeFlowError("旧版 WeFlow 不支持选择账号，请清空账号选择后重新连接") from None
            self._accounts_checked = True
            return []
        rows = payload.get("accounts")
        if not isinstance(rows, list):
            raise WeFlowError("账号列表接口结构不兼容")
        accounts = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise WeFlowError("账号列表缺少稳定 ID")
            accounts.append({"id": row["id"], "name": str(row.get("display_name") or row.get("displayName") or row["id"]),
                             "ready": row.get("ready", True) is True and row.get("enabled", True) is True,
                             "selected": row.get("selected") is True})
        self.available_accounts = accounts
        selected = self.account_id or payload.get("selected_account_id") or next((r["id"] for r in accounts if r["selected"]), None)
        if not selected and len(accounts) == 1:
            selected = accounts[0]["id"]
        if not selected:
            raise WeFlowError("未选中微信账号，请在 Alpha 选择已连接的账号后重试")
        match = next((r for r in accounts if r["id"] == selected), None)
        if not match or not match["ready"]:
            raise WeFlowError("选定的 Alpha 账号未就绪，请先连接该账号的微信数据")
        self.account_id = selected
        self.protocol = "alpha"
        self._accounts_checked = True
        return accounts

    def _check_response_account(self, payload: dict):
        if self.protocol == "alpha" and payload.get("account_id") != self.account_id:
            raise WeFlowError("接口返回了不同账号的数据，已停止同步", code="account_mismatch")

    def sessions(self, keyword="", *, should_cancel=lambda: False, progress=lambda _: None) -> list[dict]:
        health = self.get("/health")
        if health.get("status") != "ok":
            raise WeFlowError("WeFlow 健康检查未通过")
        if health.get("database_open_reported") is False:
            raise WeFlowError("Alpha 服务已启动，但微信数据库尚未连接")
        self.accounts()
        offset, seen = 0, {}
        for _ in range(MAX_SESSION_PAGES):
            if should_cancel():
                raise SyncCancelled("会话读取已取消")
            page_size = ALPHA_SESSION_PAGE_SIZE if self.protocol == "alpha" else MESSAGE_PAGE_SIZE
            payload = self.get("/api/v1/sessions", limit=page_size, offset=offset, keyword=keyword)
            self._check_response_account(payload)
            page = session_page(payload, offset, alpha=self.protocol == "alpha")
            before = len(seen)
            for row in page.sessions:
                seen[row["id"]] = row
            progress(f"已读取 {len(seen)} 个会话")
            if should_cancel():
                raise SyncCancelled("会话读取已取消")
            if page.next_offset is None:
                return list(seen.values())
            if len(seen) == before:
                raise WeFlowError("会话分页未推进，已停止读取；请缩小会话名称范围")
            offset = page.next_offset
        raise WeFlowError("会话分页超过安全上限，请按会话名称缩小范围")

    def identity(self, session_id: str, end: int) -> dict:
        self.accounts()
        # An empty one-second range can force Alpha to scan the entire history.
        # Read one bounded row for metadata; it is never ingested by this method.
        payload = self.get("/api/v1/sessions/" + urllib.parse.quote(session_id, safe="") + "/messages", limit=1, end=end)
        meta = payload.get("meta", {})
        if not isinstance(meta, dict) or not isinstance(payload.get("members", []), list):
            raise WeFlowError("WeFlow 账号元数据结构不兼容")
        owner = str(meta.get("ownerId") or "").strip()
        if not owner:
            raise WeFlowError("WeFlow 未返回 ownerId，无法安全区分账号；请升级到支持 ChatLab Pull 的版本")
        return {"owner": owner, "name": str(meta.get("name") or session_id), "members": payload.get("members", [])}

    def fetch(self, session_id: str, *, start: int, end: int, max_messages=MAX_MESSAGES,
              should_cancel: Callable[[], bool] = lambda: False, progress: Callable[[str], None] = lambda _: None,
              _identity: dict | None = None) -> tuple[dict, list[ChatMessage]]:
        if not session_id or not 0 <= start <= end or not 1 <= max_messages <= MAX_MESSAGES:
            raise WeFlowError("同步范围或消息上限无效")
        identity = _identity or self.identity(session_id, end)
        names = {str(m.get("platformId")): str(m.get("groupNickname") or m.get("accountName") or m.get("platformId")) for m in identity["members"] if isinstance(m, dict)}
        names[identity["owner"]] = "我"
        names.setdefault(session_id, identity["name"])
        if session_id.endswith("@chatroom") and self.protocol != "alpha":
            try:
                members = self.get("/api/v1/group-members", chatroomId=session_id, includeMessageCounts="0").get("members", [])
                for member in members:
                    names[str(member["wxid"])] = str(member.get("groupNickname") or member.get("displayName") or member["wxid"])
            except (WeFlowError, KeyError, TypeError):
                pass
        offset, received, seen_pages = 0, 0, set()
        collected: dict[str, ChatMessage] = {}
        collected_keys: set[str] = set()
        issues: dict[str, dict] = {}
        page_size, collected_bytes = MESSAGE_PAGE_SIZE, 0
        api_identities = {}
        while True:
            if should_cancel():
                raise SyncCancelled("同步已取消，检查点未推进")
            # Keep the time window fixed; moving both since and offset can skip rows.
            limit = min(page_size, max_messages - received)
            try:
                payload = self.get("/api/v1/messages", talker=session_id, start=start, end=end,
                                   offset=offset, limit=limit, media="0")
            except WeFlowError as exc:
                if exc.code != "response_limit" or limit <= 1:
                    raise
                page_size = max(1, limit // 2)
                progress(f"单页内容较大，每页调整为 {page_size} 条后重试")
                continue
            self._check_response_account(payload)
            if payload.get("talker") != session_id or not isinstance(payload.get("messages"), list) or type(payload.get("hasMore")) is not bool:
                raise WeFlowError("消息分页响应不兼容，检查点未推进")
            rows = payload["messages"]
            if self.protocol == "alpha" and any(not isinstance(row, dict) or row.get("conversation_id") != session_id for row in rows):
                raise WeFlowError("接口返回了不同会话的消息，已停止同步")
            if len(rows) > limit:
                raise WeFlowError("接口返回条数超过请求上限")
            page_issues = []
            decoded = decode_messages(rows, names, issues=page_issues)
            if self.protocol == "alpha":
                from .wechat_files import api_identity
                for raw in rows:
                    identity_ref = api_identity(raw)
                    if identity_ref:
                        if identity_ref[0] in api_identities and api_identities[identity_ref[0]] != identity_ref[1]:
                            raise WeFlowError("同一消息返回了不同的 API 标识，已停止同步", code="message_identity_conflict")
                        api_identities[identity_ref[0]] = identity_ref[1]
            partial_keys = {issue["message_key"] for issue in page_issues if issue.get("partial") is True}
            page_messages = {digest(msg.external_id): msg for msg in decoded}
            for issue in page_issues:
                moment = int(datetime.fromisoformat(issue["sent_at"].replace("Z", "+00:00")).timestamp())
                if not start <= moment <= end:
                    raise WeFlowError("接口未遵守时间范围，已停止同步")
                key = issue["message_key"]
                repeated_partial = (key in partial_keys and key in page_messages and key in issues
                                    and collected.get(page_messages[key].external_id) == page_messages[key]
                                    and issues[key] == issue)
                if key in collected_keys and not repeated_partial:
                    raise WeFlowError("同一消息 ID 返回了冲突的正文，已停止同步")
                issues[issue["message_key"]] = issue
            for msg in decoded:
                moment = int(datetime.fromisoformat(msg.sent_at.replace("Z", "+00:00")).timestamp())
                if not start <= moment <= end:
                    raise WeFlowError("接口未遵守时间范围，已停止同步")
                if msg.external_id in collected and collected[msg.external_id] != msg:
                    raise WeFlowError("不同消息使用相同 ID，已停止同步以避免覆盖")
                message_key = digest(msg.external_id)
                if message_key in issues and message_key not in partial_keys:
                    raise WeFlowError("同一消息 ID 返回了冲突的正文，已停止同步")
                if msg.external_id not in collected:
                    collected_bytes += msg.text_bytes()
                    if collected_bytes > MAX_IMPORT_BYTES:
                        raise WeFlowError("本次规范化消息正文超过 20 MB，请缩小同步范围", code="conversation_limit")
                collected[msg.external_id] = msg
                collected_keys.add(message_key)
            received += len(rows)
            progress(f"已读取 {received} 条消息" + (f" · {len(issues)} 条待重试" if issues else ""))
            if not payload["hasMore"]:
                break
            page_key = digest(json.dumps([m.external_id for m in decoded] + [i["message_key"] for i in page_issues]))
            cursor = payload.get("nextOffset") if self.protocol == "alpha" else payload.get("nextOffset", offset + len(rows))
            next_offset = integer(cursor, "消息分页游标")
            if not rows or page_key in seen_pages or received >= max_messages or next_offset <= offset:
                raise WeFlowError("分页停滞或超过本次上限；请缩小首次同步时间范围，检查点未推进")
            seen_pages.add(page_key)
            offset = next_offset
        if should_cancel():
            raise SyncCancelled("同步已取消，检查点未推进")
        # Alpha requests pin an account and validate every response. Legacy
        # WeFlow has only a global active account, so it needs the second guard.
        if self.protocol != "alpha":
            after = self.identity(session_id, end)
            if after["owner"] != identity["owner"]:
                raise WeFlowError("同步期间 WeFlow 切换了微信账号，未写入索引")
        return {**identity, "issues": list(issues.values()), "received_rows": received,
                "api_identities": api_identities}, list(collected.values())


def sync_session(db, client: WeFlowClient, session_id: str, *, start: int, end: int | None = None,
                 incremental=True, max_messages=MAX_MESSAGES, embedding_service=None,
                 should_cancel=lambda: False, progress=lambda _: None, _identity: dict | None = None) -> dict:
    end = int(time.time()) if end is None else end
    identity = _identity or client.identity(session_id, end)
    source_key = "weflow:" + digest("\0".join([client.base_url, identity["owner"], session_id]))
    checkpoint_key = "weflow_checkpoint:" + digest(source_key)
    checkpoint = json.loads(db.get_setting(checkpoint_key, "{}") or "{}")
    requested_start = start
    if incremental and checkpoint.get("requested_start") == requested_start:
        start = max(start, int(checkpoint.get("end", start)) - 300)
        if "retry_start" in checkpoint:
            start = max(requested_start, min(start, int(checkpoint["retry_start"])))
    identity_after, messages = client.fetch(session_id, start=start, end=end, max_messages=max_messages,
                                           should_cancel=should_cancel, progress=progress, _identity=identity)
    if identity_after["owner"] != identity["owner"]:
        raise WeFlowError("微信账号变化，未写入索引")
    if should_cancel():
        raise SyncCancelled("同步已取消，检查点未推进")
    store = ChatStore(db)
    if messages:
        result = store.ingest(platform="wechat", account=identity["owner"], conversation=identity_after["name"],
                              source_key=source_key, kind="weflow_api", messages=messages,
                              embedding_service=embedding_service, should_cancel=should_cancel)
    else:
        result = {"received": 0, "total": 0, "updated_units": 0, "reused_units": 0}
    # Index writes are idempotent. Persist the retry ledger and checkpoint together
    # only after ingestion; a crash before this point safely re-reads the range.
    now = utc_now_iso()
    issues = identity_after.get("issues", [])
    with db.conn:
        from .wechat_files import save_api_identities
        save_api_identities(db, source_key, client.base_url, getattr(client, "account_id", None),
                            identity_after.get("api_identities", {}))
        db.conn.executemany("DELETE FROM chat_sync_issues WHERE source_key=? AND message_key=?",
                            [(source_key, digest(msg.external_id)) for msg in messages])
        for issue in issues:
            db.conn.execute("""INSERT INTO chat_sync_issues(source_key,message_key,sent_at,code,chars,first_seen,last_seen)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(source_key,message_key) DO UPDATE SET
                sent_at=excluded.sent_at,code=excluded.code,chars=excluded.chars,last_seen=excluded.last_seen,
                attempts=chat_sync_issues.attempts+1""", (source_key, issue["message_key"], issue["sent_at"],
                issue["code"], issue["chars"], now, now))
        outstanding = [dict(row) for row in db.conn.execute(
            "SELECT code,COUNT(*) AS count FROM chat_sync_issues WHERE source_key=? "
            "AND unixepoch(sent_at) BETWEEN ? AND ? GROUP BY code", (source_key, requested_start, end))]
        failed = sum(row["count"] for row in outstanding)
        new_checkpoint = {"requested_start": requested_start, "end": end}
        if failed:
            # Do not certify a range with gaps. A later retry includes the failed
            # window even when it lies outside the usual five-minute overlap.
            new_checkpoint = {"requested_start": requested_start,
                              "end": checkpoint.get("end", requested_start) if checkpoint.get("requested_start") == requested_start else requested_start,
                              "retry_start": start, "incomplete": True}
        db.conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (checkpoint_key, json.dumps(new_checkpoint)))
        source = db.conn.execute("SELECT id FROM chat_sources WHERE source_key=?", (source_key,)).fetchone()
        if source:
            db.conn.execute("INSERT INTO chat_coverage(source_id,start_epoch,end_epoch,state,issue_count,checked_at) VALUES(?,?,?,?,?,?)",
                            (source[0], start, end, "partial" if failed else "completed", failed, now))
            db.conn.execute("DELETE FROM chat_coverage WHERE source_id=? AND id NOT IN "
                            "(SELECT id FROM chat_coverage WHERE source_id=? ORDER BY id DESC LIMIT 64)", (source[0], source[0]))
    return {**result, "received": identity_after.get("received_rows", result["received"]),
            "indexed_messages": len(messages), "failed_messages": failed, "issue_counts": outstanding,
            "state": "partial" if failed else "completed",
            "start": start, "end": end, "owner": identity["owner"], "session_id": session_id}
