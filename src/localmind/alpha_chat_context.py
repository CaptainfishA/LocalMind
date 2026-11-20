"""Account-pinned, bounded, read-only Alpha context. No index writes or media downloads."""

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
import time
import urllib.request

from .chat_limits import validate_body
from . import __version__
from .chat_timeline import TimelineError, _rows, _validate, snapshot
from .chats import digest, normalize_time
from .search_filters import epoch
from .weflow import WeFlowError, SyncCancelled
from .weflow_protocol import decode_messages, integer

TOOLS = frozenset({"wechat.chat.get_messages", "wechat.chat.get_message_around", "wechat.chat.resolve_chat_history"})
DISCOVERY_TOOLS = TOOLS | {"wechat.chat.get_message_raw"}


def invalid(message="Alpha 聊天数据结构不兼容", code="context_shape"):
    return WeFlowError(message, code=code)


def anchor_record(path, result, discussion, message_id):
    with snapshot(path) as conn:
        refs = [_validate(conn, result, ref) for ref in discussion.refs]
        rows = _rows(conn, refs[0]["source_id"], result.source_query, extra=" AND m.id=?", params=(message_id,))
        remote = conn.execute("SELECT * FROM chat_api_messages WHERE message_id=?", (message_id,)).fetchone()
        if len(rows) != 1 or remote is None:
            raise invalid("此消息缺少可核对的 Alpha 身份，请先更新该会话索引", "context_identity_missing")
        row = rows[0]
        expected = "weflow:" + digest("\0".join((remote["base_url"], discussion.account, remote["conversation_id"])))
        if discussion.key[2] != expected:
            raise invalid("消息与 Alpha 数据源身份不一致", "context_identity")
        return {**dict(remote), "external_id": row["external_id"], "source_id": row["source_id"],
                "source_key": discussion.key[2], "body": row["body"], "forwarded": row["forwarded"],
                "revision": conn.execute("SELECT revision FROM index_revisions WHERE id=1").fetchone()[0]}


def link_known_attachments(path, discussion, anchor, batch):
    """Reuse only an exact local message identity and unchanged display content."""
    if batch.source.endswith("forward"):
        return batch
    output = []
    with snapshot(path) as conn:
        if conn.execute("SELECT revision FROM index_revisions WHERE id=1").fetchone()[0] != anchor["revision"]:
            raise TimelineError("索引已更新，请重新读取")
        for remote in batch.rows:
            local = conn.execute("""SELECT m.*,s.platform,s.account,s.source_key,a.timestamp_ms
                FROM chat_api_messages a JOIN chat_messages m ON m.id=a.message_id
                JOIN chat_sources s ON s.id=m.source_id
                WHERE a.base_url=? AND a.account_id=? AND a.conversation_id=? AND a.api_id=?
                AND m.source_id=? LIMIT 2""", (anchor["base_url"], anchor["account_id"], anchor["conversation_id"],
                                              remote["id"], anchor["source_id"])).fetchall()
            row = dict(remote)
            if len(local) == 1:
                msg = dict(local[0])
                display = msg["display_body"] if msg["display_body"] is not None else msg["body"]
                same_sender = bool(remote.get("sender_id")) and remote["sender_id"] == msg["sender_id"]
                same_sender |= bool(remote["sender"]) and remote["sender"] == msg["sender"]
                if (tuple(msg[k] for k in ("platform", "account", "source_key")) == discussion.key[:3]
                        and display == remote["body"] and same_sender and not remote.get("unsupported")
                        and remote["sent_epoch"] is not None
                        and (remote["timestamp_ms"] == msg["timestamp_ms"] if remote.get("timestamp_ms") is not None
                             else int(remote["sent_epoch"]) == int(msg["timestamp_ms"]) // 1000)):
                    row["attachments"] = [dict(a) for a in conn.execute("""SELECT a.id,a.ordinal,a.name,a.kind,a.file_id,
                        a.origin,a.context,f.path FROM chat_attachments a LEFT JOIN files f ON f.id=a.file_id
                        WHERE a.message_id=? ORDER BY a.ordinal LIMIT 300""", (msg["id"],))]
                    row["local_message"] = msg
            output.append(row)
    return replace(batch, rows=tuple(output))


@dataclass(frozen=True)
class ContextBatch:
    rows: tuple[dict, ...]
    notice: str
    source: str
    start: float | None
    end: float | None
    complete_range: bool = False
    skipped: int = 0


class ContextCache:
    def __init__(self, ttl=120, max_bytes=2_000_000):
        self.ttl, self.max_bytes = ttl, max_bytes
        self.entries = OrderedDict()

    def clear(self):
        self.entries.clear()

    def get(self, key):
        item = self.entries.get(key)
        if item is None:
            return None
        if time.monotonic() - item[0] > self.ttl:
            del self.entries[key]
            return None
        self.entries.move_to_end(key)
        return item[1]

    def put(self, key, value):
        size = len(json.dumps(asdict(value), ensure_ascii=False).encode())
        if size > self.max_bytes:
            return
        self.entries.pop(key, None)
        while self.entries and (len(self.entries) >= 4 or sum(v[2] for v in self.entries.values()) + size > self.max_bytes):
            self.entries.popitem(last=False)
        self.entries[key] = time.monotonic(), value, size


class AlphaChatContext:
    def __init__(self, client, *, cache=None, cancelled=lambda: False):
        self.client, self.cancelled = client, cancelled
        self.cache = cache if cache is not None else ContextCache()
        self.sequence = 0
        self.tools = {}
        self.catalog_names = set()
        self.read_count = 0
        self.diagnostics = {}

    def check(self):
        if self.cancelled():
            raise SyncCancelled("聊天读取已取消")

    def rpc(self, method, params):
        if method not in {"initialize", "tools/list", "tools/call"} or method == "tools/call" and params.get("name") not in TOOLS:
            raise invalid("此操作不在聊天只读白名单", "context_tool_denied")
        self.check()
        self.sequence += 1
        request = urllib.request.Request(self.client.base_url + "/mcp", method="POST",
            data=json.dumps({"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}).encode(),
            headers={"Authorization": "Bearer " + self.client._token, "Content-Type": "application/json", "Accept": "application/json"})
        value = self.client._request_json(request, timeout=min(self.client.timeout, 20))
        self.check()
        if value.get("jsonrpc") != "2.0" or type(value.get("id")) is not int or value["id"] != self.sequence:
            raise invalid("MCP 响应标识不匹配", "context_identity")
        result = value.get("result")
        if not isinstance(result, dict):
            raise invalid("MCP 未返回结构化结果", "context_unavailable")
        if result.get("isError"):
            raise invalid("Alpha 无法读取所选消息；本地记录未改变", "context_read_failed")
        return result

    def discover(self):
        value = self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                        "clientInfo": {"name": "LocalMind", "version": __version__}})
        if "tools" not in value.get("capabilities", {}):
            raise invalid("当前接口未提供聊天 MCP 能力", "context_unavailable")
        cursor, seen = None, set()
        for _ in range(10):
            page = self.rpc("tools/list", {"cursor": cursor} if cursor else {})
            tools = page.get("tools")
            if not isinstance(tools, list) or len(tools) > 1000:
                raise invalid()
            for tool in tools:
                name = tool.get("name") if isinstance(tool, dict) else None
                if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.]{1,128}", name):
                    self.catalog_names.add(name)
                if isinstance(tool, dict) and tool.get("name") in DISCOVERY_TOOLS:
                    if tool["name"] in self.tools:
                        raise invalid("MCP 工具目录重复")
                    self.tools[tool["name"]] = tool
            cursor = page.get("nextCursor")
            if cursor is None:
                return
            if not isinstance(cursor, str) or not 0 < len(cursor) <= 2048 or cursor in seen:
                raise invalid("MCP 工具目录分页无效")
            seen.add(cursor)
        raise invalid("MCP 工具目录超限")

    def call(self, name, **arguments):
        schema = self.tools.get(name, {}).get("inputSchema", {})
        args = {"account": self.client.account_id, **arguments}
        props, required = schema.get("properties", {}), schema.get("required", [])
        if (schema.get("type") != "object" or not isinstance(props, dict) or not isinstance(required, list)
                or any(not isinstance(k, str) for k in required) or not set(args) <= props.keys() or not set(required) <= args.keys()):
            raise invalid("安装版本的聊天参数不兼容", "context_unavailable")
        for key, value in args.items():
            spec = props[key]
            if not isinstance(spec, dict) or spec.get("type") not in {"string", "integer", "boolean"}:
                raise invalid("聊天参数类型不兼容", "context_unavailable")
            if type(value) is not {"string": str, "integer": int, "boolean": bool}[spec["type"]]:
                raise invalid("聊天参数类型不兼容", "context_unavailable")
            if "enum" in spec and value not in spec["enum"]:
                raise invalid("聊天参数范围不兼容", "context_unavailable")
            if type(value) is bool:
                continue
            lower, upper = (spec.get("minimum", 0), spec.get("maximum", 2**63 - 1)) if type(value) is int else (spec.get("minLength", 0), spec.get("maxLength", 4096))
            if type(lower) not in (int, float) or type(upper) not in (int, float) or not lower <= (value if type(value) is int else len(value)) <= upper:
                raise invalid("聊天参数超出接口范围", "context_unavailable")
        result = self.rpc("tools/call", {"name": name, "arguments": args}).get("structuredContent")
        if not isinstance(result, dict) or result.get("success") is False or result.get("error"):
            raise invalid("聊天接口缺少结构化结果")
        if result.get("account_id", self.client.account_id) != self.client.account_id:
            raise invalid("接口账号不匹配", "context_identity")
        conversation = arguments.get("username")
        for key in ("username", "conversation_id", "talker"):
            if key in result and result[key] != conversation:
                raise invalid("接口会话不匹配", "context_identity")
        return result

    def anchor_query(self, anchor):
        props = self.tools.get("wechat.chat.get_messages", {}).get("inputSchema", {}).get("properties", {})
        stamp = integer(anchor["timestamp_ms"], "锚点毫秒时间", minimum=1, maximum=9999999999999)
        if {"after_unix_ms", "before_unix_ms"} <= props.keys():
            # Both bounds are exclusive. Query the exact millisecond, not a
            # rounded second or an unbounded first page.
            arguments = {"after_unix_ms": str(stamp - 1), "before_unix_ms": str(stamp + 1)}
            self.diagnostics["bridge_time_format"] = "unix_ms_exclusive"
        elif {"start_time", "end_time"} <= props.keys():
            arguments = {"start_time": stamp // 1000, "end_time": stamp // 1000}
            self.diagnostics["bridge_time_format"] = "unix_seconds_inclusive"
        else:
            raise invalid("当前消息接口缺少可核对的时间范围参数", "context_unavailable")
        if "defer_media" in props:
            arguments["defer_media"] = True
        return self.call("wechat.chat.get_messages", username=anchor["conversation_id"], limit=20, **arguments)

    def around_query(self, anchor):
        props = self.tools.get("wechat.chat.get_message_around", {}).get("inputSchema", {}).get("properties", {})
        arguments = {}
        if "sent_at_unix_ms" in props:
            arguments["sent_at_unix_ms"] = anchor["timestamp_ms"]
        server = anchor["external_id"]
        if "server_id" in props and server.isascii() and server.isdecimal() and 0 < int(server) <= 2**64 - 1:
            arguments["server_id"] = server
        self.diagnostics["around_seek_hint"] = "server_id" if "server_id" in arguments else "timestamp" if arguments else "legacy_id"
        return self.call("wechat.chat.get_message_around", username=anchor["conversation_id"],
                         anchor_id=anchor["api_id"], before=20, after=20, **arguments)

    @staticmethod
    def is_anchor(row, anchor):
        if row["id"] != anchor["api_id"]:
            return False
        if row.get("timestamp_ms") is not None:
            return row["timestamp_ms"] == anchor["timestamp_ms"]
        return row["sent_epoch"] is not None and int(row["sent_epoch"]) == int(anchor["timestamp_ms"]) // 1000

    def mcp_rows(self, value, limit):
        if isinstance(value.get("before"), list) and isinstance(value.get("after"), list) and isinstance(value.get("anchor"), dict):
            rows = value["before"] + [value["anchor"]] + value["after"]
        else:
            matches = [value[k] for k in ("messages", "items") if k in value]
            rows = matches[0] if len(matches) == 1 else None
        if not isinstance(rows, list) or len(rows) > limit or self.read_count + len(rows) > 200:
            raise invalid("聊天窗口结构或条数超限")
        self.read_count += len(rows)
        self.diagnostics["mcp_rows_read"] = self.read_count
        return rows

    @staticmethod
    def normalize(row, conversation, account):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not 0 < len(row["id"]) <= 256:
            raise invalid("消息缺少稳定身份", "context_identity")
        if row.get("conversation_id", conversation) != conversation or row.get("account_id", account) != account:
            raise invalid("消息来源不匹配", "context_identity")
        body, sender = row.get("text", ""), row.get("sender", row.get("sender_name", ""))
        if not isinstance(sender, str) or len(sender) > 256:
            raise invalid()
        unsupported = False
        try:
            validate_body(body)
        except ValueError:
            body, unsupported = "[消息正文不支持或超过读取上限]", True
        raw_time = row.get("time")
        sent_at = normalize_time(raw_time) if raw_time else None
        timestamp_ms = None
        moment = epoch(sent_at)
        if "sent_at_unix_ms" in row:
            millis = integer(row["sent_at_unix_ms"], "消息毫秒时间", minimum=1, maximum=9999999999999)
            if moment is not None and int(moment) != millis // 1000:
                raise invalid("消息时间字段互相矛盾", "context_identity")
            timestamp_ms, moment = str(millis), millis / 1000
            sent_at = normalize_time(moment)
        quote = row.get("quote") or {}
        quoted = quote.get("text", quote.get("content", "")) if isinstance(quote, dict) else ""
        try:
            validate_body(quoted)
        except ValueError:
            quoted, unsupported = "[引用正文暂不可读取]", True
        sender_id = row.get("sender_id", "")
        if not isinstance(sender_id, str) or len(sender_id) > 1024:
            raise invalid()
        return {"id": row["id"], "external_id": row["id"], "sender": sender, "sender_id": sender_id, "sent_at": sent_at,
                "sent_epoch": moment, "timestamp_ms": timestamp_ms, "body": body, "display_body": None,
                "from_me": row.get("from_me") if type(row.get("from_me")) is bool else None,
                "quote_body": quoted[:240], "quote_sender": str(quote.get("sender", ""))[:200] if isinstance(quote, dict) else "",
                "forwarded": False, "attachments": [], "forward_items": (), "gap_before": False,
                "hit": False, "source_anchor": False, "day_heading": "", "remote": True,
                "unsupported": unsupported}

    def fetch(self, anchor, *, kind="around", start=None, end=None, force=False):
        self.check()
        self.read_count = 0
        self.tools = {}
        self.catalog_names = set()
        self.diagnostics = {}
        if kind not in {"around", "forward"}:
            raise invalid("未知聊天读取操作")
        if self.client.base_url != anchor["base_url"] or self.client.account_id not in (None, anchor["account_id"]):
            raise invalid("请连接建立此索引的 Alpha 地址和账号", "context_identity")
        self.client.account_id = anchor["account_id"]
        self.client.accounts()
        if self.client.protocol != "alpha" or not self.client._token:
            raise invalid("请先连接 Alpha 本地接口", "context_unavailable")
        key = (anchor["base_url"], anchor["account_id"], anchor["conversation_id"], anchor["api_id"],
               anchor["revision"], kind, start, end, hashlib.sha256(self.client._token.encode()).hexdigest())
        cached = None if force else self.cache.get(key)
        if cached:
            self.diagnostics["cache_hit"] = True
            return cached
        available = True
        try:
            self.discover()
        except WeFlowError as exc:
            if exc.status != 404 and exc.code != "context_unavailable":
                raise
            available = False
            self.diagnostics["mcp_fallback_reason"] = exc.code or "http_404"
        self.diagnostics["forward_tool_advertised"] = "wechat.chat.resolve_chat_history" in self.tools
        self.diagnostics["raw_tool_advertised"] = "wechat.chat.get_message_raw" in self.tools
        stamp = int(anchor["timestamp_ms"]) / 1000
        lo, hi = max(stamp - 900, start or 0), min(stamp + 900, end if end is not None else stamp + 900)
        if not lo <= stamp <= hi or end is not None and stamp >= end:
            raise invalid("锚点不在当前日期范围内", "context_identity")
        value = None
        if kind == "forward" and available:
            try:
                if not anchor["external_id"].isdecimal():
                    raise invalid("缺少可核对的转发消息 server_id", "context_unavailable")
                payload = self.call("wechat.chat.resolve_chat_history", username=anchor["conversation_id"], server_id=anchor["external_id"])
                if "server_id" in payload and str(payload["server_id"]) != anchor["external_id"]:
                    raise invalid("转发外层消息身份不匹配", "context_identity")
                raw = self.mcp_rows(payload, 200)
                if not raw:
                    raise invalid("转发工具未返回子记录，尝试有界 HTTP 读取", "context_unavailable")
                rows = []
                for i, row in enumerate(raw):
                    if not isinstance(row, dict):
                        raise invalid()
                    if any(row.get(key) for key in ("children", "messages", "items", "record_items")):
                        raise invalid("嵌套转发结构需使用原始记录读取", "context_unavailable")
                    # Inner identities are scoped to the outer archive, never to the outer conversation.
                    inner = {key: value for key, value in row.items() if key != "conversation_id"}
                    item = self.normalize({**inner, "id": "forward:" + str(i)}, anchor["conversation_id"], anchor["account_id"])
                    item["from_me"] = None
                    item["forward_path"] = (i + 1,)
                    item["archive_child"] = True
                    rows.append(item)
                value = ContextBatch(tuple(rows), "Alpha 转发记录 · 原发送者独立展示；未写入索引，未核验子项附件路径", "mcp_forward", None, None)
            except WeFlowError as exc:
                if exc.code != "context_unavailable":
                    raise
        elif available:
            try:
                # Confirm that the MCP message namespace actually shares the saved HTTP ID.
                bridge = self.mcp_rows(self.anchor_query(anchor), 20)
                matches = [r for r in bridge if isinstance(r, dict) and r.get("id") == anchor["api_id"]]
                if len(matches) == 1:
                    if not self.is_anchor(self.normalize(matches[0], anchor["conversation_id"], anchor["account_id"]), anchor):
                        raise invalid("MCP 锚点时间与本地保存身份不一致", "context_identity")
                    payload = self.around_query(anchor)
                    raw = self.mcp_rows(payload, 41)
                    rows = [self.normalize(r, anchor["conversation_id"], anchor["account_id"]) for r in raw]
                    selected = [r for r in rows if self.is_anchor(r, anchor)]
                    if len(selected) != 1:
                        raise invalid("附近消息未返回同一精确锚点", "context_identity")
                    rows = [r for r in rows if r["sent_epoch"] is not None and lo <= r["sent_epoch"] <= hi
                            and (end is None or r["sent_epoch"] < end)]
                    value = ContextBatch(tuple(rows), "Alpha 锚点窗口 · 前后各最多 20 条候选，已按时间范围筛选展示；未返回历史未知，未写入索引", "mcp_around", lo, hi)
                elif not matches:
                    self.diagnostics["mcp_fallback_reason"] = "anchor_namespace_unconfirmed"
                else:
                    raise invalid("MCP 锚点身份重复", "context_identity")
            except WeFlowError as exc:
                if exc.code != "context_unavailable":
                    raise
                self.diagnostics["mcp_fallback_reason"] = exc.code
        if value is None:
            limit = 200 - self.read_count
            if limit <= 0:
                raise invalid("本轮已达到消息读取上限，未继续读取", "context_budget")
            payload = self.client.get("/api/v1/messages", talker=anchor["conversation_id"],
                                      start=int(stamp) if kind == "forward" else int(lo),
                                      end=int(stamp) if kind == "forward" else int(hi), limit=limit, media="0")
            self.client._check_response_account(payload)
            raw = payload.get("messages")
            if payload.get("talker") != anchor["conversation_id"] or not isinstance(raw, list) or len(raw) > limit:
                raise invalid("HTTP 聊天窗口来源或条数不匹配", "context_identity")
            rows, issues, forwards, seen = [], [], None, set()
            for row in raw:
                if (not isinstance(row, dict) or row.get("conversation_id") != anchor["conversation_id"]
                        or row.get("account_id", anchor["account_id"]) != anchor["account_id"]
                        or not isinstance(row.get("id"), str) or row["id"] in seen):
                    raise invalid("HTTP 消息身份缺失或重复", "context_identity")
                seen.add(row["id"])
                messages = decode_messages([row], issues=issues)
                if not messages:
                    continue
                msg = messages[0]
                item = self.normalize({"id": row["id"], "text": msg.display_body if msg.display_body is not None else msg.body,
                    "sender": msg.sender, "sender_id": msg.sender_id, "time": msg.sent_at, "from_me": msg.from_me,
                    "sent_at_unix_ms": row["sent_at_unix_ms"],
                    "quote": {"text": msg.quote_body, "sender": msg.quote_sender}}, anchor["conversation_id"], anchor["account_id"])
                item["external_id"] = msg.external_id
                rows.append(item)
                if row["id"] == anchor["api_id"]:
                    if str(row.get("sent_at_unix_ms")) != anchor["timestamp_ms"] or msg.external_id != anchor["external_id"]:
                        raise invalid("HTTP 精确消息身份不匹配", "context_identity")
                    forwards = msg.forward_items
                    self.diagnostics["http_anchor_forwarded"] = bool(msg.forwarded)
                    self.diagnostics["http_forward_items"] = len(forwards)
                    self.diagnostics["http_anchor_structure"] = {
                        "known_fields": {key: type(row[key]).__name__ for key in
                            ("text", "content", "rawContent", "raw_content", "refs", "messages", "items", "children", "recorditem", "xml") if key in row},
                        "reference_count": len(row["refs"]) if isinstance(row.get("refs"), list) else None,
                        "raw_xml_present": any(isinstance(row.get(key), str) and row[key].lstrip().startswith(("<", "&lt;"))
                                               for key in ("text", "content", "rawContent", "raw_content", "xml")),
                    }
            selected = [r for r in rows if r["id"] == anchor["api_id"]]
            if len(selected) != 1:
                raise invalid("有界窗口中未找到所选消息，请更新会话索引后重试", "context_anchor_missing")
            if kind == "forward":
                if not forwards:
                    reason = "当前 Alpha 未提供转发展开工具，且 HTTP 响应没有可解析的转发子项" if not self.diagnostics["forward_tool_advertised"] else "所选消息未返回可解析的转发子项"
                    raise invalid(reason + "；无法还原内部聊天，可继续查看外层消息和本地归档", "context_forward_unavailable")
                rows = [self.normalize({"id": "forward:" + ".".join(map(str, f.path)), "text": f.body, "sender": f.sender},
                                      anchor["conversation_id"], anchor["account_id"]) | {"forward_path": f.path, "time_text": f.time_text,
                                      "archive_child": True, "forwarded": f.kind == "nested"} for f in forwards]
                value = ContextBatch(tuple(rows), "Alpha 转发子项 · 原时间按原始文字展示，未写入索引", "http_forward", None, None, skipped=len(issues))
            else:
                rows = sorted((r for r in rows if r["sent_epoch"] is not None and lo <= r["sent_epoch"] <= hi
                               and (end is None or r["sent_epoch"] < end)), key=lambda r: r["sent_epoch"])
                center = next(i for i, r in enumerate(rows) if r["id"] == anchor["api_id"])
                complete = payload.get("hasMore") is False and payload.get("scanComplete") is True and not issues
                rows = rows[max(0, center - 20):center + 21]
                status = "请求时段已读完" if complete else "请求时段可能未读完"
                value = ContextBatch(tuple(rows), f"Alpha 有界时间窗口 · {status}；仅展示锚点前后各 20 条，未写入索引", "http_window", lo, hi, complete, len(issues))
        self.check()
        unsupported = sum(bool(r.get("unsupported")) for r in value.rows)
        if unsupported:
            value = replace(value, skipped=value.skipped + unsupported, complete_range=False,
                            notice=value.notice + f"；{unsupported} 条正文或引用未完整展示")
        if len({r["id"] for r in value.rows}) != len(value.rows) or sum(len(r["body"]) for r in value.rows) > 500_000:
            raise invalid("聊天窗口重复或内容超限")
        self.cache.put(key, value)
        return value
