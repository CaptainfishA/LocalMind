"""Pure response normalization shared by legacy WeFlow and Alpha HTTP clients."""
from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .chats import Attachment, ChatMessage, ForwardItem, digest, mentioned_files, normalize_time
from .chat_limits import MAX_FORWARD_DEPTH, MAX_FORWARD_ITEMS, MAX_MESSAGE_CHARS, MAX_RAW_MESSAGE_CHARS, MAX_XML_DEPTH, MAX_XML_NODES, validate_body
from .weflow_scope import CATEGORIES

ALPHA_SESSION_PAGE_SIZE = 200
MESSAGE_PAGE_SIZE = 500
MAX_SESSION_PAGES = 100


class WeFlowError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code


class MessageContentError(WeFlowError):
    """Recoverable payload failure; identity, account and time errors stay fatal."""


ISSUE_LABELS = {
    "body_type": "正文不是文本",
    "body_limit": "规范化正文超出安全上限",
    "invalid_text": "正文包含无效字符",
    "raw_type": "原始消息格式不支持",
    "raw_limit": "原始消息超出安全上限",
    "xml_unsafe": "XML 包含不允许的实体声明",
    "xml_invalid": "XML 结构损坏",
    "xml_complexity": "XML 层级或节点过多",
    "structured_unsupported": "结构化消息尚不支持",
    "forward_unsupported": "合并转发待解析",
    "forward_partial": "合并转发部分子项待解析，已保留可读取内容",
    "forward_limit": "合并转发层级或子项数超过安全上限",
    "attachment_invalid": "附件名称或结构不完整",
}


def content_error(code: str) -> MessageContentError:
    return MessageContentError(ISSUE_LABELS[code], code=code)


def _api_error(status: int, payload: object = None) -> WeFlowError:
    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    if not isinstance(error, dict) and isinstance(payload, dict):
        error = payload
    code = error.get("code", "") if isinstance(error, dict) else ""
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", code):
        code = ""
    messages = {
        200: "API 返回失败状态，请检查账号连接和接口版本",
        400: "API 参数不兼容，请检查页大小、时间范围及 Alpha / WeFlow 版本",
        401: "API 鉴权失败，请检查 Alpha / WeFlow 的访问 Token",
        403: "API 拒绝访问，请检查 Token、接口权限和本机监听地址",
        404: "当前服务不支持此接口；请核对 Alpha / WeFlow 版本",
        409: "API 当前状态不允许读取，请检查账号连接状态",
        413: "API 请求超出读取上限，请缩小时间范围或每页条数",
        423: "Alpha 暂停了接口访问；请在 Alpha 中检查应用锁、账号连接或更新要求",
        429: "API 请求过于频繁，请稍后重试",
    }
    message = messages.get(status, f"API HTTP {status}，请检查账号选择和服务状态")
    if status in {502, 503, 504}:
        message = "Alpha 上游数据服务未就绪或查询超时，请确认微信数据库已连接后重试"
    if code:
        message += f"（{code}）"
    return WeFlowError(message, status=status, code=code)


def integer(value: object, label: str, *, minimum=0, maximum=2**63 - 1) -> int:
    # Decimal strings preserve wire-level int64 precision. Floats/bools do not.
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 20:
        value = int(value)
    if type(value) is not int or not minimum <= value <= maximum:
        raise WeFlowError(f"{label}格式不兼容")
    return value


@dataclass(frozen=True)
class SessionPage:
    sessions: list[dict]
    next_offset: int | None


def session_page(payload: dict, offset: int, *, alpha: bool) -> SessionPage:
    rows = payload.get("items") if alpha and "items" in payload else payload.get("sessions", payload.get("items"))
    if not isinstance(rows, list):
        raise WeFlowError("会话列表接口不兼容")
    mirror = payload.get("sessions", []) if alpha else []
    by_id = {r.get("username"): r for r in mirror if isinstance(r, dict) and isinstance(r.get("username"), str)} if isinstance(mirror, list) else {}
    sessions = []
    for row in rows:
        if not isinstance(row, dict):
            raise WeFlowError("会话记录结构不兼容")
        identity = row.get("username") or row.get("id")
        if not isinstance(identity, str) or not identity or len(identity) > 1024:
            raise WeFlowError("会话缺少稳定 ID")
        name = row.get("displayName") or row.get("display_name")
        session = {"id": identity, "name": name[:512] if isinstance(name, str) and name else identity}
        metadata = by_id.get(identity, {})
        known_types = {value for value in (row.get("sessionType"), metadata.get("sessionType")) if isinstance(value, str) and value in CATEGORIES}
        category = next(iter(known_types)) if len(known_types) == 1 else "unknown" if known_types else None
        # Do not guess private/official status from names or numeric enums.
        if not isinstance(category, str) or category not in CATEGORIES:
            category = "group" if identity.endswith("@chatroom") else "unknown"
        session["category"] = category
        session["category_source"] = "conflict" if len(known_types) > 1 else "api" if known_types else "id" if category == "group" else "unknown"
        aliases = [row.get(key) or metadata.get(key) for key in ("remark", "nickName", "nickname", "alias")]
        session["aliases"] = list(dict.fromkeys(v[:256] for v in aliases if isinstance(v, str) and v))
        value = row.get("last_activity_unix_ms") if "last_activity_unix_ms" in row else row.get("lastTimestamp")
        if value is not None:
            try:
                moment = integer(value, "会话时间", minimum=1)
                session["last_activity"] = moment // 1000 if "last_activity_unix_ms" in row else moment
            except WeFlowError:
                pass  # Unknown activity must never exclude a conversation.
        sessions.append(session)
    more = payload.get("hasMore", False if not alpha else None)
    if type(more) is not bool:
        raise WeFlowError("会话分页标记格式不兼容")
    if not more:
        if payload.get("scanComplete") is False:
            raise WeFlowError("Alpha 会话扫描尚未完成，请稍后重新读取")
        return SessionPage(sessions, None)
    cursor = payload.get("nextOffset") if alpha else payload.get("nextOffset", offset + len(rows))
    next_offset = integer(cursor, "会话分页游标")
    if next_offset <= offset:
        raise WeFlowError("会话分页未推进，已停止读取；请缩小会话名称范围")
    return SessionPage(sessions, next_offset)


def _xml_root(raw: str):
    if len(raw) > MAX_RAW_MESSAGE_CHARS:
        raise content_error("raw_limit")
    text = raw if "<" in raw else html.unescape(raw)
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise content_error("xml_unsafe")
    start = text.find("<")
    if start < 0:
        return None
    try:
        parser = ET.XMLPullParser(events=("start", "end"))
        depth = nodes = 0
        root = None
        for offset in range(start, len(text), 4096):
            parser.feed(text[offset:offset + 4096])
            for event, element in parser.read_events():
                if event == "start":
                    depth += 1
                    nodes += 1
                    if root is None:
                        root = element
                    if depth > MAX_XML_DEPTH or nodes > MAX_XML_NODES:
                        raise content_error("xml_complexity")
                else:
                    depth -= 1
        parser.close()
    except ET.ParseError:
        raise content_error("xml_invalid") from None
    return root


def _app_attachment(root) -> tuple[list[Attachment], str | None]:
    if root is None:
        return [], None
    app = root if root.tag == "appmsg" else root.find("appmsg")
    if app is None:
        return [], None
    reply = app.findtext("refermsg/svrid") or None
    if app.findtext("type") != "6":
        return [], reply
    name = (app.findtext("title") or "").strip()
    if not name or len(name) > 240:
        return [], reply
    md5 = (app.findtext("appattach/filemd5") or app.findtext("md5") or "").lower()
    if not re.fullmatch(r"[a-f0-9]{32}", md5):
        md5 = ""
    size_text = app.findtext("appattach/totallen") or ""
    size = int(size_text) if size_text.isascii() and size_text.isdigit() and len(size_text) <= 18 else None
    return [Attachment(name=name, size=size, md5=md5)], reply


def _xml_attachment(raw: str) -> tuple[list[Attachment], str | None]:
    # Compatibility helper for callers probing an optional raw field.
    try:
        return _app_attachment(_xml_root(raw))
    except MessageContentError:
        return [], None


def _forward_content(root) -> tuple[str, list[Attachment], bool, tuple[ForwardItem, ...]] | None:
    app = root if root is not None and root.tag == "appmsg" else root.find("appmsg") if root is not None else None
    if root is not None and root.tag == "recordinfo":
        container = root
    elif app is not None and (app.findtext("type") == "19" or app.find("recorditem") is not None):
        container = app.find("recorditem")
    else:
        return None
    if container is None:
        raise content_error("forward_unsupported")
    budget = {"items": 0, "nodes": 0, "chars": 0}
    partial = False
    passages: list[str] = []
    attachments: list[Attachment] = []
    items: list[ForwardItem] = []

    def children(value):
        # Only descend through record containers, never into a quoted/link card.
        for _ in range(4):
            if value.tag == "datalist":
                rows = list(value)
                if any(row.tag != "dataitem" for row in rows):
                    raise content_error("forward_unsupported")
                count = value.get("count")
                if count is not None and (not count.isascii() or not count.isdecimal() or len(count) > 6 or int(count) != len(rows)):
                    raise content_error("forward_unsupported")
                return rows
            nested = list(value)
            if nested:
                candidates = [child for child in nested if child.tag in {"recordinfo", "recordxml", "datalist"}]
                if len(candidates) != 1:
                    raise content_error("forward_unsupported")
                value = candidates[0]
            else:
                text = (value.text or "").strip()
                if not text.startswith(("<", "&lt;")):
                    raise content_error("forward_unsupported")
                budget["chars"] += len(text)
                if budget["chars"] > MAX_RAW_MESSAGE_CHARS * MAX_FORWARD_DEPTH:
                    raise content_error("forward_limit")
                value = _xml_root(text)
                if value is None or value.tag not in {"recordinfo", "recordxml", "recorditem", "datalist"}:
                    raise content_error("forward_unsupported")
                budget["nodes"] += sum(1 for _node in value.iter())
                if budget["nodes"] > MAX_XML_NODES:
                    raise content_error("xml_complexity")
        raise content_error("forward_unsupported")

    def field(item, key, limit=MAX_MESSAGE_CHARS):
        value = item.find(key)
        if value is None:
            return ""
        if len(value):
            raise content_error("forward_unsupported")
        text = (value.text or "").strip()
        if len(text) > limit:
            raise content_error("body_limit")
        try:
            validate_body(text)
        except ValueError:
            raise content_error("invalid_text") from None
        return text

    def walk(value, prefix=()):
        nonlocal partial
        if len(prefix) >= MAX_FORWARD_DEPTH:
            raise content_error("forward_limit")
        rows = children(value)
        budget["items"] += len(rows)
        if budget["items"] > MAX_FORWARD_ITEMS:
            raise content_error("forward_limit")
        entries = []
        for index, item in enumerate(rows):
            path = prefix + (index + 1,)
            checkpoint = (len(items), len(attachments), len(passages))
            position = ".".join(map(str, path))
            kind, body, att = "unknown", "", None
            try:
                datatype = item.get("datatype") or field(item, "datatype", 8)
                sender, moment = field(item, "sourcename", 200), field(item, "sourcetime", 128)
                origin = f"合并转发内第 {position} 项\n原发送者: {sender or '未知'}\n原时间（原始文本）: {moment or '未知'}"
                title = field(item, "datatitle", 240)
                if datatype == "17":
                    nested = item.find("recordxml")
                    if nested is None:
                        raise content_error("forward_unsupported")
                    passages.append(origin + "\n[嵌套转发]")
                    items.append(ForwardItem(path, sender, moment, "nested", "[嵌套转发]"))
                    walk(nested, path)
                    entries.append(("nested", "", None))
                    continue
                if datatype == "1":
                    kind = "text"
                    body = field(item, "datadesc") or field(item, "content")
                elif datatype == "8":
                    if not title:
                        raise content_error("attachment_invalid")
                    kind = "file"
                    size_text = field(item, "datasize", 18)
                    size = int(size_text) if size_text and size_text.isascii() and size_text.isdecimal() else None
                    if size_text and size is None:
                        raise content_error("attachment_invalid")
                    md5 = field(item, "fullmd5", 64) or field(item, "md5", 64) or field(item, "datamd5", 64)
                    md5 = md5.lower() if re.fullmatch(r"[A-Fa-f0-9]{32}", md5) else ""
                    body = "[文件] " + title
                    # Source-device paths/CDN URLs/keys do not identify local files.
                    att = Attachment(title, size=size, md5=md5, context=body, origin=origin)
                    attachments.append(att)
                else:
                    partial = True
                    body = "[未解析转发项]\n" + "\n".join(filter(None, (title, field(item, "datadesc"))))
                formatted = origin + "\n" + body
                items.append(ForwardItem(path, sender, moment, kind, body,
                                         len(attachments) - 1 if att is not None else None))
                passages.append(formatted)
                entries.append((kind, formatted, att))
            except MessageContentError as exc:
                if exc.code in {"forward_limit", "raw_limit", "xml_complexity"}:
                    raise
                partial = True
                del items[checkpoint[0]:]
                del attachments[checkpoint[1]:]
                del passages[checkpoint[2]:]
                items.append(ForwardItem(path, kind="unknown", body="[子项待解析]"))
                passages.append(f"合并转发内第 {position} 项\n[子项待解析]")
                entries.append(("unknown", "", None))
        # Assign only nearby text with a unique nearest file in this container.
        # A tie or another card/file in between is not evidence of association.
        contexts: dict[int, list[str]] = {}
        for index, (kind, formatted, _) in enumerate(entries):
            if kind != "text" or not formatted:
                continue
            candidates = []
            for direction in (-1, 1):
                for distance in (1, 2):
                    other = index + direction * distance
                    if not 0 <= other < len(entries):
                        break
                    candidate_kind, _, candidate = entries[other]
                    if candidate_kind == "file":
                        candidates.append((distance, other))
                        break
                    if candidate_kind != "text":
                        break
            if candidates:
                nearest = min(distance for distance, _ in candidates)
                matches = [other for distance, other in candidates if distance == nearest]
                if len(matches) == 1:
                    contexts.setdefault(matches[0], []).append(formatted)
        for index, values in contexts.items():
            entries[index][2].context += "\n相邻转发消息（非附件正文）:\n" + "\n".join(values)

    walk(container)
    if not passages:
        raise content_error("forward_unsupported")
    body = "[合并转发]\n" + "\n\n".join(passages)
    if len(body) + sum(len(a.context) + len(a.origin) for a in attachments) > MAX_MESSAGE_CHARS:
        raise content_error("body_limit")
    return body, attachments, partial, tuple(items)


def _structured_text(root, attachments: list[Attachment]) -> str:
    if attachments:
        return "[文件] " + attachments[0].name
    if root is not None:
        app = root if root.tag == "appmsg" else root.find("appmsg")
        if app is not None:
            if app.findtext("type") == "6":
                raise content_error("attachment_invalid")
            parts = [app.findtext(name) or "" for name in ("title", "des")]
            if any(parts):
                return "\n".join(part for part in parts if part)
    raise content_error("structured_unsupported")


def _message_id(value: object) -> str:
    if value is None:
        return ""
    if type(value) is int:
        return str(value)
    if isinstance(value, str) and len(value) <= 256 and "\x00" not in value:
        return value
    raise WeFlowError("消息 ID 格式不兼容，拒绝可能丢失精度的值")


def decode_messages(rows: list[dict], names: dict[str, str] | None = None, *,
                    issues: list[dict] | None = None) -> list[ChatMessage]:
    names = names or {}
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise WeFlowError("WeFlow 消息结构不兼容")
        alpha_id = None
        if "sent_at_unix_ms" in row:
            millis = integer(row["sent_at_unix_ms"], "Alpha 消息毫秒时间戳", minimum=1, maximum=9999999999999)
            alpha_id = _message_id(row.get("id"))
            row = {**row, "createTime": millis / 1000, "content": row.get("text", ""),
                   "senderUsername": row.get("sender_id"), "localId": row.get("local_id"),
                   "serverId": row.get("platform_message_id") or row.get("server_id"),
                   "localType": row.get("source_type")}
        timestamp = row.get("createTime")
        if type(timestamp) not in {int, float} or not 0 < timestamp < 10000000000:
            raise WeFlowError("WeFlow 消息缺少有效的秒级时间戳")
        sender_id = str(row.get("senderUsername") or "")
        sender = str(row.get("sender_name") or names.get(sender_id, sender_id or ("我" if row.get("isSend") == 1 else "未知发送者")))[:200]
        content = row.get("content")
        if content is None or content == "":
            content = row.get("parsedContent")
        content = "" if content is None else content
        raw = row.get("rawContent")
        if raw is None or raw == "":
            raw = content if isinstance(content, str) and content.lstrip().startswith(("<", "&lt;")) else ""
        server_id = _message_id(row.get("serverId"))
        if server_id and server_id != "0":
            identity = server_id
        elif alpha_id:
            identity = "alpha:" + alpha_id
        else:
            identity = "local:" + digest(json.dumps([row.get("localId"), timestamp, sender_id, row.get("localType"), raw or content], ensure_ascii=False))
        sent_at = normalize_time(timestamp)
        try:
            if not isinstance(content, str):
                raise content_error("body_type")
            if not isinstance(raw, str):
                raise content_error("raw_type")
            if max(len(raw), len(content)) > MAX_RAW_MESSAGE_CHARS:
                raise content_error("raw_limit")
            root = _xml_root(raw) if raw.lstrip().startswith(("<", "&lt;")) else None
            forward = _forward_content(root)
            attachments, xml_reply = _app_attachment(root) if forward is None else (forward[1], None)
            if forward is not None:
                content = forward[0]
            elif content.lstrip().startswith(("<", "&lt;")) or (not content and root is not None):
                content = _structured_text(root, attachments)
            if not attachments and content.startswith("[文件]"):
                name = content.removeprefix("[文件]").strip()
                if name and len(name) <= 240:
                    attachments = [Attachment(name)]
            display = row.get("display")
            if forward is None and isinstance(display, dict):
                filename = display.get("fileName")
                if not attachments and display.get("kind") == "file" and isinstance(filename, str) and 0 < len(filename) <= 240:
                    attachments = [Attachment(filename)]
                if len(attachments) == 1 and attachments[0].name == filename:
                    size = str(display.get("fileSize", ""))
                    if size.isascii() and size.isdigit() and len(size) < 20:
                        attachments[0].size = int(size)
            if len(content) > MAX_MESSAGE_CHARS:
                raise content_error("body_limit")
            try:
                validate_body(content)
            except ValueError:
                raise content_error("invalid_text") from None
            if not attachments and forward is None:
                attachments = mentioned_files(content)
            display_body = content
            quote_body, quote_sender = "", ""
            quote = row.get("quote")
            if isinstance(quote, dict) and isinstance(quote.get("content"), str):
                quote_sender = str(quote.get("accountName") or quote.get("sender") or "未知发送者")[:100]
                quote_body = quote["content"][:240]
                snapshot = f"\n引用快照（不是附件正文） {quote_sender}: {quote['content'][:240]}"
                # Never truncate the main body to make room for a quoted preview.
                if len(content) + len(snapshot) <= MAX_MESSAGE_CHARS:
                    content += snapshot
            try:
                validate_body(content)
                validate_body(quote_body)
                if "\x00" in sender_id or "\x00" in quote_sender:
                    raise ValueError("Invalid display metadata")
            except ValueError:
                raise content_error("invalid_text") from None
            if forward is not None and len(content) + sum(len(a.context) + len(a.origin) for a in attachments) > MAX_MESSAGE_CHARS:
                raise content_error("body_limit")
            if forward is not None and forward[2]:
                if issues is None:
                    raise content_error("forward_partial")
                issues.append({"message_key": digest(identity), "sent_at": sent_at, "code": "forward_partial",
                               "chars": len(content), "partial": True})
        except MessageContentError as exc:
            if issues is None:
                raise
            issues.append({"message_key": digest(identity), "sent_at": sent_at, "code": exc.code,
                           "chars": len(content) if isinstance(content, str) else 0})
            continue
        direction = row.get("from_me")
        if type(direction) is not bool:
            direction = bool(row["isSend"]) if type(row.get("isSend")) is int and row["isSend"] in (0, 1) else None
        result.append(ChatMessage(identity, content, sender, normalize_time(timestamp), attachments,
                                  str(row.get("replyToMessageId") or xml_reply or "") or None, forward is not None,
                                  sender_id[:1024], direction, str(row.get("localType") or "")[:128],
                                  display_body if display_body != content else None, quote_body, quote_sender,
                                  forward[3] if forward is not None else ()))
    return result
