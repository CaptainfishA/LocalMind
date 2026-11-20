from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

import numpy as np

from .models import utc_now_iso
from .chunker import split_text
from .chat_limits import MAX_FORWARD_DEPTH, MAX_FORWARD_ITEMS, MAX_MESSAGE_CHARS, validate_body
from .search_filters import chat_predicates, epoch, file_predicates, interval_predicates, where

if TYPE_CHECKING:
    from .db import LocalMindDB
    from .embeddings import EmbeddingService

CHAT_PREFIX = "localmind-chat://"
MAX_MESSAGES = 20000
MAX_IMPORT_BYTES = 20 * 1024 * 1024
LOCAL_TZ = timezone(timedelta(hours=8))


def init_chat_schema(conn: sqlite3.Connection) -> None:
    conn.create_function("localmind_epoch", 1, epoch)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_sources (
            id INTEGER PRIMARY KEY, source_key TEXT NOT NULL UNIQUE,
            platform TEXT NOT NULL, account TEXT NOT NULL, conversation TEXT NOT NULL,
            kind TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES chat_sources(id) ON DELETE CASCADE,
            external_id TEXT NOT NULL, sender TEXT NOT NULL, sent_at TEXT,
            body TEXT NOT NULL, observed_at TEXT NOT NULL, reply_to TEXT,
            UNIQUE(source_id, external_id)
        );
        CREATE TABLE IF NOT EXISTS chat_attachments (
            id INTEGER PRIMARY KEY, message_id INTEGER NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, name TEXT NOT NULL, path_hint TEXT NOT NULL,
            sha256 TEXT NOT NULL, size INTEGER, kind TEXT NOT NULL,
            file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
            link_state TEXT NOT NULL DEFAULT 'missing', linked_hash TEXT,
            UNIQUE(message_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS chat_units (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            owner_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            message_id INTEGER NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            attachment_id INTEGER REFERENCES chat_attachments(id) ON DELETE CASCADE,
            unit_key TEXT NOT NULL UNIQUE, fingerprint TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_message_source ON chat_messages(source_id);
        CREATE INDEX IF NOT EXISTS idx_chat_attachment_message ON chat_attachments(message_id);
        CREATE INDEX IF NOT EXISTS idx_chat_attachment_file ON chat_attachments(file_id);
        CREATE INDEX IF NOT EXISTS idx_chat_unit_message ON chat_units(message_id);
        CREATE INDEX IF NOT EXISTS idx_files_name_chat ON files(name COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_files_hash_chat ON files(file_hash);
        CREATE TABLE IF NOT EXISTS chat_file_digests (
            file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
            sha256 TEXT NOT NULL, md5 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_sync_issues (
            source_key TEXT NOT NULL, message_key TEXT NOT NULL,
            sent_at TEXT NOT NULL, code TEXT NOT NULL, chars INTEGER NOT NULL,
            first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(source_key, message_key)
        );
        CREATE TABLE IF NOT EXISTS chat_attachment_locations (
            attachment_id INTEGER PRIMARY KEY REFERENCES chat_attachments(id) ON DELETE CASCADE,
            signature TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, origin TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_api_messages (
            message_id INTEGER PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
            base_url TEXT NOT NULL, account_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
            api_id TEXT NOT NULL, timestamp_ms TEXT NOT NULL, source_type TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_coverage (
            id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES chat_sources(id) ON DELETE CASCADE,
            start_epoch INTEGER NOT NULL, end_epoch INTEGER NOT NULL, state TEXT NOT NULL,
            issue_count INTEGER NOT NULL, checked_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_coverage_source ON chat_coverage(source_id,id DESC);
    """)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(chat_attachments)")}
    if "md5" not in columns:
        conn.execute("ALTER TABLE chat_attachments ADD COLUMN md5 TEXT NOT NULL DEFAULT ''")
    for column in ("context", "origin"):
        if column not in columns:
            conn.execute(f"ALTER TABLE chat_attachments ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
    if "forwarded" not in {r[1] for r in conn.execute("PRAGMA table_info(chat_messages)")}:
        conn.execute("ALTER TABLE chat_messages ADD COLUMN forwarded INTEGER NOT NULL DEFAULT 0")
    columns = {r[1] for r in conn.execute("PRAGMA table_info(chat_messages)")}
    for name, declaration in (("sender_id", "TEXT NOT NULL DEFAULT ''"), ("from_me", "INTEGER"),
                              ("message_type", "TEXT NOT NULL DEFAULT ''"), ("display_body", "TEXT"),
                              ("quote_body", "TEXT NOT NULL DEFAULT ''"), ("quote_sender", "TEXT NOT NULL DEFAULT ''"),
                              ("forward_json", "TEXT NOT NULL DEFAULT '[]'"),
                              ("sent_epoch", "REAL")):
        if name not in columns:
            conn.execute(f"ALTER TABLE chat_messages ADD COLUMN {name} {declaration}")
    if "sent_epoch" not in columns:
        conn.execute("UPDATE chat_messages SET sent_epoch=localmind_epoch(sent_at)")
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_chat_timeline ON chat_messages(source_id,sent_epoch,id);
        CREATE INDEX IF NOT EXISTS idx_chat_reply ON chat_messages(source_id,reply_to);
        CREATE TRIGGER IF NOT EXISTS chat_time_update AFTER UPDATE OF sent_at ON chat_messages
        BEGIN UPDATE chat_messages SET sent_epoch=localmind_epoch(NEW.sent_at) WHERE id=NEW.id; END;
        CREATE TRIGGER IF NOT EXISTS chat_time_insert AFTER INSERT ON chat_messages
        WHEN NEW.sent_epoch IS NULL AND NEW.sent_at IS NOT NULL
        BEGIN UPDATE chat_messages SET sent_epoch=localmind_epoch(NEW.sent_at) WHERE id=NEW.id; END;
    """)


@dataclass
class Attachment:
    name: str
    path: str = ""
    sha256: str = ""
    size: int | None = None
    kind: str = "explicit"
    md5: str = ""
    context: str = ""
    origin: str = ""


@dataclass
class ForwardItem:
    path: tuple[int, ...]
    sender: str = ""
    time_text: str = ""
    kind: str = "text"
    body: str = ""
    attachment_ordinal: int | None = None


def forward_items(value: str) -> tuple[ForwardItem, ...]:
    try:
        rows = json.loads(value)
        if not isinstance(rows, list) or len(rows) > MAX_FORWARD_ITEMS:
            raise ValueError("转发子项结构无效")
        return tuple(ForwardItem(**{**row, "path": tuple(row["path"])}) for row in rows)
    except (TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("转发子项结构无效")


@dataclass
class ChatMessage:
    external_id: str
    body: str
    sender: str = ""
    sent_at: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    reply_to: str | None = None
    forwarded: bool = False
    sender_id: str = ""
    from_me: bool | None = None
    message_type: str = ""
    display_body: str | None = None
    quote_body: str = ""
    quote_sender: str = ""
    forward_items: tuple[ForwardItem, ...] = ()

    def text_bytes(self) -> int:
        text = self.body + (self.display_body or "") + self.quote_body + self.quote_sender + self.sender_id + self.message_type
        text += "".join(item.body + item.sender + item.time_text for item in self.forward_items)
        return len(text.encode("utf-8")) + sum(len((a.context + a.origin).encode("utf-8")) for a in self.attachments)

    def validate(self) -> None:
        validate_body(self.body)
        if len(self.forward_items) > MAX_FORWARD_ITEMS or self.forward_items and not self.forwarded:
            raise ValueError("转发子项数量或类型无效")
        paths = set()
        nested = {item.path for item in self.forward_items if item.kind == "nested"}
        for item in self.forward_items:
            if (not 1 <= len(item.path) <= MAX_FORWARD_DEPTH or item.path in paths
                    or len(item.path) > 1 and item.path[:-1] not in nested
                    or any(type(i) is not int or not 1 <= i <= MAX_FORWARD_ITEMS for i in item.path)
                    or item.kind not in {"text", "file", "nested", "unknown"}):
                raise ValueError("转发子项身份无效")
            paths.add(item.path)
            for text in (item.body, item.sender, item.time_text):
                validate_body(text)
            if len(item.sender) > 200 or len(item.time_text) > 128:
                raise ValueError("转发子项元数据超限")
            ordinal = item.attachment_ordinal
            if ordinal is not None and (item.kind != "file" or type(ordinal) is not int or not 0 <= ordinal < len(self.attachments)):
                raise ValueError("转发附件身份无效")
        if self.from_me is not None and type(self.from_me) is not bool:
            raise ValueError("消息发送方向无效")
        for value in (self.sender_id, self.message_type, self.quote_sender):
            if not isinstance(value, str) or len(value) > 1024 or "\x00" in value:
                raise ValueError("消息展示身份无效")
        for value in (self.display_body, self.quote_body):
            if value is not None:
                validate_body(value)
        if not self.external_id or len(self.attachments) > (MAX_FORWARD_ITEMS if self.forwarded else 12):
            raise ValueError("消息标识或附件数量无效")
        for att in self.attachments:
            validate_body(att.context)
            validate_body(att.origin)
            if (att.context or att.origin) and not self.forwarded:
                raise ValueError("独立附件上下文必须来自合并转发")
        if len(self.body) + sum(len(a.context) + len(a.origin) for a in self.attachments) > MAX_MESSAGE_CHARS:
            raise ValueError("消息及转发附件上下文超过规范化安全上限")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_time(value: object) -> str | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (float, int)):
            dt = datetime.fromtimestamp(float(value), UTC)
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=LOCAL_TZ)
        return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    except (ValueError, OverflowError, OSError) as exc:
        raise ValueError("消息时间无效；应为 ISO 时间或秒级 Unix 时间戳") from exc


def mentioned_files(text: str) -> list[Attachment]:
    # Mentions are only candidates, never proof that a message sent a file.
    pattern = r"[^\s<>\"|/\\:：，。；【】\[\]]{1,120}\.(?:pdf|docx|xlsx|pptx|txt|md|csv|png|jpe?g|zip|7z)(?![a-zA-Z0-9])"
    names = list(dict.fromkeys(re.findall(pattern, text, flags=re.I)))[:12]
    return [Attachment(name=name, kind="mention") for name in names]


def parse_export(path: str | Path) -> tuple[dict, list[ChatMessage]]:
    path = Path(path)
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ValueError("单次聊天导入上限为 20 MB，请按会话或时间拆分")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not all(isinstance(row, dict) for row in records):
            raise ValueError("JSONL 每行必须是对象")
        header = records[0] if records and records[0].get("_type") == "header" else {}
        raw = [row for row in records if row.get("_type", "message") == "message"]
        meta = header.get("meta", {})
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise ValueError("JSON 应包含 messages 数组；支持 LocalMind 和 ChatLab 交换格式")
        meta, raw = data.get("meta", {}), data["messages"]
    else:
        raise ValueError("聊天记录支持 UTF-8 JSON/JSONL；客户端加密数据库不是可导入格式")
    if not isinstance(meta, dict) or not 0 < len(raw) <= MAX_MESSAGES:
        raise ValueError("每次需包含 1 至 20000 条消息")
    messages: list[ChatMessage] = []
    ids: set[str] = set()
    occurrences: dict[str, int] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("每条消息必须是 JSON 对象")
        body = row.get("text", row.get("content"))
        body = "" if body is None else body
        validate_body(body)
        sender = str(row.get("groupNickname") or row.get("accountName") or row.get("sender") or "")[:200]
        sent_at = normalize_time(row.get("timestamp", row.get("sent_at")))
        attachments = []
        values = row.get("attachments", [])
        if not isinstance(values, list) or len(values) > 12:
            raise ValueError("attachments 必须是最多 12 个对象的数组")
        for att in values:
            if not isinstance(att, dict):
                raise ValueError("附件必须是对象")
            hint = str(att.get("path") or "")[:2048]
            name = str(att.get("name") or PureWindowsPath(hint).name).strip()[:240]
            sha = str(att.get("sha256") or "").lower()
            size = att.get("size")
            if not name or (sha and not re.fullmatch(r"[a-f0-9]{64}", sha)) or (size is not None and (type(size) is not int or size < 0)):
                raise ValueError("附件名称、SHA-256 或字节大小无效")
            attachments.append(Attachment(name, hint, sha, size))
        if not attachments:
            attachments = mentioned_files(body)
        # Keep generated IDs stable across additive attachment-schema migrations.
        identity = digest(json.dumps([sender, sent_at, body, [{k: v for k, v in asdict(a).items() if k not in {"context", "origin"}} for a in attachments]], ensure_ascii=False, sort_keys=True))
        occurrences[identity] = occurrences.get(identity, 0) + 1
        if not values and row.get("type") == 4 and body:
            name = body.removeprefix("[文件]").strip()
            if name:
                attachments = [Attachment(name=name[:240])]
        ext_id = str(row.get("platformMessageId") or row.get("id") or row.get("messageId") or f"{identity}:{occurrences[identity]}")[:256]
        if ext_id in ids:
            raise ValueError("导入数据包含重复消息 ID；未执行导入")
        ids.add(ext_id)
        reply = row.get("replyToMessageId") or row.get("reply_to")
        messages.append(ChatMessage(ext_id, body, sender, sent_at, attachments, str(reply) if reply else None))
    return meta, messages


class ChatStore:
    def __init__(self, db: LocalMindDB):
        self.db = db

    def sources(self) -> list[dict]:
        return [dict(row) for row in self.db.conn.execute("""
            SELECT s.*, COUNT(m.id) AS messages FROM chat_sources s
            LEFT JOIN chat_messages m ON m.source_id=s.id GROUP BY s.id ORDER BY s.updated_at DESC
        """)]

    def attachments(self, source_id: int | None = None, *, attachment_id: int | None = None) -> list[dict]:
        sql = """SELECT a.*,a.id AS attachment_id,a.name AS attachment_name,
            s.platform,s.account,s.source_key,s.conversation,m.external_id,m.sender,m.sent_at,
            COALESCE(f.path,l.path) AS local_path,f.parse_status,f.ocr_status,f.error AS content_error,
            (SELECT COUNT(*) FROM chunks c WHERE c.file_id=f.id) AS content_chunks,
            (SELECT COUNT(*) FROM chunks c JOIN vectors v ON v.chunk_id=c.id
                WHERE c.file_id=f.id AND v.model_name=(SELECT value FROM settings WHERE key='embedding_model')) AS content_vectors
            FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
            JOIN chat_sources s ON s.id=m.source_id LEFT JOIN files f ON f.id=a.file_id
            LEFT JOIN chat_attachment_locations l ON l.attachment_id=a.id"""
        params, conditions = [], []
        if source_id is not None:
            conditions.append("s.id=?")
            params.append(source_id)
        if attachment_id is not None:
            conditions.append("a.id=?")
            params.append(attachment_id)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return [dict(row) for row in self.db.conn.execute(sql + " ORDER BY a.id DESC LIMIT 500", params)]

    @staticmethod
    def _source_conditions(query):
        chat_dates = query and (query.scope in {"chat_files", "messages"}
                               or query.chat_platform or query.conversation or query.sender)
        return chat_predicates(query, include_dates=bool(chat_dates)) if query else ([], [])

    def file_sources(self, file_id: int, query=None, *, cursor=None, limit=25, text=""):
        """Read one page of linked attachment metadata; never match by filename."""
        if type(file_id) is not int or file_id <= 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("无效的来源分页参数")
        conditions, params = self._source_conditions(query)
        conditions.insert(0, "a.file_id=?")
        params.insert(0, file_id)
        text = text.strip()
        if text:
            conditions.append("(instr(lower(s.conversation),lower(?))>0 OR instr(lower(m.sender),lower(?))>0)")
            params.extend([text, text])
        # Unknown times sort last; attachment ID makes equal timestamps stable.
        moment = "coalesce(localmind_epoch(m.sent_at),-1e20)"
        if cursor is not None:
            conditions.append(f"({moment},a.id)<(?,?)")
            params.extend(cursor)
        rows = [dict(row) for row in self.db.conn.execute(f"""
            SELECT a.id AS attachment_id,m.id AS message_id,m.sent_at,m.sender,
                s.platform,s.account,s.conversation,a.name,{moment} AS sort_time
            FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
            JOIN chat_sources s ON s.id=m.source_id
            WHERE 1=1{where(conditions)}
            ORDER BY sort_time DESC,a.id DESC LIMIT ?""", params + [limit + 1])]
        more = len(rows) > limit
        rows = rows[:limit]
        return rows, (rows[-1]["sort_time"], rows[-1]["attachment_id"]) if more else None

    def attachment_context(self, attachment_id, *, file_id=None, query=None, source=None, file_path=None):
        conditions, params = self._source_conditions(query)
        if file_id is not None:
            conditions.append("a.file_id=?")
            params.append(file_id)
        if file_path is not None:
            conditions.append("f.path=?")
            params.append(file_path)
        if source is not None:
            # Validate identity in the same read as the body, even if another process synced meanwhile.
            for column, value in (("s.platform", source.platform), ("s.account", source.account),
                                  ("s.source_key", source.source_key), ("m.external_id", source.external_id),
                                  ("m.id", source.message_id), ("a.id", source.attachment_id),
                                  ("a.name", source.attachment_name), ("a.origin", source.origin)):
                conditions.append(f"{column}=?")
                params.append(value)
        row = self.db.conn.execute("""SELECT s.platform,s.account,s.conversation,m.sender,m.sent_at,m.observed_at,
            m.external_id,m.id AS message_id,substr(m.body,1,4000) AS body,length(m.body)>4000 AS body_truncated,
            a.id AS attachment_id,a.file_id AS target_id,a.link_state,
            a.linked_hash,f.file_hash AS target_hash,substr(a.context,1,4000) AS attachment_context,
            length(a.context)>4000 AS context_truncated,a.origin,l.path AS resolved_path
            FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
            JOIN chat_sources s ON s.id=m.source_id LEFT JOIN files f ON f.id=a.file_id
            LEFT JOIN chat_attachment_locations l ON l.attachment_id=a.id WHERE a.id=?""" + where(conditions),
            [attachment_id] + params).fetchone()
        return context_label(dict(row)) if row else ""

    def _candidates(self, att: Attachment) -> tuple[int | None, str]:
        if att.kind == "explicit" and att.path and (Path(att.path).is_absolute() or PureWindowsPath(att.path).is_absolute()) and not att.path.startswith(("\\\\", "//", CHAT_PREFIX)):
            row = self.db.conn.execute("SELECT id, file_hash, size FROM files WHERE path=? COLLATE NOCASE", (os.path.normpath(att.path),)).fetchone()
            if row and (not att.sha256 or att.sha256 == row["file_hash"]) and (att.size is None or att.size == row["size"]):
                return int(row["id"]), "path"
        if att.kind == "explicit" and att.sha256:
            rows = self.db.conn.execute("SELECT id FROM files WHERE file_hash=? AND path NOT LIKE 'localmind-chat://%' LIMIT 2", (att.sha256,)).fetchall()
            if len(rows) == 1:
                return int(rows[0]["id"]), "hash"
            if rows:
                return None, "ambiguous"
        sql = "SELECT id FROM files WHERE name=? COLLATE NOCASE AND path NOT LIKE 'localmind-chat://%'"
        params: list = [att.name]
        if att.size is not None:
            sql += " AND size=?"
            params.append(att.size)
        rows = self.db.conn.execute(sql + " LIMIT 2", params).fetchall()
        if att.kind == "explicit" and att.md5:
            candidates = self.db.conn.execute(sql + " LIMIT 11", params).fetchall()
            if len(candidates) > 10:
                return None, "ambiguous"
            matches = [int(row["id"]) for row in candidates if self._file_md5(int(row["id"])) == att.md5]
            if len(matches) == 1:
                return matches[0], "hash"
            if len(matches) > 1:
                return None, "ambiguous"
        return None, "ambiguous" if len(rows) > 1 else "candidate" if rows else "missing"

    def _file_md5(self, file_id: int) -> str | None:
        row = self.db.get_file(file_id)
        path = Path(row["path"])
        if str(path).startswith(("\\\\", "//")):
            return None
        try:
            before = path.stat()
            if before.st_size > 50 * 1024 * 1024:
                return None
            cached = self.db.conn.execute("SELECT * FROM chat_file_digests WHERE file_id=? AND sha256=?", (file_id, row["file_hash"])).fetchone()
            modified = datetime.fromtimestamp(before.st_mtime).isoformat(timespec="microseconds")
            if cached and before.st_size == row["size"] and modified == row["modified_at"]:
                return cached["md5"]
            md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    md5.update(block)
                    sha.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha.hexdigest() != row["file_hash"]:
                return None
            self.db.conn.execute("INSERT INTO chat_file_digests(file_id,sha256,md5) VALUES(?,?,?) ON CONFLICT(file_id) DO UPDATE SET sha256=excluded.sha256,md5=excluded.md5", (file_id, row["file_hash"], md5.hexdigest()))
            return md5.hexdigest()
        except OSError:
            return None

    def resolve_links(self, source_id: int | None = None, *, attachment_ids: list[int] | None = None) -> int:
        if attachment_ids is not None and not attachment_ids:
            return 0
        updated = 0
        with self.db.conn:
            sql = """SELECT a.*,l.path AS resolved_path,l.signature AS resolved_signature,l.sha256 AS resolved_sha
                FROM chat_attachments a LEFT JOIN chat_attachment_locations l ON l.attachment_id=a.id
                WHERE a.file_id IS NULL"""
            params = []
            if source_id is not None:
                sql += " AND a.message_id IN (SELECT id FROM chat_messages WHERE source_id=?)"
                params.append(source_id)
            if attachment_ids is not None:
                sql += f" AND a.id IN ({','.join('?' for _ in attachment_ids)})"
                params.extend(attachment_ids)
            rows = self.db.conn.execute(sql, params).fetchall()
            for row in rows:
                att = Attachment(row["name"], row["path_hint"], row["sha256"], row["size"], row["kind"], row["md5"])
                signature = digest(json.dumps([row[k] for k in ("message_id", "name", "size", "md5", "sha256", "origin")]))
                if row["resolved_path"] and row["resolved_signature"] == signature:
                    att = Attachment(row["name"], row["resolved_path"], row["resolved_sha"])
                file_id, state = self._candidates(att)
                target = self.db.get_file(file_id) if file_id is not None else None
                if state != row["link_state"] or file_id is not None:
                    self.db.conn.execute("UPDATE chat_attachments SET file_id=?, link_state=?, linked_hash=? WHERE id=?", (file_id, state, target["file_hash"] if target else None, row["id"]))
                    updated += 1
        return updated

    def bind(self, attachment_id: int, file_id: int) -> None:
        target = self.db.get_file(file_id)
        if not target or str(target["path"]).startswith(CHAT_PREFIX) or not Path(target["path"]).is_file():
            raise ValueError("关联目标必须是已索引且存在的本地文件")
        with self.db.conn:
            cursor = self.db.conn.execute("UPDATE chat_attachments SET file_id=?, link_state='manual', linked_hash=? WHERE id=?", (file_id, target["file_hash"], attachment_id))
            if not cursor.rowcount:
                raise ValueError("附件记录不存在")

    def remove_source(self, source_id: int) -> None:
        with self.db.conn:
            self.db.conn.execute("DELETE FROM settings WHERE key LIKE 'weflow_batch:%'")
            source = self.db.conn.execute("SELECT source_key FROM chat_sources WHERE id=?", (source_id,)).fetchone()
            if source:
                self.db.conn.execute("DELETE FROM settings WHERE key=?", ("weflow_checkpoint:" + digest(source[0]),))
                self.db.conn.execute("DELETE FROM chat_sync_issues WHERE source_key=?", (source[0],))
            owners = self.db.conn.execute("SELECT DISTINCT u.owner_file_id FROM chat_units u JOIN chat_messages m ON m.id=u.message_id WHERE m.source_id=?", (source_id,)).fetchall()
            for owner in owners:
                file_id = int(owner[0])
                self.db._delete_file_index_no_commit(file_id)
                self.db.conn.execute("DELETE FROM files WHERE id=?", (file_id,))
            self.db.conn.execute("DELETE FROM chat_sources WHERE id=?", (source_id,))

    def ingest(self, *, platform: str, account: str, conversation: str, source_key: str,
               messages: list[ChatMessage], kind: str = "import", embedding_service: EmbeddingService | None = None,
               should_cancel=lambda: False, refresh_existing: bool = False) -> dict:
        if platform != "wechat" or not account.strip() or not conversation.strip():
            raise ValueError("请填写平台、账号标签和会话名称，避免不同账号或聊天串联")
        if (not messages and not refresh_existing) or len(messages) > MAX_MESSAGES:
            raise ValueError("消息数应为 1 至 20000")
        old_source = self.db.conn.execute("SELECT * FROM chat_sources WHERE source_key=?", (source_key,)).fetchone()
        if refresh_existing and old_source is None:
            raise ValueError("本地会话索引已移除，请刷新列表")
        if old_source and ((old_source["platform"], old_source["account"]) != (platform, account) or (kind != "weflow_api" and old_source["conversation"] != conversation)):
            raise ValueError("数据源身份发生变化，请使用新的数据源")
        merged: dict[str, ChatMessage] = {}
        msg_ids = {}
        att_ids = {}
        if old_source:
            old_attachments: dict[int, list[Attachment]] = {}
            for a in self.db.conn.execute("SELECT a.* FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id WHERE m.source_id=? ORDER BY a.ordinal", (old_source["id"],)):
                old_attachments.setdefault(a["message_id"], []).append(Attachment(a["name"], a["path_hint"], a["sha256"], a["size"], a["kind"], a["md5"], a["context"], a["origin"]))
            for row in self.db.conn.execute("SELECT * FROM chat_messages WHERE source_id=? ORDER BY id", (old_source["id"],)):
                atts = old_attachments.get(row["id"], [])
                msg_ids[row["external_id"]] = row["id"]
                merged[row["external_id"]] = ChatMessage(row["external_id"], row["body"], row["sender"], row["sent_at"], atts, row["reply_to"], bool(row["forwarded"]),
                    row["sender_id"], bool(row["from_me"]) if row["from_me"] is not None else None,
                    row["message_type"], row["display_body"], row["quote_body"], row["quote_sender"], forward_items(row["forward_json"]))
            for a in self.db.conn.execute("SELECT a.id,a.ordinal,m.external_id FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id WHERE m.source_id=?", (old_source["id"],)):
                att_ids[(a["external_id"], a["ordinal"])] = a["id"]
        changed_messages = set()
        for msg in messages:
            msg.validate()
            if merged.get(msg.external_id) != msg:
                changed_messages.add(msg.external_id)
                merged[msg.external_id] = msg
        if len(merged) > MAX_MESSAGES:
            raise ValueError("单个数据源累计上限 20000 条；请移除该会话索引后选择更短范围，或使用分段离线导入")
        if sum(m.text_bytes() for m in merged.values()) > MAX_IMPORT_BYTES:
            raise ValueError("单会话消息正文超过 20 MB，请缩小同步范围")
        ordered = sorted(merged.values(), key=lambda m: (m.sent_at or "9999", m.external_id))
        existing_units = {r["unit_key"]: dict(r) for r in self.db.conn.execute("SELECT u.unit_key, u.fingerprint, v.model_name FROM chat_units u JOIN chat_messages m ON m.id=u.message_id LEFT JOIN vectors v ON v.chunk_id=u.chunk_id WHERE m.source_id=?", (old_source["id"] if old_source else -1,))}
        if should_cancel():
            raise RuntimeError("同步已取消，未修改索引")
        model_name = embedding_service.active_model_name if embedding_service else None
        indexed_model = self.db.get_setting("embedding_model")
        if embedding_service and indexed_model and indexed_model != model_name and self.db.vector_models():
            raise ValueError("聊天与文件的向量模型不一致；请恢复相同模型后重试")
        pending = []
        valid_keys = set()
        replies: dict[str, list[ChatMessage]] = {}
        for msg in ordered:
            if msg.reply_to:
                replies.setdefault(msg.reply_to, []).append(msg)
        for i, msg in enumerate(ordered):
            if should_cancel():
                raise RuntimeError("同步已取消，未修改索引")
            neighbors = []
            if msg.sent_at:
                moment = datetime.fromisoformat(msg.sent_at.replace("Z", "+00:00"))
                for other in ordered[max(0, i - 2):i + 3]:
                    if other is not msg and other.sent_at and abs((datetime.fromisoformat(other.sent_at.replace("Z", "+00:00")) - moment).total_seconds()) <= 300:
                        neighbors.append(f"{other.sender or '未知发送者'}: {'[合并转发]' if other.forwarded else other.body[:180]}")
            if msg.reply_to in merged:
                reply = merged[msg.reply_to]
                neighbors.insert(0, f"引用 {reply.sender}: {'[合并转发]' if reply.forwarded else reply.body[:240]}")
            for reply in replies.get(msg.external_id, [])[-3:]:
                neighbors.insert(0, f"回复此消息 {reply.sender}: {'[合并转发]' if reply.forwarded else reply.body[:240]}")
            units = list(enumerate(msg.attachments))
            if not units or msg.forwarded:
                units.append((-1 if msg.forwarded else 0, None))
            for ordinal, att in units:
                name = att.name if att else f"{conversation} - {msg.sender or '聊天记录'}"
                body = f"平台: {platform}\n会话: {conversation}\n发送者: {msg.sender or '未识别'}\n消息时间: {msg.sent_at or '未知（不是采集时间）'}\n"
                if att:
                    body += f"{'附件' if att.kind == 'explicit' else '文件名提及，待确认'}: {att.name}\n"
                    if att.origin:
                        body += att.origin + "\n"
                passages = []
                message_body = att.context if att and msg.forwarded else msg.body
                for part_index, (part, _, _) in enumerate(split_text(message_body or "[非文本消息]", min_size=200, max_size=600, overlap=60)):
                    passage = body + "消息: " + part
                    if neighbors and (not msg.forwarded or att is None):
                        passage += "\n上下文（引用 / 相邻消息，非附件内容）:\n" + "\n".join(neighbors[:5])
                    passages.append((str(part_index), passage))
                if att and not msg.forwarded:
                    # A direct reply proves which attachment the long description
                    # belongs to; preserve its tail without widening adjacency.
                    for reply in replies.get(msg.external_id, [])[-3:]:
                        if len(reply.body) <= 240 or reply.forwarded:
                            continue
                        for part_index, (part, _, _) in enumerate(split_text(reply.body, min_size=200, max_size=600, overlap=60)):
                            context = f"回复此附件（不是附件正文） {reply.sender} / {reply.sent_at or '时间未知'}:\n{part}"
                            passages.append((f"reply:{digest(reply.external_id)}:{part_index}", body + context))
                # Split the complete input, including metadata and neighboring messages.
                # Shortening only the message body still leaves the embedding tail truncated.
                windows = [(part_key if n == 0 else f"{part_key}:window:{n}", text)
                           for part_key, passage in passages
                           for n, (text, _, _) in enumerate(split_text(passage))]
                for part_key, passage in windows:
                    key = digest(source_key + "\0" + msg.external_id + "\0" + str(ordinal) + "\0" + part_key)
                    valid_keys.add(key)
                    fp = digest(passage)
                    old = existing_units.get(key)
                    if not old or old["fingerprint"] != fp or (model_name and old["model_name"] != model_name):
                        pending.append((msg, ordinal, att, name, passage, key, fp))
        vectors = None
        if pending and embedding_service:
            batches = []
            for offset in range(0, len(pending), 128):
                if should_cancel():
                    raise RuntimeError("同步已取消，未修改索引")
                batches.append(embedding_service.encode_documents([unit[4] for unit in pending[offset:offset + 128]]))
            vectors = np.vstack(batches)
            if vectors.shape[0] != len(pending) or not np.isfinite(vectors).all():
                raise ValueError("聊天向量生成失败；未修改已有聊天索引")
        if should_cancel():
            raise RuntimeError("同步已取消，未修改索引")
        if old_source and not changed_messages and not pending and len(valid_keys) == len(existing_units) and old_source["conversation"] == conversation:
            self.resolve_links(int(old_source["id"]))
            return {"source_id": old_source["id"], "received": len(messages), "total": len(merged), "updated_units": 0, "reused_units": len(valid_keys)}
        now = utc_now_iso()
        with self.db.conn:
            self.db.conn.execute("""INSERT INTO chat_sources(source_key, platform, account, conversation, kind, updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET updated_at=excluded.updated_at,conversation=excluded.conversation""", (source_key, platform, account, conversation, kind, now))
            source_id = int(self.db.conn.execute("SELECT id FROM chat_sources WHERE source_key=?", (source_key,)).fetchone()[0])
            for msg in (msg for key, msg in merged.items() if key in changed_messages):
                self.db.conn.execute("""INSERT INTO chat_messages(source_id,external_id,sender,sent_at,body,observed_at,reply_to,forwarded,
                    sender_id,from_me,message_type,display_body,quote_body,quote_sender,sent_epoch,forward_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,localmind_epoch(?),?) ON CONFLICT(source_id,external_id) DO UPDATE SET
                    sender=excluded.sender,sent_at=excluded.sent_at,body=excluded.body,reply_to=excluded.reply_to,forwarded=excluded.forwarded,
                    sender_id=excluded.sender_id,from_me=excluded.from_me,message_type=excluded.message_type,
                    display_body=excluded.display_body,quote_body=excluded.quote_body,quote_sender=excluded.quote_sender,forward_json=excluded.forward_json""",
                    (source_id,msg.external_id,msg.sender,msg.sent_at,msg.body,now,msg.reply_to,int(msg.forwarded),
                     msg.sender_id,msg.from_me,msg.message_type,msg.display_body,msg.quote_body,msg.quote_sender,msg.sent_at,
                     json.dumps([asdict(item) for item in msg.forward_items], ensure_ascii=False)))
                mid = int(self.db.conn.execute("SELECT id FROM chat_messages WHERE source_id=? AND external_id=?", (source_id, msg.external_id)).fetchone()[0])
                msg_ids[msg.external_id] = mid
                for ordinal, att in enumerate(msg.attachments):
                    old_att = self.db.conn.execute("SELECT * FROM chat_attachments WHERE message_id=? AND ordinal=?", (mid, ordinal)).fetchone()
                    same = old_att and (old_att["name"], old_att["path_hint"], old_att["sha256"], old_att["size"], old_att["kind"], old_att["md5"], old_att["origin"]) == (att.name, att.path, att.sha256, att.size, att.kind, att.md5, att.origin)
                    fid, state = (old_att["file_id"], old_att["link_state"]) if same else self._candidates(att)
                    file_row = self.db.get_file(fid) if fid else None
                    self.db.conn.execute("""INSERT INTO chat_attachments(message_id,ordinal,name,path_hint,sha256,size,kind,file_id,link_state,linked_hash,md5,context,origin)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(message_id,ordinal) DO UPDATE SET
                        name=excluded.name,path_hint=excluded.path_hint,sha256=excluded.sha256,size=excluded.size,kind=excluded.kind,
                        file_id=excluded.file_id,link_state=excluded.link_state,linked_hash=excluded.linked_hash,md5=excluded.md5,context=excluded.context,origin=excluded.origin""", (mid, ordinal, att.name, att.path, att.sha256, att.size, att.kind, fid, state, old_att["linked_hash"] if same else file_row["file_hash"] if file_row else None, att.md5, att.context, att.origin))
                    att_ids[(msg.external_id, ordinal)] = int(self.db.conn.execute("SELECT id FROM chat_attachments WHERE message_id=? AND ordinal=?", (mid, ordinal)).fetchone()[0])
            owners = self.db.conn.execute("SELECT u.unit_key, u.owner_file_id FROM chat_units u JOIN chat_messages m ON m.id=u.message_id WHERE m.source_id=?", (source_id,)).fetchall()
            for owner in owners:
                if owner["unit_key"] not in valid_keys:
                    self.db._delete_file_index_no_commit(owner["owner_file_id"])
                    self.db.conn.execute("DELETE FROM files WHERE id=?", (owner["owner_file_id"],))
            for index, (msg, ordinal, att, name, body, key, fp) in enumerate(pending):
                path = CHAT_PREFIX + key
                old = self.db.get_file_by_path(path)
                if old:
                    fid = int(old["id"])
                    self.db._delete_file_index_no_commit(fid)
                    self.db.conn.execute("UPDATE files SET name=?,ext=?,modified_at=?,file_hash=? WHERE id=?", (name, PureWindowsPath(name).suffix.lstrip(".").lower() if att else "chat", msg.sent_at or "", fp, fid))
                else:
                    cur = self.db.conn.execute("""INSERT INTO files(path,name,ext,size,created_at,modified_at,file_hash,scan_status,parse_status,ocr_status,needs_ocr,indexed_at)
                        VALUES(?,?,?,?,?,?,?,'chat','parsed','not_required',0,?)""", (path, name, PureWindowsPath(name).suffix.lstrip(".").lower() if att else "chat", len(body.encode("utf-8")), now, msg.sent_at or "", fp, now))
                    fid = int(cur.lastrowid)
                cur = self.db.conn.execute("INSERT INTO chunks(file_id,chunk_text,source_type,created_at) VALUES(?,?,'chat',?)", (fid, body, now))
                cid = int(cur.lastrowid)
                self.db.conn.execute("INSERT INTO chunk_fts(rowid,file_name,file_path,chunk_text,source_type) VALUES(?,?,?,?,'chat')", (cid, name, conversation, body))
                self.db.conn.execute("INSERT INTO chat_units(chunk_id,owner_file_id,message_id,attachment_id,unit_key,fingerprint) VALUES(?,?,?,?,?,?)", (cid, fid, msg_ids[msg.external_id], att_ids.get((msg.external_id, ordinal)) if att else None, key, fp))
                if vectors is not None:
                    vector = np.asarray(vectors[index], dtype="float32")
                    self.db.conn.execute("INSERT INTO vectors(chunk_id,vector_id,model_name,vector_dim,embedding) VALUES(?,?,?,?,?)", (cid, cid, model_name, len(vector), vector.tobytes()))
            for msg in (merged[key] for key in changed_messages):
                self.db.conn.execute("DELETE FROM chat_attachments WHERE message_id=? AND ordinal>=?", (msg_ids[msg.external_id], len(msg.attachments)))
            if model_name and not indexed_model:
                self.db.conn.execute("INSERT INTO settings(key,value) VALUES('embedding_model',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (model_name,))
        self.resolve_links(source_id)
        return {"source_id": source_id, "received": len(messages), "total": len(merged), "updated_units": len(pending), "reused_units": len(valid_keys) - len(pending)}

    def enrich(self, rows: dict[int, sqlite3.Row], query=None) -> dict[int, dict]:
        if not rows:
            return {}
        ids = list(rows)
        fids = list({int(r["file_id"]) for r in rows.values()})
        conditions, params = chat_predicates(query) if query else ([], [])
        columns = """m.id AS message_id,m.external_id,m.sender,m.sent_at,
            substr(m.body,1,4000) AS body,length(m.body)>4000 AS body_truncated,m.observed_at,
            s.platform,s.account,s.source_key,s.conversation,s.kind,a.id AS attachment_id,a.file_id AS target_id,
            substr(a.context,1,4000) AS attachment_context,length(a.context)>4000 AS context_truncated,a.origin,
            a.link_state,a.linked_hash,a.name AS attachment_name,f.path AS target_path,f.name AS target_name,l.path AS resolved_path,
            f.ext AS target_ext,f.modified_at AS target_modified,f.file_hash AS target_hash"""
        own_refs = [dict(r) for r in self.db.conn.execute(f"""SELECT u.chunk_id,u.owner_file_id,{columns}
            FROM chat_units u JOIN chat_messages m ON m.id=u.message_id JOIN chat_sources s ON s.id=m.source_id
            LEFT JOIN chat_attachments a ON a.id=u.attachment_id LEFT JOIN files f ON f.id=a.file_id
            LEFT JOIN chat_attachment_locations l ON l.attachment_id=a.id
            WHERE u.chunk_id IN ({','.join('?' for _ in ids)}){where(conditions)}""", ids + params)]
        conditions, params = self._source_conditions(query)
        # One row per attachment, not per context chunk; filter before choosing recent sources.
        refs = [dict(r) for r in self.db.conn.execute(f"""WITH refs AS (
            SELECT {columns},
                ROW_NUMBER() OVER (PARTITION BY a.file_id ORDER BY localmind_epoch(m.sent_at) DESC,m.id DESC,a.id DESC) AS ref_rank
            FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
            JOIN chat_sources s ON s.id=m.source_id JOIN files f ON f.id=a.file_id
            LEFT JOIN chat_attachment_locations l ON l.attachment_id=a.id
            WHERE a.file_id IN ({','.join('?' for _ in fids)}){where(conditions)})
            SELECT * FROM refs WHERE ref_rank<=3 ORDER BY target_id,ref_rank""", fids + params)]
        by_chunk = {r["chunk_id"]: r for r in own_refs}
        by_target: dict[int, list[dict]] = {}
        for ref in refs:
            if ref["target_id"]:
                by_target.setdefault(ref["target_id"], []).append(ref)
        result = {}
        for cid, raw in rows.items():
            row = dict(raw)
            own = by_chunk.get(cid)
            row["chat_refs"] = [own] if own else by_target.get(row["file_id"], [])
            if own and own["target_path"] and (not query or query.scope != "messages"):
                row.update(file_id=own["target_id"], name=own["target_name"], path=own["target_path"], ext=own["target_ext"], modified_at=own["target_modified"])
            result[cid] = row
        return result

    def scope_ids(self, query) -> set[int] | None:
        if query.scope not in {"all", "files", "chat_files", "messages"}:
            raise ValueError("无效的检索范围")
        if query.evidence not in {"all", "content", "chat"} or (query.scope != "chat_files" and query.evidence != "all"):
            raise ValueError("线索类型筛选仅支持微信附件范围")
        identity_filter = bool(query.chat_platform or query.conversation or query.sender)
        constrained = identity_filter or query.file_type or query.path_contains or query.modified_after or query.modified_before
        if query.scope == "all" and not constrained:
            return None
        ids: set[int] = set()
        if query.scope in {"all", "files"} and not identity_filter:
            filters, params = file_predicates(query)
            dates, moments = interval_predicates("f.modified_at", query)
            ids.update(int(r[0]) for r in self.db.conn.execute(
                "SELECT c.id FROM chunks c JOIN files f ON f.id=c.file_id WHERE f.path NOT LIKE 'localmind-chat://%'"
                + where(filters + dates), params + moments))
        elif query.scope != "messages" and query.evidence != "chat":
            filters, params = chat_predicates(query)
            content_filters, content_params = file_predicates(query)
            ids.update(int(r[0]) for r in self.db.conn.execute(
                "SELECT c.id FROM chunks c JOIN files f ON f.id=c.file_id "
                "JOIN chat_attachments a ON a.file_id=f.id JOIN chat_messages m ON m.id=a.message_id "
                "JOIN chat_sources s ON s.id=m.source_id WHERE 1=1" + where(filters + content_filters),
                params + content_params))
        if query.scope != "files" and query.evidence != "content":
            attachment_only = " AND u.attachment_id IS NOT NULL" if query.scope == "chat_files" else ""
            filters, params = chat_predicates(query)
            ext, path = ("f.ext", "f.path") if query.scope == "messages" else ("coalesce(target.ext,f.ext)", "coalesce(target.path,f.path)")
            content_filters, content_params = file_predicates(query, ext=ext, path=path)
            ids.update(int(r[0]) for r in self.db.conn.execute(
                "SELECT u.chunk_id FROM chat_units u JOIN files f ON f.id=u.owner_file_id "
                "JOIN chat_messages m ON m.id=u.message_id JOIN chat_sources s ON s.id=m.source_id "
                "LEFT JOIN chat_attachments a ON a.id=u.attachment_id LEFT JOIN files target ON target.id=a.file_id "
                "WHERE 1=1" + attachment_only + where(filters + content_filters), params + content_params))
        return ids


def chat_source_reference(ref):
    from .models import ChatSourceReference

    if ref is None:
        return None
    return ChatSourceReference(
        message_id=int(ref["message_id"]), attachment_id=ref.get("attachment_id"),
        **{key: str(ref.get(key) or "") for key in (
            "platform", "account", "source_key", "external_id", "conversation", "sender",
            "sent_at", "attachment_name", "origin",
        )},
    )


def context_label(ref: dict) -> str:
    status = {"manual": "人工确认", "path": "路径匹配", "hash": "哈希匹配", "candidate": "同名候选，待确认", "ambiguous": "多个候选，待确认", "missing": "未找到本地文件"}.get(ref.get("link_state"), "聊天记录")
    if ref.get("attachment_id") and not ref.get("target_id"):
        status = status if status in {"同名候选，待确认", "多个候选，待确认"} else "未找到本地文件"
    if ref.get("target_id") and ref.get("linked_hash") and ref.get("target_hash") != ref["linked_hash"]:
        status += "；本地内容已变化"
    if ref.get("resolved_path"):
        status = "已记录本地路径，打开时复核"
    origin = "\n" + ref["origin"] if ref.get("origin") else ""
    body = ref.get("attachment_context") or ref["body"]
    truncated = len(body) > 4000 or (ref.get("context_truncated") if ref.get("attachment_context") else ref.get("body_truncated"))
    body = body[:4000] + ("\n…（聊天来源节选）" if truncated else "")
    return f"微信 / {ref['account']} / {ref['conversation']}\n发送者: {ref['sender'] or '未识别'}\n消息时间: {ref['sent_at'] or '未知'}\n采集时间: {ref['observed_at']}\n消息 ID: {ref.get('external_id', ref['message_id'])}{origin}\n关联: {status}\n\n{body}"
