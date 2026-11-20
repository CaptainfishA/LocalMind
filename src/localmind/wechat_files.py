"""On-demand attachment locations. Never treats a filename mention as a file."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path, PureWindowsPath

from .chats import CHAT_PREFIX, digest
from .config import MAX_FILE_SIZE_BYTES
from .weflow import SyncCancelled, WeFlowError

RESOLVE_TOOL = "alpha.messages.resolve_search_hit"


def api_identity(row):
    if not isinstance(row, dict):
        return None
    api_id = row.get("id")
    stamp, source = str(row.get("sent_at_unix_ms", "")), str(row.get("source_type", ""))
    conversation = row.get("conversation_id")
    if (not isinstance(api_id, str) or not re.fullmatch(r"msg-[0-9a-f]{64}", api_id)
            or not re.fullmatch(r"[0-9]{1,19}", stamp) or not re.fullmatch(r"[0-9]{1,20}", source)
            or not isinstance(conversation, str) or not 0 < len(conversation) <= 1024):
        return None
    external = str(row.get("platform_message_id") or row.get("server_id") or "")
    external = external if external and external != "0" else "alpha:" + api_id
    return external, (conversation, api_id, stamp, source)


def save_api_identities(db, source_key, base_url, account_id, identities):
    if not account_id:
        return
    for external, values in identities.items():
        db.conn.execute("""INSERT INTO chat_api_messages
            SELECT m.id,?,?,?,?,?,? FROM chat_messages m JOIN chat_sources s ON s.id=m.source_id
            WHERE s.source_key=? AND m.external_id=?
            ON CONFLICT(message_id) DO UPDATE SET base_url=excluded.base_url,account_id=excluded.account_id,
            conversation_id=excluded.conversation_id,api_id=excluded.api_id,
            timestamp_ms=excluded.timestamp_ms,source_type=excluded.source_type""",
            (base_url, account_id, *values, source_key, external))


def local_file(value):
    if (not isinstance(value, str) or "\x00" in value or value.startswith(("\\\\", "//"))
            or "://" in value or (os.name == "nt" and (not PureWindowsPath(value).is_absolute()
            or ":" in value[2:]))):
        raise WeFlowError("附件路径不是受支持的本地文件", code="file_path_invalid")
    path = Path(value)
    if not path.is_absolute():
        raise WeFlowError("附件路径必须为绝对路径", code="file_path_invalid")
    try:
        path = path.resolve(strict=True)
        if str(path).startswith(("\\\\", "//")) or not path.is_file():
            raise OSError()
    except OSError:
        raise WeFlowError("本地附件已移动或删除，请重新定位", code="file_missing") from None
    return path


def file_digest(path, cancel=lambda: False):
    before = path.stat()
    if before.st_size > MAX_FILE_SIZE_BYTES:
        raise WeFlowError("附件超过当前 50 MB 校验上限，可打开所在目录后手动查看", code="file_limit")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            if cancel():
                raise SyncCancelled("附件定位已取消")
            sha.update(block)
            md5.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise WeFlowError("附件在校验期间发生变化，请重试", code="file_changed")
    return sha.hexdigest(), md5.hexdigest(), after.st_size


class AttachmentLocator:
    def __init__(self, db, client=None, *, cache_dir, root="", cancel=lambda: False, progress=lambda _: None,
                 client_factory=None):
        self.db, self.client = db, client
        self.cache_dir, self.root = Path(cache_dir), root
        self.cancel, self.progress = cancel, progress
        self.client_factory = client_factory
        self.sequence = 0

    def check_cancel(self):
        if self.cancel():
            raise SyncCancelled("附件定位已取消")

    def attachment(self, attachment_id):
        row = self.db.conn.execute("""SELECT a.*,m.external_id,m.sent_at,m.source_id,
            s.source_key,s.account,s.kind AS source_kind,f.path AS indexed_path
            FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
            JOIN chat_sources s ON s.id=m.source_id LEFT JOIN files f ON f.id=a.file_id WHERE a.id=?""",
            (attachment_id,)).fetchone()
        if row is None:
            raise WeFlowError("附件索引已移除，请重新搜索", code="attachment_missing")
        return dict(row)

    @staticmethod
    def signature(att):
        return digest(json.dumps([att[k] for k in ("message_id", "name", "size", "md5", "sha256", "origin")]))

    def remember(self, att, path, origin, *, expected_sha=""):
        path = local_file(str(path))
        sha, md5, size = file_digest(path, self.cancel)
        if expected_sha and sha != expected_sha:
            raise WeFlowError("附件缓存内容已变化，请重新定位", code="file_changed")
        if origin != "manual" and ((att["size"] is not None and att["size"] != size)
                                  or (att["md5"] and att["md5"].lower() != md5)
                                  or (att["sha256"] and att["sha256"].lower() != sha)):
            raise WeFlowError("附件大小或哈希不匹配，未打开文件", code="file_mismatch")
        self.check_cancel()
        with self.db.conn:
            self.db.conn.execute("""INSERT INTO chat_attachment_locations VALUES(?,?,?,?,?)
                ON CONFLICT(attachment_id) DO UPDATE SET signature=excluded.signature,path=excluded.path,
                sha256=excluded.sha256,origin=excluded.origin""",
                (att["id"], self.signature(att), str(path), sha, origin))
        return {"path": str(path), "origin": origin, "attachment": att["id"]}

    def bind(self, attachment_id, path):
        result = self.remember(self.attachment(attachment_id), path, "manual")
        target = self.db.get_file_by_path(result["path"])
        with self.db.conn:
            self.db.conn.execute("UPDATE chat_attachments SET file_id=?,link_state='manual',linked_hash=? WHERE id=?",
                                 (target["id"] if target else None, target["file_hash"] if target else None, attachment_id))
        return result

    def confirmed_location(self, attachment_id):
        """Read only an established association, without lookup or API access."""
        att = self.attachment(attachment_id)
        self.check_cancel()
        last_error = None
        cached = self.db.conn.execute("SELECT * FROM chat_attachment_locations WHERE attachment_id=?", (attachment_id,)).fetchone()
        if cached and cached["signature"] == self.signature(att):
            try:
                return self.remember(att, cached["path"], cached["origin"], expected_sha=cached["sha256"])
            except WeFlowError as exc:
                if isinstance(exc, SyncCancelled):
                    raise
                last_error = exc
        if att["indexed_path"] and not att["indexed_path"].startswith(CHAT_PREFIX):
            try:
                return self.remember(att, att["indexed_path"], "manual" if att["link_state"] == "manual" else "indexed")
            except WeFlowError as exc:
                if isinstance(exc, SyncCancelled):
                    raise
                last_error = exc
        raise last_error or WeFlowError("附件尚未确认本地路径，请先打开定位或关联本地文件", code="file_unlocated")

    def locate(self, attachment_id):
        att = self.attachment(attachment_id)
        try:
            return self.confirmed_location(attachment_id)
        except WeFlowError as exc:
            if isinstance(exc, SyncCancelled):
                raise
        candidates = self.candidates(att)
        if candidates:
            if att["kind"] == "explicit" and (att["md5"] or att["sha256"]):
                verified = []
                for path in candidates:
                    sha, md5, size = file_digest(Path(path), self.cancel)
                    if (not att["md5"] or md5 == att["md5"].lower()) and (not att["sha256"] or sha == att["sha256"].lower()):
                        verified.append(path)
                if len(verified) == 1:
                    return self.remember(att, verified[0], "hash")
                candidates = verified
            if candidates:
                return {"candidates": candidates, "attachment": attachment_id, "name": att["name"]}
        if att["kind"] != "explicit":
            raise WeFlowError("此结果只是正文提及的文件名，请关联实际文件", code="file_mention")
        if att["source_kind"] != "weflow_api":
            raise WeFlowError("离线导入没有可回查的接口，请关联已保存文件", code="file_offline")
        if self.client is None and self.client_factory is not None:
            self.client = self.client_factory()
        if not self.client or not self.client._token:
            raise WeFlowError("请在微信索引页填写 Alpha Token，或设置微信文件目录 / 关联已保存文件", code="file_token_required")
        remote = self.remote_identity(att)
        self.progress("正在通过 Alpha 精确定位附件")
        resolved = self.resolve_remote(remote)
        # Forwarded cards can contain many files. Never open the outer card's arbitrary first resource.
        if att["origin"]:
            raise WeFlowError("合并转发内附件尚不能从此接口唯一定位，请设置微信文件目录或关联实际文件", code="file_forward_ambiguous")
        display = resolved.get("display", {})
        if isinstance(display, dict) and display.get("fileName") and display["fileName"] != att["name"]:
            raise WeFlowError("接口返回的文件名与索引不一致，请更新该会话", code="file_mismatch")
        refs = resolved.get("refs", [])
        if not isinstance(refs, list) or len(refs) > 200:
            raise WeFlowError("附件引用结构不兼容", code="file_refs_invalid")
        refs = [r for r in refs if isinstance(r, dict) and r.get("kind") == "file"]
        if len(refs) > 1:
            raise WeFlowError("同一消息有多个文件引用，无法唯一定位，请关联实际文件", code="file_refs_ambiguous")
        paths = [resolved[k] for k in ("local_path", "mediaLocalPath") if resolved.get(k)]
        if len(refs) == 1:
            paths += [refs[0][k] for k in ("local_path", "mediaLocalPath") if refs[0].get(k)]
        if paths and len(set(paths)) == 1:
            path = local_file(paths[0])
            if path.suffix.lower() != Path(att["name"]).suffix.lower():
                raise WeFlowError("接口返回的附件扩展名不匹配", code="file_mismatch")
            return self.remember(att, path, "alpha_path")
        if len(refs) != 1:
            raise WeFlowError("Alpha 未返回唯一可用的文件引用，请在微信下载附件或关联已保存文件", code="file_unavailable")
        path = self.download(att, refs[0])
        return self.remember(att, path, "alpha_cache")

    def candidates(self, att):
        paths = {r[0] for r in self.db.conn.execute("SELECT path FROM files WHERE name=? COLLATE NOCASE LIMIT 21", (att["name"],))
                 if not r[0].startswith(CHAT_PREFIX)}
        if self.root:
            root = Path(self.root).resolve()
            if not root.is_dir() or str(root).startswith(("\\\\", "//")) or root.parent == root:
                raise WeFlowError("请选择具体的本地微信文件目录，不能选择整个磁盘或网络目录", code="file_root_invalid")
            self.progress("正在微信文件目录按名称定位")
            started, count = time.monotonic(), 0
            pending = [root]
            while pending:
                self.check_cancel()
                folder = pending.pop()
                try:
                    with os.scandir(folder) as entries:
                        for entry in entries:
                            count += 1
                            if count > 100_000 or time.monotonic() - started > 15:
                                raise WeFlowError("目录较大，请选择更具体的 msg/file 或 FileStorage/File 子目录", code="file_lookup_limit")
                            self.check_cancel()
                            # Do not follow junctions/symlinks out of the user's selected root.
                            info = entry.stat(follow_symlinks=False)
                            if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                                continue
                            if stat.S_ISDIR(info.st_mode):
                                pending.append(Path(entry.path))
                            elif entry.name.casefold() == att["name"].casefold() and stat.S_ISREG(info.st_mode):
                                if att["size"] is None or info.st_size == att["size"]:
                                    paths.add(str(Path(entry.path).resolve()))
                            if len(paths) > 20:
                                raise WeFlowError("同名候选过多，请缩小微信文件目录", code="file_candidates_limit")
                except (PermissionError, FileNotFoundError):
                    continue
        valid = []
        for path in sorted(paths):
            try:
                actual = local_file(path)
                if att["size"] is None or actual.stat().st_size == att["size"]:
                    valid.append(str(actual))
            except (OSError, WeFlowError):
                continue
        return valid

    def remote_identity(self, att):
        client = self.client
        client.accounts()
        if client.protocol != "alpha":
            raise WeFlowError("自动附件定位需要 Alpha 的精确消息接口", code="file_api_unavailable")
        row = self.db.conn.execute("SELECT * FROM chat_api_messages WHERE message_id=?", (att["message_id"],)).fetchone()
        if row:
            if row["base_url"] != client.base_url or row["account_id"] != client.account_id:
                raise WeFlowError("附件所属账号或 API 地址与当前连接不同，请切换回原账号", code="account_mismatch")
            return dict(row)
        self.progress("正在为旧索引恢复原消息位置")
        sessions = client.sessions(should_cancel=self.cancel)
        matches = [s["id"] for s in sessions if "weflow:" + digest("\0".join([client.base_url, att["account"], s["id"]])) == att["source_key"]]
        if len(matches) != 1 or not att["sent_at"]:
            raise WeFlowError("未找到原会话，请连接建立此索引的 Alpha 账号", code="file_source_missing")
        session = matches[0]
        if client.identity(session, int(time.time()))["owner"] != att["account"]:
            raise WeFlowError("原会话账号与当前账号不匹配", code="account_mismatch")
        stamp = int(datetime.fromisoformat(att["sent_at"].replace("Z", "+00:00")).timestamp())
        payload = client.get("/api/v1/messages", talker=session, start=stamp, end=stamp, limit=200, media="0")
        client._check_response_account(payload)
        if payload.get("talker") != session or not isinstance(payload.get("messages"), list) or len(payload["messages"]) > 200:
            raise WeFlowError("原消息定位响应不匹配", code="file_identity")
        matches = []
        for raw in payload["messages"]:
            identity = api_identity(raw)
            if identity and identity[0] == att["external_id"] and identity[1][0] == session:
                matches.append(identity)
        if len(matches) != 1:
            raise WeFlowError("原消息未唯一定位，请更新该会话索引后重试", code="file_identity")
        self.check_cancel()
        with self.db.conn:
            save_api_identities(self.db, att["source_key"], client.base_url, client.account_id, dict(matches))
        return dict(self.db.conn.execute("SELECT * FROM chat_api_messages WHERE message_id=?", (att["message_id"],)).fetchone())

    def rpc(self, method, params):
        if method not in {"initialize", "tools/list", "tools/call"} or (method == "tools/call" and params.get("name") != RESOLVE_TOOL):
            raise WeFlowError("附件接口不在只读白名单", code="file_tool_denied")
        self.check_cancel()
        self.sequence += 1
        request = urllib.request.Request(self.client.base_url + "/mcp", method="POST",
            data=json.dumps({"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "Authorization": "Bearer " + self.client._token})
        value = self.client._request_json(request, timeout=min(self.client.timeout, 45))
        self.check_cancel()
        if value.get("jsonrpc") != "2.0" or type(value.get("id")) is not int or value["id"] != self.sequence or not isinstance(value.get("result"), dict):
            raise WeFlowError("附件接口响应标识不匹配", code="file_identity")
        result = value["result"]
        if result.get("isError"):
            raise WeFlowError("Alpha 无法解析此附件，请确认已在微信下载或另存文件", code="file_unavailable")
        return result

    def resolve_remote(self, remote):
        self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "LocalMind", "version": "0.5.0"}})
        cursor, found, seen = None, None, set()
        for _ in range(10):
            page = self.rpc("tools/list", {"cursor": cursor} if cursor else {})
            tools = page.get("tools", [])
            if not isinstance(tools, list) or len(tools) > 1000:
                break
            for tool in tools:
                if isinstance(tool, dict) and tool.get("name") == RESOLVE_TOOL:
                    found = tool
            cursor = page.get("nextCursor")
            if not cursor or not isinstance(cursor, str) or cursor in seen:
                break
            seen.add(cursor)
        args = {"account_id": remote["account_id"], "conversation_id": remote["conversation_id"],
                "message_id": remote["api_id"], "sent_at_unix_ms": remote["timestamp_ms"], "source_type": remote["source_type"]}
        schema = found.get("inputSchema", {}) if found else {}
        properties = schema.get("properties", {})
        if (schema.get("type") != "object" or not isinstance(properties, dict)
                or not set(args) <= set(properties) or not set(schema.get("required", [])) <= set(args)
                or any(properties[k].get("type") != "string" for k in args)):
            raise WeFlowError("当前 Alpha 未提供兼容的精确附件接口", code="file_api_unavailable")
        value = self.rpc("tools/call", {"name": RESOLVE_TOOL, "arguments": args}).get("structuredContent")
        if (not isinstance(value, dict) or value.get("id") != remote["api_id"]
                or value.get("conversation_id") != remote["conversation_id"]
                or str(value.get("sent_at_unix_ms")) != remote["timestamp_ms"]
                or str(value.get("source_type")) != remote["source_type"]
                or value.get("account_id", remote["account_id"]) != remote["account_id"]):
            raise WeFlowError("附件消息标识不匹配，未打开文件", code="file_identity")
        return value

    def download(self, att, ref):
        uri = urllib.parse.urlsplit(ref.get("uri", ""))
        if uri.scheme == "alpha-media" and uri.netloc == "sha256" and re.fullmatch(r"/[0-9a-f]{64}", uri.path):
            route = "/api/v1/media/" + uri.path[1:]
        elif uri.scheme in {"", "http"}:
            absolute = urllib.parse.urlsplit(urllib.parse.urljoin(self.client.base_url, ref.get("uri", "")))
            base = urllib.parse.urlsplit(self.client.base_url)
            if (absolute.scheme, absolute.netloc) != (base.scheme, base.netloc) or absolute.query or absolute.fragment:
                raise WeFlowError("已阻止非本机或携带查询参数的附件地址", code="file_url_invalid")
            route = absolute.path
        else:
            raise WeFlowError("附件引用协议不兼容", code="file_url_invalid")
        if not re.fullmatch(r"/api/v1/media/[A-Za-z0-9_-]{1,256}", route):
            raise WeFlowError("附件地址不在本机媒体白名单", code="file_url_invalid")
        name = att["name"]
        if (name != PureWindowsPath(name).name or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
                or name.rstrip(" .") != name or len(name) > 200
                or re.fullmatch(r"(?i)(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name)):
            raise WeFlowError("附件文件名不适合直接保存，请手动关联", code="file_name_invalid")
        root = (self.cache_dir / "wechat-files").resolve()
        target = root / digest(att["source_key"])[:16] / str(att["id"])
        target.mkdir(parents=True, exist_ok=True)
        if not target.resolve().is_relative_to(root):
            raise WeFlowError("附件缓存目录指向外部位置", code="file_path_invalid")
        temporary = None
        request = urllib.request.Request(self.client.base_url + route, headers={"Authorization": "Bearer " + self.client._token})
        try:
            with self.client.opener.open(request, timeout=min(self.client.timeout, 45)) as response:
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdigit() or int(length) > MAX_FILE_SIZE_BYTES):
                    raise WeFlowError("附件超过 50 MB 缓存上限", code="file_limit")
                with tempfile.NamedTemporaryFile(dir=target, suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    total = 0
                    while block := response.read(1024 * 1024):
                        self.check_cancel()
                        total += len(block)
                        if total > MAX_FILE_SIZE_BYTES:
                            raise WeFlowError("附件超过 50 MB 缓存上限", code="file_limit")
                        stream.write(block)
                if length is not None and total != int(length):
                    raise WeFlowError("附件传输不完整，请重试", code="file_incomplete")
            sha, md5, size = file_digest(temporary, self.cancel)
            if (att["size"] is not None and att["size"] != size) or (att["md5"] and att["md5"] != md5) or (att["sha256"] and att["sha256"] != sha):
                raise WeFlowError("附件内容校验失败，未保留缓存", code="file_mismatch")
            destination = target / sha
            name_budget = 240 - len(str(destination)) - 1
            if name_budget < 24:
                raise WeFlowError("缓存路径太长，请在隐私设置中选择更短的缓存目录", code="file_path_length")
            if len(name) > name_budget:
                suffix = Path(name).suffix
                name = name[:max(1, name_budget - len(suffix) - 13)] + "-" + digest(name)[:12] + suffix
            destination.mkdir(exist_ok=True)
            destination = destination / name
            if not destination.resolve().is_relative_to(root):
                raise WeFlowError("附件缓存目标超出目录", code="file_path_invalid")
            os.replace(temporary, destination)
            temporary = None
            return destination
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise WeFlowError("Alpha 尚未保存此附件（media.not_found）。可在 Alpha 另存附件后重试，或设置微信文件目录直接定位微信已有文件。", code="file_not_materialized") from None
            raise WeFlowError(f"Alpha 附件读取失败（HTTP {exc.code}），请检查 Token 或重试", code="file_download_failed") from None
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
            raise WeFlowError("附件读取中断，请重试", code="file_download_failed") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
