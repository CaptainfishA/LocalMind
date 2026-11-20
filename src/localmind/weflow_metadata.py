"""Bounded, account-pinned contact metadata; no chat or mutation tools are exposed."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import OrderedDict

from .weflow import SyncCancelled, WeFlowError

TOOLS = {
    "resolve": "wechat.contacts.resolve_contact",
    "labels": "wechat.contacts.get_contact_profile",
    "members": "wechat.contacts.get_group_members",
}
MAX_METADATA_ITEMS = 20_000


def invalid(message="联系人元数据结构不兼容"):
    return WeFlowError(message, code="metadata_shape")


def identity(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise invalid("联系人缺少有效的稳定标识")
    return value


def record_id(row):
    if not isinstance(row, dict):
        raise invalid()
    values = [row[key] for key in ("username", "wxid", "id") if key in row]
    if not values or any(identity(value) != values[0] for value in values):
        raise invalid("联系人标识缺失或冲突")
    return values[0]


def rows_from(value, keys):
    candidates = [value[key] for key in keys if key in value]
    if len(candidates) != 1 or not isinstance(candidates[0], list) or len(candidates[0]) > MAX_METADATA_ITEMS:
        raise invalid()
    return candidates[0]


def tool_failure(result, operation):
    candidates = [result.get("structuredContent")]
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            try:
                candidates.append(json.loads(item["text"]))
            except ValueError:
                pass
    for value in candidates:
        if not isinstance(value, dict):
            continue
        error = value.get("error", value)
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str) and "_" in code and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,99}", code):
            if code == "engine.desktop_operation_not_allowed":
                message = ("当前 Alpha 未允许联系人详情操作；标签能力暂不可用，可继续按名称选择" if operation == "labels"
                           else "当前 Alpha 未允许此联系人或群成员操作，请检查公开接口权限")
                return WeFlowError(message, code=code)
            return WeFlowError(f"联系人查询失败（{code}）", code=code)
    return WeFlowError("联系人查询失败；请检查接口权限或重试", code="metadata_read_failed")


class MetadataCache:
    """Only normalized labels/member IDs, with an LRU and total-item memory bound."""
    def __init__(self, *, ttl=600, max_entries=3000, max_items=100_000, clock=time.monotonic):
        self.ttl, self.max_entries, self.max_items, self.clock = ttl, max_entries, max_items, clock
        self.entries = OrderedDict()
        self.items = 0

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return None
        if self.clock() - entry[0] >= self.ttl:
            self._remove(key)
            return None
        self.entries.move_to_end(key)
        return entry[1], entry[2]

    def _remove(self, key):
        self.items -= len(self.entries.pop(key)[1])

    def forget(self, key):
        if key in self.entries:
            self._remove(key)

    def put(self, key, values):
        if key in self.entries:
            self._remove(key)
        if len(values) > self.max_items:
            return
        while self.entries and (len(self.entries) >= self.max_entries or self.items + len(values) > self.max_items):
            self._remove(next(iter(self.entries)))
        stamp = time.time()
        self.entries[key] = (self.clock(), frozenset(values), stamp)
        self.items += len(values)


class AlphaMetadata:
    def __init__(self, client, cache=None, *, should_cancel=lambda: False, progress=lambda _: None, interval=0.1):
        self.client = client
        self.cache = cache if cache is not None else MetadataCache()
        self.cancel, self.progress = should_cancel, progress
        self.interval, self.next_request = interval, 0.0
        self.tools = None
        self.sequence = 0

    def check_cancel(self):
        if self.cancel():
            raise SyncCancelled("元数据读取已取消")

    def rpc(self, method, params):
        if method not in {"initialize", "tools/list", "tools/call"}:
            raise WeFlowError("仅允许联系人只读 MCP 协议", code="metadata_tool_denied")
        if method == "tools/call" and params.get("name") not in TOOLS.values():
            raise WeFlowError("此工具不在联系人只读白名单", code="metadata_tool_denied")
        self.check_cancel()
        while time.monotonic() < self.next_request:
            time.sleep(max(0, min(0.02, self.next_request - time.monotonic())))
            self.check_cancel()
        self.sequence += 1
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.client._token:
            headers["Authorization"] = "Bearer " + self.client._token
        body = json.dumps({"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}).encode()
        if len(body) > 1024 * 1024:
            raise invalid("元数据请求超过安全上限")
        request = urllib.request.Request(self.client.base_url + "/mcp", data=body, headers=headers, method="POST")
        try:
            value = self.client._request_json(request, timeout=min(self.client.timeout, 45))
        finally:
            self.next_request = time.monotonic() + self.interval
        self.check_cancel()
        if value.get("jsonrpc") != "2.0" or type(value.get("id")) is not int or value["id"] != self.sequence:
            raise invalid("MCP 响应标识不匹配")
        result = value.get("result")
        if not isinstance(result, dict):
            raise invalid("MCP 未返回结构化结果")
        return result

    def discover(self):
        if self.tools is not None:
            return
        self.check_cancel()
        self.client.accounts()
        if self.client.protocol != "alpha" or not self.client.account_id:
            raise WeFlowError("当前接口不能固定账号，标签与成员筛选不可用", code="metadata_unavailable")
        value = self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                        "clientInfo": {"name": "LocalMind", "version": "0.5.0"}})
        if not isinstance(value.get("capabilities"), dict) or "tools" not in value["capabilities"]:
            raise WeFlowError("当前接口未提供 MCP 工具能力", code="metadata_unavailable")
        found, cursor, cursors = {}, None, set()
        for _ in range(10):
            page = self.rpc("tools/list", {"cursor": cursor} if cursor else {})
            rows = page.get("tools")
            if not isinstance(rows, list) or len(rows) > 1000:
                raise invalid("MCP 工具目录不兼容")
            for row in rows:
                if isinstance(row, dict) and row.get("name") in TOOLS.values():
                    if row["name"] in found:
                        raise invalid("MCP 工具目录重复")
                    found[row["name"]] = row
            cursor = page.get("nextCursor")
            if cursor is None:
                self.tools = found
                return
            if not isinstance(cursor, str) or not cursor or len(cursor) > 2048 or cursor in cursors:
                raise invalid("MCP 工具目录分页未推进")
            cursors.add(cursor)
        raise invalid("MCP 工具目录超过安全上限")

    def call(self, operation, **arguments):
        self.discover()
        name = TOOLS[operation]
        tool = self.tools.get(name)
        if not tool:
            raise WeFlowError("安装版本未提供所需联系人能力", code="metadata_unavailable")
        schema = tool.get("inputSchema", {})
        if not isinstance(schema, dict):
            raise WeFlowError("联系人参数声明不兼容", code="metadata_unavailable")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        arguments = {"account": self.client.account_id, **arguments}
        if operation == "members" and isinstance(properties, dict) and isinstance(properties.get("includeMessageCounts"), dict) and properties["includeMessageCounts"].get("type") == "boolean":
            arguments["includeMessageCounts"] = False
        if (schema.get("type") != "object" or not isinstance(properties, dict) or not isinstance(required, list)
                or any(not isinstance(k, str) for k in required)
                or not set(arguments) <= set(properties) or not set(required) <= set(arguments)):
            raise WeFlowError("安装版本的联系人参数不兼容，未发送查询", code="metadata_unavailable")
        for key, value in arguments.items():
            spec = properties[key]
            if (not isinstance(spec, dict) or not isinstance(spec.get("type"), str) or spec["type"] not in {"string", "integer", "boolean"}
                    or type(value) is not {"string": str, "integer": int, "boolean": bool}[spec["type"]]):
                raise WeFlowError("联系人参数类型不兼容，未发送查询", code="metadata_unavailable")
            if isinstance(value, str) and not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", 10000):
                raise invalid("联系人查询参数长度超出接口范围")
            if type(value) is int and not spec.get("minimum", 0) <= value <= spec.get("maximum", 2**31 - 1):
                raise invalid("联系人分页参数超出接口范围")
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError") is True:
            # Tool error text may contain personal data; never forward it to the UI/log.
            raise tool_failure(result, operation)
        value = result.get("structuredContent")
        if not isinstance(value, dict):
            raise invalid("MCP 缺少结构化联系人数据")
        if "account_id" in value and value["account_id"] != self.client.account_id:
            raise WeFlowError("联系人数据账号不匹配，已停止", code="account_mismatch")
        if value.get("success") is False or value.get("error"):
            raise WeFlowError("联系人查询未成功", code="metadata_read_failed")
        return value

    def resolve(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 256:
            raise invalid("请输入不超过 256 字的联系人名称或标识")
        value = self.call("resolve", query=query.strip())
        rows = rows_from(value, ("contacts", "items", "candidates"))
        if len(rows) > 20:
            raise invalid("联系人候选超过接口上限，请缩小名称范围")
        result, seen = [], set()
        for row in rows:
            uid = record_id(row)
            if uid in seen:
                raise invalid("联系人候选包含重复标识")
            seen.add(uid)
            name = next((row[key] for key in ("remark", "display_name", "displayName", "name", "nickname") if row.get(key)), uid)
            if not isinstance(name, str) or len(name) > 512:
                raise invalid()
            result.append({"id": uid, "name": name})
        return result

    def _read_values(self, kind, uid, *, force=False):
        uid = identity(uid)
        self.discover()
        key = (self.client.base_url, self.client.account_id, kind, uid)
        if force:
            self.cache.forget(key)
        cached = None if force else self.cache.get(key)
        if cached is not None:
            return cached[0], cached[1], True
        if kind == "labels":
            value = self.call(kind, username=uid)
            row = value.get("profile", value.get("contact", value))
            if record_id(row) != uid:
                raise WeFlowError("联系人详情标识不匹配", code="metadata_identity")
            labels = row.get("labels")
            if not isinstance(labels, list) or len(labels) > 500:
                raise invalid("该联系人未返回可用标签信息")
            values = set()
            for label in labels:
                text = label if isinstance(label, str) else label.get("name") if isinstance(label, dict) else None
                if not isinstance(text, str) or not text.strip() or len(text) > 256:
                    raise invalid("联系人标签结构不兼容")
                values.add(text)
        else:
            values = self._members(uid)
        self.cache.put(key, values)
        return frozenset(values), time.time(), False

    def _members(self, uid):
        tool = self.tools.get(TOOLS["members"], {})
        props = tool.get("inputSchema", {}).get("properties", {})
        paged = all(isinstance(props.get(key), dict) and props[key].get("type") == "integer" for key in ("limit", "offset"))
        limit = min(200, props["limit"].get("maximum", 200)) if paged else None
        offset, seen, expected_total = 0, set(), None
        for _ in range(100):
            value = self.call("members", chatroomId=uid, **({"limit": limit, "offset": offset} if paged else {}))
            returned_id = value.get("chatroomId", value.get("username", value.get("conversation_id")))
            # Installed Alpha omits the group echo; account + JSON-RPC ID still bind the response.
            if (returned_id is not None and returned_id != uid) or (returned_id is None and value.get("account_id") != self.client.account_id):
                raise WeFlowError("群成员所属账号或群不匹配", code="metadata_identity")
            if value.get("truncated") is True:
                raise invalid("群成员列表被截断")
            rows = rows_from(value, ("members", "items"))
            if paged and len(rows) > limit:
                raise invalid("群成员页超过请求上限")
            current = {record_id(row) for row in rows}
            if len(current) != len(rows) or seen & current:
                raise invalid("群成员分页重复，不能确认完整列表")
            seen.update(current)
            if len(seen) > MAX_METADATA_ITEMS:
                raise invalid("群成员超过安全上限")
            if "count" in value and (type(value["count"]) is not int or value["count"] != len(rows)):
                raise invalid("群成员本页数量不一致")
            if "total" in value and value["total"] is not None:
                total = value["total"]
                if type(total) is not int or total < len(seen) or (expected_total is not None and expected_total != total):
                    raise invalid("群成员总量在读取期间改变")
                expected_total = total
            more = value.get("hasMore", False)
            if type(more) is not bool:
                raise invalid("群成员分页标记不兼容")
            if not more:
                if value.get("nextCursor") or (expected_total is not None and len(seen) != expected_total):
                    raise invalid("群成员列表不完整，不能判断未匹配成员")
                return seen
            if not paged or not rows:
                raise invalid("群成员列表不完整且无法继续分页")
            offset += len(rows)
        raise invalid("群成员分页超过安全上限")

    def collect(self, kind, ids, *, force=False):
        if kind not in {"labels", "members"} or not isinstance(ids, list) or len(ids) > MAX_METADATA_ITEMS:
            raise invalid("元数据范围无效")
        if len(set(identity(uid) for uid in ids)) != len(ids):
            raise invalid("元数据范围包含重复标识")
        values, unknown, timestamps, reused = {}, {}, {}, 0
        stopped = ""
        for index, uid in enumerate(ids):
            if stopped or self.cancel():
                stopped = stopped or "cancelled"
                unknown[uid] = "已取消" if stopped == "cancelled" else "未读取"
                continue
            try:
                data, stamp, cached = self._read_values(kind, uid, force=force)
                values[uid], timestamps[uid] = data, stamp
                reused += int(cached)
            except SyncCancelled:
                stopped, unknown[uid] = "cancelled", "已取消"
            except WeFlowError as exc:
                unknown[uid] = str(exc)
                if (exc.status in {401, 403, 404, 423, 429, 502, 503, 504}
                        or exc.code in {"transport_unavailable", "account_mismatch", "metadata_identity", "metadata_unavailable",
                                        "engine.desktop_operation_not_allowed"}):
                    stopped = "unavailable"
            self.progress(f"{'标签' if kind == 'labels' else '群成员'} {index + 1}/{len(ids)} · 未确认 {len(unknown)}")
        if self.cancel():
            stopped = "cancelled"
        return {"values": values, "unknown": unknown, "timestamps": timestamps, "cached": reused,
                "state": stopped or ("partial" if unknown else "complete")}

    def groups(self, ids, person, *, force=False):
        person = identity(person)
        result = self.collect("members", ids, force=force)
        values = result.pop("values")
        result["matches"] = {uid for uid, members in values.items() if person in members}
        result["nonmatches"] = set(values) - result["matches"]
        return result
