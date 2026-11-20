from __future__ import annotations

import io
import json
from threading import Event
from urllib.parse import urlsplit

import pytest

from localmind.weflow import SyncCancelled, WeFlowClient, WeFlowError
from localmind.weflow_metadata import AlphaMetadata, MetadataCache, TOOLS
from test_weflow_batch import BatchTransport


class MetadataTransport(BatchTransport):
    """Synthetic contract, deliberately not presented as a real account capture."""
    def __init__(self):
        super().__init__(4)
        self.rpc_calls = []
        self.profiles = {"chat-0": {"username": "chat-0", "labels": ["工作", {"name": "研发"}]},
                         "chat-1": {"username": "chat-1", "labels": []}}
        self.groups = {"group-a": {"chatroomId": "group-a", "members": [{"username": "person-a"}, {"username": "person-b"}]},
                       "group-b": {"chatroomId": "group-b", "members": [{"username": "person-b"}]}}
        self.candidates = [{"username": "person-a", "name": "王同事"}, {"username": "person-b", "name": "王同事"}]
        self.tools = [{"name": name, "inputSchema": {"type": "object", "properties": {key: {"type": "string"} for key in ("account", field)}, "required": [field]},
                       "outputSchema": {"type": "object"}}
                      for name, field in ((TOOLS["resolve"], "query"), (TOOLS["labels"], "username"), (TOOLS["members"], "chatroomId"))]
        self.tools[2]["inputSchema"]["properties"]["includeMessageCounts"] = {"type": "boolean"}
        self.tool_errors = set()
        self.error_code = None
        self.account = "test-account"
        self.response_id = None
        self.on_request = lambda _request: None

    def open(self, request, timeout):
        if urlsplit(request.full_url).path != "/mcp":
            return super().open(request, timeout)
        assert request.method == "POST" and request.get_header("Authorization") == "Bearer synthetic-token"
        assert not urlsplit(request.full_url).query and timeout <= 45
        call = json.loads(request.data)
        self.rpc_calls.append(call)
        self.on_request(call)
        if call["method"] == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "mock", "version": "0"}}
        elif call["method"] == "tools/list":
            result = {"tools": self.tools}
        else:
            name, arguments = call["params"]["name"], call["params"]["arguments"]
            assert name in TOOLS.values() and arguments["account"] == "test-account"
            if name in self.tool_errors:
                text = json.dumps({"code": self.error_code, "message": "private-token-and-personal-data"}) if self.error_code else "private-token-and-personal-data"
                result = {"isError": True, "content": [{"type": "text", "text": text}]}
            else:
                if name == TOOLS["resolve"]:
                    value = {"candidates": self.candidates}
                elif name == TOOLS["labels"]:
                    value = {"profile": self.profiles.get(arguments["username"], {"username": arguments["username"]})}
                else:
                    assert arguments["includeMessageCounts"] is False
                    value = self.groups[arguments["chatroomId"]]
                result = {"structuredContent": {"account_id": self.account, **value}}
        return io.BytesIO(json.dumps({"jsonrpc": "2.0", "id": self.response_id or call["id"], "result": result}).encode())


def service(cache=None, **kwargs):
    client = WeFlowClient(token="synthetic-token", account="test-account")
    client.opener = MetadataTransport()
    return AlphaMetadata(client, cache, interval=0, **kwargs)


def test_discovery_readonly_pin_and_no_chat_calls():
    meta = service()
    assert [row["id"] for row in meta.resolve("王同事")] == ["person-a", "person-b"]
    labels = meta.collect("labels", ["chat-0", "chat-1", "chat-2"])
    assert labels["values"] == {"chat-0": {"工作", "研发"}, "chat-1": set()}
    assert set(labels["unknown"]) == {"chat-2"} and labels["state"] == "partial"
    groups = meta.groups(["group-a", "group-b"], "person-a")
    assert groups["matches"] == {"group-a"} and groups["nonmatches"] == {"group-b"}
    assert not groups["unknown"]
    assert all(path == "/api/v1/accounts" for path, _ in meta.client.opener.calls)
    with pytest.raises(WeFlowError, match="白名单"):
        meta.rpc("tools/call", {"name": "wechat.chat.send_text", "arguments": {}})
    with pytest.raises(WeFlowError):
        meta.rpc("delete", {})


@pytest.mark.parametrize("change", ["missing", "no_account", "new_required", "wrong_type"])
def test_missing_or_changed_input_capability_never_calls_tool(change):
    meta = service()
    tools = meta.client.opener.tools
    if change == "missing":
        tools.pop(1)
    elif change == "no_account":
        del tools[1]["inputSchema"]["properties"]["account"]
    elif change == "new_required":
        tools[1]["inputSchema"]["required"].append("unrecognized")
    else:
        tools[1]["inputSchema"]["properties"]["username"]["type"] = "integer"
    result = meta.collect("labels", ["chat-0", "chat-1"])
    assert result["state"] == "unavailable" and len(result["unknown"]) == 2
    assert not any(call["method"] == "tools/call" for call in meta.client.opener.rpc_calls)


@pytest.mark.parametrize("change", ["account", "identity", "truncated", "total", "duplicate", "missing_members"])
def test_incomplete_or_wrong_group_is_unknown_not_absent(change):
    meta = service()
    row = meta.client.opener.groups["group-a"]
    if change == "account":
        meta.client.opener.account = "another"
    elif change == "identity":
        row["chatroomId"] = "other-group"
    elif change == "truncated":
        row["hasMore"] = True
    elif change == "total":
        row["total"] = 99
    elif change == "duplicate":
        row["members"].append(row["members"][0])
    else:
        del row["members"]
    result = meta.groups(["group-a", "group-b"], "person-a")
    assert "group-a" in result["unknown"] and "group-a" not in result["nonmatches"]
    assert not any(key[-1] == "group-a" for key in meta.cache.entries)


def test_missing_labels_are_not_empty_and_refresh_removes_stale_results():
    meta = service()
    assert meta.collect("labels", ["chat-0"])["values"]["chat-0"]
    meta.client.opener.profiles["chat-0"] = {"username": "chat-0"}
    assert meta.collect("labels", ["chat-0"])["cached"] == 1
    refreshed = meta.collect("labels", ["chat-0"], force=True)
    assert "chat-0" in refreshed["unknown"] and not refreshed["values"]
    assert "chat-0" in meta.collect("labels", ["chat-0"])["unknown"]


def test_cache_ttl_lru_and_account_isolation():
    clock = [0]
    cache = MetadataCache(clock=lambda: clock[0], max_entries=2, max_items=3)
    meta = service(cache)
    meta.collect("labels", ["chat-0", "chat-1"])
    before = len(meta.client.opener.rpc_calls)
    assert meta.collect("labels", ["chat-0", "chat-1"])["cached"] == 2
    assert len(meta.client.opener.rpc_calls) == before
    assert cache.get((meta.client.base_url, "different-account", "labels", "chat-0")) is None
    cache.put(("new",), {"value", "second"})
    assert cache.items <= 3 and len(cache.entries) <= 2
    clock[0] = 601
    assert cache.get(("new",)) is None


def test_cancel_after_request_never_accepts_response_and_preserves_unread_states():
    cancel = Event()
    meta = service(should_cancel=cancel.is_set)
    meta.discover()
    meta.client.opener.on_request = lambda _request: cancel.set()
    result = meta.groups(["group-a", "group-b"], "person-a")
    assert result["state"] == "cancelled" and len(result["unknown"]) == 2
    assert not result["matches"] and not result["nonmatches"] and not meta.cache.entries


def test_tool_errors_are_sanitized_and_responses_bound_to_request():
    meta = service()
    meta.client.opener.tool_errors.add(TOOLS["labels"])
    result = meta.collect("labels", ["chat-0"])
    assert "private-token" not in str(result) and "chat-0" in result["unknown"]
    meta.client.opener.response_id = 999
    with pytest.raises(WeFlowError, match="标识不匹配"):
        meta.resolve("name")


def test_installed_alpha_denied_profile_stops_bulk_without_claiming_empty_labels():
    meta = service()
    meta.client.opener.tool_errors.add(TOOLS["labels"])
    meta.client.opener.error_code = "engine.desktop_operation_not_allowed"
    result = meta.collect("labels", ["chat-0", "chat-1", "chat-2"])
    assert result["state"] == "unavailable" and len(result["unknown"]) == 3 and not result["values"]
    assert len([call for call in meta.client.opener.rpc_calls if call["method"] == "tools/call"]) == 1
    assert "private-token" not in str(result) and "标签能力暂不可用" in result["unknown"]["chat-0"]
    meta.client.opener.tool_errors.add(TOOLS["members"])
    groups = meta.groups(["group-a"], "person-a")
    assert groups["state"] == "unavailable" and "标签" not in groups["unknown"]["group-a"]


def test_installed_alpha_group_envelope_no_group_echo_and_contact_display_name():
    meta = service()
    meta.client.opener.groups["group-a"] = {"count": 2, "items": [{"username": "person-a", "nickname": "Name A"}, {"username": "person-b"}], "hasMore": False, "nextCursor": None}
    result = meta.groups(["group-a"], "person-a")
    assert result["matches"] == {"group-a"} and not result["unknown"]
    meta.client.opener.candidates = [{"username": "person-a", "display_name": "Display A", "nickname": "Name A"}]
    assert meta.resolve("Display")[0]["name"] == "Display A"


def test_member_pagination_joins_before_deciding_absence():
    meta = service()
    props = meta.client.opener.tools[2]["inputSchema"]["properties"]
    props["limit"] = {"type": "integer", "minimum": 1, "maximum": 200}
    props["offset"] = {"type": "integer", "minimum": 0}
    seen_offsets = []

    def page(call):
        if call["method"] != "tools/call" or call["params"]["name"] != TOOLS["members"]:
            return
        arguments = call["params"]["arguments"]
        offset = arguments["offset"]
        seen_offsets.append(offset)
        count = min(200, 401 - offset)
        meta.client.opener.groups["group-a"] = {"items": [{"username": f"member-{i}"} for i in range(offset, offset + count)],
                                                  "count": count, "total": 401, "hasMore": offset + count < 401, "nextCursor": None}

    meta.client.opener.on_request = page
    result = meta.groups(["group-a"], "member-400")
    assert result["matches"] == {"group-a"} and not result["unknown"] and seen_offsets == [0, 200, 400]
    assert meta.groups(["group-a"], "missing")["nonmatches"] == {"group-a"}
    assert seen_offsets == [0, 200, 400]
