"""Synthetic contracts based on the installed Alpha tools/list from 2026-10-29."""

import copy

import pytest

from localmind.alpha_chat_context import AlphaChatContext
from localmind.weflow import WeFlowError
from test_alpha_chat_context import ANCHOR, Client, MID, STAMP, mcp_message


def modern_client(*, canonical=True):
    client = Client()
    client.tools = [
        client.tool("get_messages", {"after_unix_ms": "string", "before_unix_ms": "string", "defer_media": "boolean", "limit": "integer"}),
        client.tool("get_message_around", {"anchor_id": "string", "sent_at_unix_ms": "string", "server_id": "string", "before": "integer", "after": "integer"}),
    ]
    client.tools[0]["inputSchema"]["required"] = ["username"]
    client.tools[1]["inputSchema"]["required"] = ["username", "anchor_id"]
    row = {"id": MID, "text": "合成上下文", "sender_name": "测试成员", "sender_id": "person",
           "sent_at_unix_ms": str(STAMP), "conversation_id": "chat", "account_id": "acct"} if canonical else mcp_message()
    client.bridge = {"items": [row]}
    client.around = {"items": [copy.deepcopy(row)]}
    return client


@pytest.mark.parametrize("canonical", [True, False])
def test_observed_schema_uses_exclusive_milliseconds_and_fast_anchor_seek(canonical):
    client = modern_client(canonical=canonical)
    reader = AlphaChatContext(client)
    value = reader.fetch(ANCHOR)
    assert value.source == "mcp_around" and len(value.rows) == 1
    assert value.rows[0]["sender"] == "测试成员"
    assert not any("http" in call for call in client.calls)
    calls = [call["params"]["arguments"] for call in client.calls if call.get("method") == "tools/call"]
    assert calls[0] == {"account": "acct", "username": "chat", "limit": 20,
                        "after_unix_ms": str(STAMP - 1), "before_unix_ms": str(STAMP + 1), "defer_media": True}
    assert calls[1] == {"account": "acct", "username": "chat", "anchor_id": MID,
                        "sent_at_unix_ms": str(STAMP), "server_id": "123456", "before": 20, "after": 20}
    assert reader.diagnostics["bridge_time_format"] == "unix_ms_exclusive"
    assert reader.diagnostics["around_seek_hint"] == "server_id"
    assert reader.diagnostics["forward_tool_advertised"] is False
    assert all(secret not in str(reader.diagnostics) for secret in (MID, "acct", "合成上下文", "synthetic-token"))


@pytest.mark.parametrize("phase", ["bridge", "around"])
def test_exact_timestamp_changes_fail_closed_even_within_the_same_second(phase):
    client = modern_client()
    getattr(client, phase)["items"][0]["sent_at_unix_ms"] = str(STAMP + 1)
    with pytest.raises(WeFlowError) as error:
        AlphaChatContext(client).fetch(ANCHOR)
    assert error.value.code == "context_identity"
    assert not any("http" in call for call in client.calls)


def test_conflicting_canonical_and_legacy_times_are_not_silently_preferred():
    client = modern_client()
    client.around["items"][0]["time"] = "2026-10-20T08:00:01+08:00"
    with pytest.raises(WeFlowError, match="互相矛盾"):
        AlphaChatContext(client).fetch(ANCHOR)


@pytest.mark.parametrize("value", [True, 1792454400123.0, "invalid", "9999999999999999999999"])
def test_canonical_time_rejects_noninteger_or_out_of_range_values(value):
    client = modern_client()
    client.bridge["items"][0]["sent_at_unix_ms"] = value
    with pytest.raises(WeFlowError):
        AlphaChatContext(client).fetch(ANCHOR)


def test_missing_optional_seek_fields_still_supports_legacy_contract():
    client = modern_client()
    del client.tools[1]["inputSchema"]["properties"]["server_id"]
    assert AlphaChatContext(client).fetch(ANCHOR).source == "mcp_around"
    arguments = client.calls[-1]["params"]["arguments"]
    assert arguments["sent_at_unix_ms"] == str(STAMP) and "server_id" not in arguments


def test_opaque_external_id_is_never_sent_as_server_id():
    client = modern_client()
    assert AlphaChatContext(client).fetch({**ANCHOR, "external_id": "alpha:" + MID}).source == "mcp_around"
    assert "server_id" not in client.calls[-1]["params"]["arguments"]


def test_incomplete_millisecond_schema_uses_bounded_http_not_unbounded_mcp():
    client = modern_client()
    del client.tools[0]["inputSchema"]["properties"]["before_unix_ms"]
    value = AlphaChatContext(client).fetch(ANCHOR)
    assert value.source == "http_window"
    assert not any(call.get("method") == "tools/call" for call in client.calls)
    assert client.calls[-1]["params"]["limit"] == 200


def test_wrong_namespace_fallback_preserves_remaining_read_budget():
    client = modern_client()
    client.bridge["items"][0]["id"] = "another-namespace"
    reader = AlphaChatContext(client)
    assert reader.fetch(ANCHOR).source == "http_window"
    assert client.calls[-1]["params"]["limit"] == 199
    assert reader.diagnostics["mcp_fallback_reason"] == "anchor_namespace_unconfirmed"


def test_duplicate_bridge_id_is_not_treated_as_missing_namespace():
    client = modern_client()
    client.bridge["items"] *= 2
    with pytest.raises(WeFlowError, match="身份重复"):
        AlphaChatContext(client).fetch(ANCHOR)


@pytest.mark.parametrize("use_http", [True, False])
def test_exclusive_date_boundaries_retain_millisecond_precision(use_http):
    client = modern_client()
    extra = {**client.around["items"][0], "id": "msg-" + "2" * 64, "sent_at_unix_ms": str(STAMP + 100)}
    if use_http:
        client.tools = []
        client.http["messages"].append({**client.http["messages"][0], **extra, "platform_message_id": "999"})
    else:
        client.around["items"].append(extra)
    value = AlphaChatContext(client).fetch(ANCHOR, end=(STAMP + 100) / 1000)
    assert [row["id"] for row in value.rows] == [MID]
    assert value.rows[0]["timestamp_ms"] == str(STAMP)


def test_missing_forward_tool_and_absent_payload_have_specific_diagnostics():
    reader = AlphaChatContext(modern_client())
    with pytest.raises(WeFlowError) as error:
        reader.fetch(ANCHOR, kind="forward")
    assert error.value.code == "context_forward_unavailable"
    assert "未提供转发展开工具" in str(error.value) and "无法还原内部聊天" in str(error.value)
    assert reader.diagnostics["forward_tool_advertised"] is False
    assert reader.diagnostics["http_forward_items"] == 0
    assert reader.diagnostics["raw_tool_advertised"] is False
    assert "合成上下文" not in str(reader.diagnostics)


def test_empty_forward_response_is_not_a_false_success():
    client = Client()
    client.forward = {"messages": []}
    with pytest.raises(WeFlowError) as error:
        AlphaChatContext(client).fetch(ANCHOR, kind="forward")
    assert error.value.code == "context_forward_unavailable"
    assert any("http" in call for call in client.calls)


def test_boolean_parameters_do_not_accept_integer_substitutes():
    reader = AlphaChatContext(modern_client())
    reader.discover()
    with pytest.raises(WeFlowError, match="类型"):
        reader.call("wechat.chat.get_messages", username="chat", limit=20,
                    after_unix_ms=str(STAMP - 1), before_unix_ms=str(STAMP + 1), defer_media=1)


def test_raw_tool_discovery_does_not_grant_unimplemented_execution():
    client = modern_client()
    client.tools.append(client.tool("get_message_raw", {"message_id": "string"}))
    reader = AlphaChatContext(client)
    reader.discover()
    assert "wechat.chat.get_message_raw" in reader.tools
    assert "wechat.chat.get_message_raw" in reader.catalog_names
    with pytest.raises(WeFlowError, match="白名单"):
        reader.call("wechat.chat.get_message_raw", username="chat", message_id=MID)
