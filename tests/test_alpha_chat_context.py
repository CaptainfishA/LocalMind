import copy
import json
from dataclasses import replace

import pytest

from localmind.alpha_chat_context import AlphaChatContext, ContextCache, anchor_record, link_known_attachments
from localmind.chats import Attachment, ChatMessage, ChatStore, digest
from localmind.weflow import WeFlowError
from localmind.wechat_files import save_api_identities
from test_chat_timeline import corpus, result
from localmind.chat_timeline import discussions


MID = "msg-" + "1" * 64
STAMP = 1792454400123
ANCHOR = {"api_id": MID, "timestamp_ms": str(STAMP), "source_type": "1", "external_id": "123456",
          "base_url": "http://127.0.0.1:5031", "account_id": "acct", "conversation_id": "chat", "revision": 1}


def mcp_message(mid=MID, text="合成上下文", **kwargs):
    return {"id": mid, "time": "2026-10-20T08:00:00+08:00", "sender": "测试成员", "text": text, **kwargs}


def test_unsupported_remote_text_is_a_visible_gap_not_total_context_failure():
    client = Client()
    client.around["messages"][0]["text"] = {"unsupported": "shape"}
    value = AlphaChatContext(client).fetch(ANCHOR)
    assert value.skipped == 1 and len(value.rows) == 3 and not value.complete_range
    assert "不支持" in value.rows[0]["body"]


def test_remote_dates_keep_exclusive_upper_bound():
    client = Client()
    client.around["messages"][-1]["time"] = "2026-10-20T08:00:01+08:00"
    value = AlphaChatContext(client).fetch(ANCHOR, end=STAMP // 1000 + 1)
    assert len(value.rows) == 2


def test_remote_reader_can_be_reused_without_duplicate_tool_state():
    reader = AlphaChatContext(Client())
    first = reader.fetch(ANCHOR)
    assert reader.fetch(ANCHOR, force=True) == first


def test_forward_child_conversation_is_not_confused_with_outer_chat():
    client = Client()
    client.forward["messages"][0]["conversation_id"] = "original-forwarded-chat"
    value = AlphaChatContext(client).fetch(ANCHOR, kind="forward")
    assert value.rows[0]["sender"] == "原作者甲" and value.rows[0]["from_me"] is None


def test_forward_wrong_outer_server_id_is_rejected():
    client = Client()
    client.forward["server_id"] = "99999"
    with pytest.raises(WeFlowError, match="外层消息身份"):
        AlphaChatContext(client).fetch(ANCHOR, kind="forward")


@pytest.mark.parametrize("changed", [None, "body", "sender", "api_id", "time"])
def test_remote_attachment_requires_exact_local_message(corpus, changed):
    key = "weflow:" + digest("\0".join((ANCHOR["base_url"], "owner", "chat")))
    ChatStore(corpus).ingest(platform="wechat", account="owner", source_key=key, conversation="合成会话",
        messages=[ChatMessage(ANCHOR["external_id"], "合成上下文", sender="测试成员", sent_at="2026-10-20T08:00:00+08:00",
                              attachments=[Attachment("合成.pdf")])])
    hit = result(corpus, ANCHOR["external_id"], key)
    with corpus.conn:
        corpus.conn.execute("INSERT INTO chat_api_messages VALUES(?,?,?,?,?,?,?)",
            (hit.chat_message_id, ANCHOR["base_url"], "acct", "chat", MID, str(STAMP), "1"))
    group = discussions(corpus.db_path, hit)[0][0]
    anchor = anchor_record(corpus.db_path, hit, group, hit.chat_message_id)
    client = Client()
    if changed == "body":
        client.around["messages"][1]["text"] = "已修改正文"
    elif changed == "sender":
        client.around["messages"][1]["sender"] = "同名文件的另一个发送者"
    batch = AlphaChatContext(client).fetch(anchor)
    if changed in {"api_id", "time"}:
        key_name = "id" if changed == "api_id" else "sent_epoch"
        batch.rows[1][key_name] = "not-indexed" if changed == "api_id" else 0
    linked = link_known_attachments(corpus.db_path, group, anchor, batch)
    assert bool(linked.rows[1]["attachments"]) == (changed is None)
    assert not batch.rows[1]["attachments"]


class Client:
    base_url = ANCHOR["base_url"]
    account_id = "acct"
    protocol = "alpha"
    timeout = 15
    _token = "synthetic-token"

    def __init__(self):
        self.calls = []
        self.tools = [self.tool("get_messages", {"start_time": "integer", "end_time": "integer", "limit": "integer"}),
                      self.tool("get_message_around", {"anchor_id": "string", "before": "integer", "after": "integer"}),
                      self.tool("resolve_chat_history", {"server_id": "string"})]
        self.around = {"messages": [mcp_message("before"), mcp_message(), mcp_message("after", from_me=True)]}
        self.bridge = {"messages": [mcp_message()]}
        self.forward = {"messages": [mcp_message("a", sender="原作者甲"), mcp_message("b", sender="原作者乙")]}
        self.http = {"account_id": "acct", "talker": "chat", "hasMore": False, "scanComplete": True,
                     "messages": [{"id": MID, "conversation_id": "chat", "sent_at_unix_ms": str(STAMP),
                                   "platform_message_id": "123456", "source_type": 1, "sender_id": "person", "text": "合成上下文"}]}

    def tool(self, name, args):
        return {"name": "wechat.chat." + name, "inputSchema": {"type": "object",
            "properties": {k: {"type": t} for k, t in {"account": "string", "username": "string", **args}.items()}}}

    def accounts(self):
        return [{"id": "acct"}]

    def _request_json(self, req, timeout=None):
        request = json.loads(req.data)
        self.calls.append(request)
        method, params = request["method"], request["params"]
        if method == "initialize":
            result = {"capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": self.tools}
        else:
            assert params["arguments"]["account"] == "acct" and params["arguments"]["username"] == "chat"
            value = {"wechat.chat.get_messages": self.bridge, "wechat.chat.get_message_around": self.around,
                     "wechat.chat.resolve_chat_history": self.forward}[params["name"]]
            result = {"structuredContent": value}
        return {"jsonrpc": "2.0", "id": request["id"], "result": copy.deepcopy(result)}

    def get(self, route, **params):
        assert route == "/api/v1/messages" and params["media"] == "0"
        self.calls.append({"http": route, "params": params})
        return copy.deepcopy(self.http)

    def _check_response_account(self, value):
        if value.get("account_id") != self.account_id:
            raise WeFlowError("account mismatch", code="context_identity")


def test_mcp_bridge_uses_verified_namespace_and_bounded_context():
    client = Client()
    data = AlphaChatContext(client).fetch(ANCHOR)
    assert data.source == "mcp_around" and len(data.rows) == 3 and not data.complete_range
    assert data.rows[-1]["from_me"] is True
    assert all(not r["attachments"] for r in data.rows)
    requests = [r["params"] for r in client.calls if r.get("method") == "tools/call"]
    assert [r["name"] for r in requests] == ["wechat.chat.get_messages", "wechat.chat.get_message_around"]
    assert requests[-1]["arguments"]["before"] == requests[-1]["arguments"]["after"] == 20


@pytest.mark.parametrize("case", ["no_tool", "schema", "namespace"])
def test_unsupported_mcp_falls_back_to_http_with_bounded_budget(case):
    client = Client()
    if case == "no_tool":
        client.tools = []
    elif case == "schema":
        client.tools[0]["inputSchema"]["required"] = ["unsupported"]
    else:
        client.bridge = {"messages": [mcp_message("different-id")]}
    data = AlphaChatContext(client).fetch(ANCHOR)
    assert data.source == "http_window" and data.complete_range
    req = client.calls[-1]["params"]
    assert req["end"] - req["start"] <= 1800 and 1 <= req["limit"] <= 200
    assert not any(r.get("params", {}).get("name") == "wechat.chat.get_message_around" for r in client.calls)


@pytest.mark.parametrize("field,value", [("account_id", "another"), ("conversation_id", "another"),
                                        ("messages", [mcp_message("not-anchor")])])
def test_mcp_identity_mismatch_never_falls_back_or_displays_wrong_context(field, value):
    client = Client()
    client.around[field] = value
    with pytest.raises(WeFlowError):
        AlphaChatContext(client).fetch(ANCHOR)
    assert not any("http" in r for r in client.calls)


def test_mcp_tool_denylist_is_not_generic_agent_execution():
    client = Client()
    reader = AlphaChatContext(client)
    with pytest.raises(WeFlowError, match="白名单"):
        reader.rpc("tools/call", {"name": "wechat.chat.send_text"})
    assert not client.calls


def test_context_cache_is_scoped_by_revision_and_credentials():
    client, cache = Client(), ContextCache()
    first = AlphaChatContext(client, cache=cache).fetch(ANCHOR)
    n = len(client.calls)
    assert AlphaChatContext(client, cache=cache).fetch(ANCHOR) is first
    assert len(client.calls) == n
    client._token = "other-synthetic-token"
    AlphaChatContext(client, cache=cache).fetch(ANCHOR)
    assert len(client.calls) > n
    n = len(client.calls)
    AlphaChatContext(client, cache=cache).fetch({**ANCHOR, "revision": 2})
    assert len(client.calls) > n
    cache.clear()
    assert not cache.entries


def test_forward_authors_are_not_outer_sender_and_not_associated_by_name():
    data = AlphaChatContext(Client()).fetch(ANCHOR, kind="forward")
    assert data.source == "mcp_forward"
    assert [r["sender"] for r in data.rows] == ["原作者甲", "原作者乙"]
    assert [r["forward_path"] for r in data.rows] == [(1,), (2,)]
    assert all(r["from_me"] is None and not r["attachments"] for r in data.rows)


def test_context_filters_dates_without_expanding_and_no_false_complete():
    client = Client()
    client.tools = []
    client.http["scanComplete"] = False
    data = AlphaChatContext(client).fetch(ANCHOR, start=STAMP / 1000 - 20, end=STAMP / 1000 + 20)
    assert not data.complete_range
    assert client.calls[-1]["params"]["end"] - client.calls[-1]["params"]["start"] <= 40


def test_lookup_is_read_only_and_rejects_reused_api_source(corpus):
    key = "weflow:" + digest("\0".join((ANCHOR["base_url"], "owner", "chat")))
    ChatStore(corpus).ingest(platform="wechat", account="owner", source_key=key, conversation="测试群", kind="weflow_api",
        messages=[ChatMessage("123456", "合成上下文", sent_at="2026-10-20T00:00:00Z")])
    with corpus.conn:
        save_api_identities(corpus, key, ANCHOR["base_url"], "acct", {"123456": ("chat", MID, str(STAMP), "1")})
    hit = result(corpus, "123456", key)
    group = discussions(corpus.db_path, hit)[0][0]
    revision = corpus.index_revisions()
    anchor = anchor_record(corpus.db_path, hit, group, hit.chat_message_id)
    assert anchor["api_id"] == MID and corpus.index_revisions() == revision
    with corpus.conn:
        corpus.conn.execute("UPDATE chat_api_messages SET conversation_id='other'")
    with pytest.raises(WeFlowError):
        anchor_record(corpus.db_path, hit, group, hit.chat_message_id)


def test_http_rejects_changed_exact_identity_and_duplicates():
    client = Client()
    client.tools = []
    client.http["messages"][0]["platform_message_id"] = "another"
    with pytest.raises(WeFlowError):
        AlphaChatContext(client).fetch(ANCHOR)
    client = Client()
    client.tools = []
    client.http["messages"] *= 2
    with pytest.raises(WeFlowError):
        AlphaChatContext(client).fetch(ANCHOR)
