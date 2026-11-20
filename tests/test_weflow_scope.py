from __future__ import annotations

import copy

import pytest

from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.weflow import SyncCancelled, WeFlowError
from localmind.weflow_batch import sync_all_sessions
from localmind.weflow_protocol import session_page
from localmind.weflow_scope import load_scope, make_scope, restored_ids, save_scope, validate_scope
from test_weflow_batch import START, END, batch_client, run
from test_weflow_alpha import alpha_message


@pytest.fixture
def db(tmp_path):
    value = LocalMindDB(tmp_path / "index.db")
    yield value
    value.close()


def selection(client, ids):
    rows = client.sessions()
    return make_scope(client.base_url, client.account_id, rows, ids)


def message_calls(client):
    return [q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"]


def test_category_uses_exact_metadata_and_mirror():
    payload = {"items": [{"id": "person", "display_name": "项目交流群"}, {"id": "group"},
                         {"id": "official"}, {"id": "company", "sessionType": "enterprise"},
                         {"id": "mystery", "display_name": "公众号"}, {"id": "known@chatroom"}],
               "sessions": [{"username": "person", "sessionType": "private", "remark": "王同事"},
                            {"username": "group", "sessionType": "group"},
                            {"username": "official", "sessionType": "official"}], "hasMore": False}
    rows = session_page(payload, 0, alpha=True).sessions
    assert [r["category"] for r in rows] == ["private", "group", "official", "enterprise", "unknown", "group"]
    assert rows[0]["aliases"] == ["王同事"]
    assert restored_ids(rows, None) == {"person", "group", "known@chatroom"}


@pytest.mark.parametrize("kind", [0, 1, [], {}, "PRIVATE", "not-supported"])
def test_unknown_enums_are_not_guessed(kind):
    rows = session_page({"sessions": [{"username": "any", "sessionType": kind}], "hasMore": False}, 0, alpha=False).sessions
    assert rows[0]["category"] == "unknown" and restored_ids(rows, None) == set()


def test_selected_only_and_preferences_are_account_bound(db):
    client = batch_client(4)
    scope = selection(client, ["chat-1", "chat-3"])
    save_scope(db, scope)
    assert load_scope(db, client.base_url, client.account_id) == scope
    assert load_scope(db, client.base_url, "different") is None
    assert load_scope(db, "http://127.0.0.1:5032", client.account_id) is None
    assert load_scope(db, client.base_url, None) is None
    result = run(db, client, scope=scope)
    assert result["state"] == "completed" and result["total_sessions"] == 2
    assert message_calls(client) == ["chat-1", "chat-3"]
    assert len(ChatStore(db).sources()) == 2
    assert "synthetic-batch-token" not in str(scope)
    db.clear_index()
    assert load_scope(db, client.base_url, client.account_id) is None


@pytest.mark.parametrize("change", ["account", "url", "ids", "types"])
def test_tampered_or_wrong_account_scope_stops_before_messages(db, change):
    client = batch_client(2)
    scope = selection(client, ["chat-0"])
    if change == "account":
        scope["account_id"] = "different"
    elif change == "url":
        scope["base_url"] = "http://127.0.0.1:5032"
    elif change == "ids":
        scope["ids"].append("chat-1")
    else:
        scope["types"]["chat-0"] = []
    with pytest.raises(WeFlowError):
        run(db, client, scope=scope)
    assert not message_calls(client)
    assert not db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_batch:%'").fetchone()


def test_empty_selection_cannot_mean_all(db):
    client = batch_client(2)
    with pytest.raises(ValueError, match="至少"):
        selection(client, [])
    scope = selection(client, ["chat-0"])
    scope["ids"] = []
    with pytest.raises(WeFlowError):
        run(db, client, scope=scope)
    assert not message_calls(client)


def test_resume_and_refresh_never_add_new_conversations(db):
    client = batch_client(3)
    scope = selection(client, ["chat-0", "chat-1"])
    client.opener.errors["chat-1"] = 400
    assert run(db, client, scope=scope)["state"] == "partial"
    client.opener.errors.clear()
    client.opener.sessions.append({"id": "new-chat", "display_name": "新群聊", "sessionType": "group"})
    client.opener.rows["new-chat"] = [alpha_message(101, conversation_id="new-chat")]
    client.opener.calls.clear()
    result = sync_all_sessions(db, client, scope=scope, start=START, end=END + 100)
    assert result["resumed"] and result["end"] == END
    assert result["total_sessions"] == 2 and result["new_sessions"] == 1
    assert message_calls(client) == ["chat-1"]
    client.opener.calls.clear()
    result = sync_all_sessions(db, client, scope=scope, start=START, end=END + 200)
    assert not result["resumed"] and result["state"] == "completed" and result["new_sessions"] == 1
    assert message_calls(client) == ["chat-0", "chat-1"]
    assert len(ChatStore(db).sources()) == 2
    assert restored_ids(client.sessions(), scope) == {"chat-0", "chat-1"}


def test_changed_selection_creates_new_queue_without_deleting_old_index(db):
    client = batch_client(3)
    first = selection(client, ["chat-0", "chat-1"])
    client.opener.errors["chat-1"] = 400
    run(db, client, scope=first)
    client.opener.errors.clear()
    second = selection(client, ["chat-2"])
    client.opener.calls.clear()
    result = run(db, client, scope=second)
    assert not result["resumed"] and result["total_sessions"] == 1
    assert message_calls(client) == ["chat-2"]
    assert {r["conversation"] for r in ChatStore(db).sources()} == {"chat-0", "chat-2"}


@pytest.mark.parametrize("change", ["missing", "type"])
def test_missing_or_reclassified_session_is_not_read(db, change):
    client = batch_client(2)
    scope = selection(client, ["chat-0", "chat-1"])
    if change == "missing":
        client.opener.sessions.pop(0)
    else:
        client.opener.sessions[0]["sessionType"] = "official"
    client.opener.calls.clear()
    result = run(db, client, scope=scope)
    assert result["failed_sessions"] == 1 and result["succeeded_sessions"] == 1
    assert message_calls(client) == ["chat-1"]
    assert "chat-0" not in restored_ids(client.sessions(), scope)


def test_preferences_do_not_expand_after_restart(db):
    client = batch_client(2)
    scope = selection(client, ["chat-0"])
    save_scope(db, scope)
    reopened = LocalMindDB(db.db_path)
    try:
        assert load_scope(reopened, client.base_url, client.account_id) == scope
        rows = client.sessions() + [{"id": "new", "name": "新会话", "category": "private"}]
        assert restored_ids(rows, load_scope(reopened, client.base_url, client.account_id)) == {"chat-0"}
    finally:
        reopened.close()


def test_metadata_only_changes_do_not_invalidate_explicit_ids():
    client = batch_client(2)
    first = selection(client, ["chat-0"])
    second = copy.deepcopy(first)
    second["known_ids"].append("new")
    assert validate_scope(second, client.base_url, client.account_id)["fingerprint"] == first["fingerprint"]


def test_conflicting_category_metadata_is_unknown():
    payload = {"items": [{"id": "a@chatroom", "sessionType": "group"}],
               "sessions": [{"username": "a@chatroom", "sessionType": "official"}], "hasMore": False}
    row = session_page(payload, 0, alpha=True).sessions[0]
    assert row["category"] == "unknown" and row["category_source"] == "conflict"


def test_cancel_after_last_metadata_page_does_not_open_selection():
    client = batch_client(2)
    cancelled = False

    def progress(_message):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(SyncCancelled):
        client.sessions(should_cancel=lambda: cancelled, progress=progress)
    assert not message_calls(client)
