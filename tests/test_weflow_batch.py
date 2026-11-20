from __future__ import annotations

import io
import json
import urllib.error
from threading import Event
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.models import SearchQuery
from localmind.reranker import RerankerService
from localmind.searcher import SearchService
from localmind.weflow import WeFlowClient
from localmind.weflow_batch import batch_key, load_batch, sync_all_sessions
from localmind.weflow_protocol import session_page
from test_weflow_alpha import alpha_message


START, END = 1779990000, 1780000100


class BatchTransport:
    def __init__(self, count=3):
        self.sessions = [{"id": f"chat-{i}", "display_name": f"项目会话{i}", "sessionType": "private"} for i in range(count)]
        self.rows = {s["id"]: [alpha_message(i + 1, conversation_id=s["id"], text=f"[文件] 项目资料{i}.pdf")]
                     for i, s in enumerate(self.sessions)}
        self.errors = {}
        self.owners = {}
        self.calls = []

    def open(self, request, timeout):
        url = urlsplit(request.full_url)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        self.calls.append((url.path, query))
        if url.path == "/health":
            value = {"status": "ok"}
        elif url.path == "/api/v1/accounts":
            value = {"accounts": [{"id": "test-account", "display_name": "模拟账号", "selected": True}],
                     "selected_account_id": "test-account"}
        elif url.path == "/api/v1/sessions":
            offset, limit = int(query["offset"]), int(query["limit"])
            assert limit <= 200 and "keyword" not in query
            value = {"items": self.sessions[offset:offset + limit], "hasMore": offset + limit < len(self.sessions),
                     "nextOffset": offset + limit, "account_id": "test-account"}
        elif url.path.startswith("/api/v1/sessions/"):
            sid = unquote(url.path.split("/")[-2])
            value = {"meta": {"ownerId": self.owners.get(sid, "test-owner"), "name": sid}, "members": []}
        elif url.path == "/api/v1/messages":
            sid = query["talker"]
            if sid in self.errors:
                raise urllib.error.HTTPError(request.full_url, self.errors[sid], "synthetic", {}, io.BytesIO(b"{}"))
            rows = [r for r in self.rows[sid] if int(query["start"]) <= r["sent_at_unix_ms"] // 1000 <= int(query["end"])]
            offset, limit = int(query["offset"]), int(query["limit"])
            value = {"messages": rows[offset:offset + limit], "account_id": "test-account", "talker": sid,
                     "hasMore": offset + limit < len(rows), "nextOffset": offset + limit}
        else:
            raise AssertionError("unexpected request")
        return io.BytesIO(json.dumps(value).encode())


def batch_client(count=3):
    client = WeFlowClient(token="synthetic-batch-token")
    client.opener = BatchTransport(count)
    return client


@pytest.fixture
def db(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    yield db
    db.close()


def run(db, client, **kwargs):
    return sync_all_sessions(db, client, start=START, end=END, **kwargs)


def test_one_click_all_conversations_and_repeat(db):
    client = batch_client()
    embedding = EmbeddingService(force_hash=True)
    first = run(db, client, embedding_service=embedding)
    assert first["state"] == "completed" and first["succeeded_sessions"] == 3
    assert first["received"] == 3 and len(ChatStore(db).attachments()) == 3
    repeat = run(db, client, embedding_service=embedding)
    assert repeat["updated_units"] == 0 and repeat["reused_units"] == 3
    assert len(ChatStore(db).sources()) == 3
    service = SearchService(db, embedding_service=embedding, reranker_service=RerankerService(enabled=False))
    assert len(service.search(SearchQuery("项目资料", scope="chat_files", limit=10))) == 3
    assert "synthetic-batch-token" not in db.get_setting(batch_key(client))
    assert not db.conn.execute("PRAGMA foreign_key_check").fetchall()


def test_failure_isolation_and_resume_only_failed(db):
    client = batch_client()
    client.opener.errors["chat-1"] = 400
    first = run(db, client)
    assert first["state"] == "partial" and first["failed_sessions"] == 1 and first["succeeded_sessions"] == 2
    assert len(ChatStore(db).sources()) == 2
    client.opener.errors.clear()
    client.opener.calls.clear()
    result = run(db, client)
    assert result["resumed"] and result["state"] == "completed"
    assert [q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"] == ["chat-1"]


def test_pause_resume_after_reopening_database(db):
    client = batch_client()
    cancel = Event()

    def update(event):
        if event.get("row", {}).get("status") == "done":
            cancel.set()

    result = run(db, client, on_update=update, should_cancel=cancel.is_set)
    assert result["state"] == "paused" and result["succeeded_sessions"] == 1
    second = LocalMindDB(db.db_path)
    client.opener.calls.clear()
    try:
        result = run(second, client)
        assert result["resumed"] and result["state"] == "completed"
        assert [q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"] == ["chat-1", "chat-2"]
    finally:
        second.close()


def test_pause_before_current_request_does_not_advance(db):
    client = batch_client()
    cancel = Event()

    def update(event):
        if event.get("row", {}).get("status") == "running":
            cancel.set()

    result = run(db, client, on_update=update, should_cancel=cancel.is_set)
    assert result["state"] == "paused" and result["succeeded_sessions"] == 0
    assert not ChatStore(db).sources()
    assert not any(p == "/api/v1/messages" for p, _ in client.opener.calls)
    assert not db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()


def test_activity_skip_only_when_known_older_than_range(db):
    client = batch_client()
    client.opener.sessions[0]["last_activity_unix_ms"] = (START - 1) * 1000
    client.opener.sessions[1]["last_activity_unix_ms"] = "invalid"
    result = run(db, client)
    assert result["skipped_sessions"] == 1 and result["succeeded_sessions"] == 2
    assert {q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"} == {"chat-1", "chat-2"}
    page = session_page({"sessions": [{"username": "legacy", "lastTimestamp": START}], "hasMore": False}, 0, alpha=False)
    assert page.sessions[0]["last_activity"] == START


@pytest.mark.parametrize("status", [401, 403, 423, 429, 503])
def test_global_access_failure_stops_queue(db, status):
    client = batch_client()
    client.opener.errors["chat-0"] = status
    result = run(db, client)
    assert result["state"] == "partial" and result["failed_sessions"] == 1
    assert [q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"] == ["chat-0"]


def test_account_identity_change_stops_before_reading_next_conversation(db):
    client = batch_client()
    client.opener.owners["chat-1"] = "different-owner"
    result = run(db, client)
    assert result["state"] == "partial" and result["succeeded_sessions"] == 1
    assert [q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"] == ["chat-0"]


def test_message_cap_is_explicit_failure_not_truncation(db):
    client = batch_client(2)
    client.opener.rows["chat-0"] *= 2
    result = run(db, client, max_messages=1)
    assert result["state"] == "partial" and result["succeeded_sessions"] == 1
    assert result["job"]["sessions"][0]["status"] == "failed"
    assert len(ChatStore(db).sources()) == 1


def test_clear_and_remove_invalidate_resume_metadata(db):
    client = batch_client()
    run(db, client)
    ChatStore(db).remove_source(ChatStore(db).sources()[0]["id"])
    assert load_batch(db, client) is None
    run(db, client)
    db.clear_index()
    assert load_batch(db, client) is None and not ChatStore(db).sources()


def test_empty_and_paginated_account(db):
    client = batch_client(0)
    assert run(db, client)["state"] == "completed"
    client = batch_client(405)
    for row in client.opener.sessions:
        row["last_activity_unix_ms"] = (START - 1) * 1000
    result = run(db, client)
    assert result["total_sessions"] == result["skipped_sessions"] == 405
    assert [q["offset"] for p, q in client.opener.calls if p == "/api/v1/sessions"] == ["0", "200", "400"]


def test_resume_keeps_snapshot_and_next_run_discovers_new_messages(db):
    client = batch_client(2)
    client.opener.errors["chat-1"] = 400
    run(db, client)
    client.opener.errors.clear()
    client.opener.rows["chat-0"].append(alpha_message(99, conversation_id="chat-0", sent_at_unix_ms=(END + 10) * 1000))
    client.opener.sessions.append({"id": "chat-new", "display_name": "新会话"})
    client.opener.rows["chat-new"] = [alpha_message(100, conversation_id="chat-new")]
    resumed = sync_all_sessions(db, client, start=START, end=END + 100)
    assert resumed["resumed"] and resumed["end"] == END
    assert resumed["total_sessions"] == 3 and resumed["state"] == "completed"
    assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 3
    fresh = sync_all_sessions(db, client, start=START, end=END + 100)
    assert not fresh["resumed"] and fresh["end"] == END + 100
    assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 4


@pytest.mark.parametrize("options", [{"incremental": False}, {"embedding_service": "hash"}])
def test_resume_does_not_skip_when_rebuilding_or_enabling_vectors(db, options):
    client = batch_client(2)
    client.opener.errors["chat-1"] = 400
    run(db, client)
    client.opener.errors.clear()
    client.opener.calls.clear()
    if options.get("embedding_service") == "hash":
        options = {"embedding_service": EmbeddingService(force_hash=True)}
    result = run(db, client, **options)
    assert not result["resumed"] and result["state"] == "completed"
    assert [q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"] == ["chat-0", "chat-1"]


def test_pause_during_embedding_preserves_current_conversation_atomicity(db):
    cancel = Event()

    class CancellingEmbedding(EmbeddingService):
        def encode_documents(self, texts):
            result = super().encode_documents(texts)
            cancel.set()
            return result

    client = batch_client()
    result = run(db, client, embedding_service=CancellingEmbedding(force_hash=True), should_cancel=cancel.is_set)
    assert result["state"] == "paused" and result["job"]["sessions"][0]["status"] == "pending"
    assert not ChatStore(db).sources()
    assert not db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()
