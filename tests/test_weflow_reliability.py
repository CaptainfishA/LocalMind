from __future__ import annotations

import html
import json

import pytest

from localmind.chat_limits import MAX_MESSAGE_CHARS, MAX_RAW_MESSAGE_CHARS
from localmind.chats import Attachment, ChatMessage, ChatStore, parse_export, normalize_time
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.models import SearchQuery
from localmind.weflow import WeFlowError, decode_messages, sync_session
from localmind import weflow
from localmind.weflow_protocol import MessageContentError
from test_weflow_alpha import alpha_client, alpha_message
from test_weflow_batch import START, END, batch_client, run
from test_weflow_v04 import index_file, ingest, searcher


@pytest.fixture
def db(tmp_path):
    value = LocalMindDB(tmp_path / "index.db")
    yield value
    value.close()


def checkpoint(db):
    return json.loads(db.conn.execute("SELECT value FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()[0])


def test_long_message_tail_search_and_stable_chunks(db, tmp_path):
    body = "项目背景与会议记录。" * 2400 + "蓝桥终稿验收口令"
    msg = decode_messages([alpha_message(1, text=body)])[0]
    assert len(msg.body) > 16000 and msg.body == body
    embedding = EmbeddingService(force_hash=True)
    first = ingest(db, [msg], embedding=embedding)
    assert first["updated_units"] > 30
    assert db.conn.execute("SELECT COUNT(DISTINCT message_id) FROM chat_units").fetchone()[0] == 1
    for mode in ("keyword", "semantic", "hybrid"):
        found = searcher(db, embedding, mode).search(SearchQuery("蓝桥终稿验收口令", scope="messages"))
        assert found and "蓝桥终稿验收口令" in found[0].snippet
    ids = db.conn.execute("SELECT chunk_id FROM vectors ORDER BY chunk_id").fetchall()
    assert ingest(db, [msg], embedding=embedding)["updated_units"] == 0
    assert db.conn.execute("SELECT chunk_id FROM vectors ORDER BY chunk_id").fetchall() == ids
    path = tmp_path / "long.json"
    path.write_text(json.dumps({"messages": [{"id": "long", "content": body}]}), encoding="utf-8")
    assert parse_export(path)[1][0].body == body


@pytest.mark.parametrize("size", [16000, 16001, MAX_MESSAGE_CHARS])
def test_limits_are_normalized_characters_not_bytes(size):
    body = "中" * size
    msg = decode_messages([alpha_message(1, text=body, quote={"content": "引用"})])[0]
    assert msg.body.startswith(body)
    assert len(msg.body) <= MAX_MESSAGE_CHARS
    if size == MAX_MESSAGE_CHARS:
        assert msg.body == body  # No main-body truncation for a preview.


@pytest.mark.parametrize("escaped", [False, True])
def test_large_xml_normalizes_before_body_limit(escaped):
    xml = "<msg><appmsg><type>6</type><title>预算审批.xlsx</title><appattach><totallen>123</totallen></appattach><extra>" + "x" * 160001 + "</extra></appmsg></msg>"
    msg = decode_messages([alpha_message(1, text=html.escape(xml) if escaped else xml)])[0]
    assert msg.body == "[文件] 预算审批.xlsx"
    assert msg.attachments[0].size == 123


@pytest.mark.parametrize("body,code", [
    ({"private": "must-not-leak"}, "body_type"),
    ([], "body_type"), (False, "body_type"),
    ("中" * (MAX_MESSAGE_CHARS + 1), "body_limit"),
    ("x" * (MAX_RAW_MESSAGE_CHARS + 1), "raw_limit"),
    ("bad\x00text", "invalid_text"),
    ("<msg><appmsg>", "xml_invalid"),
    ('<!DOCTYPE msg [<!ENTITY x "private">]><msg>&x;</msg>', "xml_unsafe"),
    ("<a>" * 65 + "text" + "</a>" * 65, "xml_complexity"),
    ("<msg>" + "<a/>" * 10001 + "</msg>", "xml_complexity"),
    ("<msg><unknown>private</unknown></msg>", "structured_unsupported"),
    ("<msg><appmsg><title>聊天记录</title><recorditem>private</recorditem></appmsg></msg>", "forward_unsupported"),
], ids=["object", "list", "boolean", "body-cap", "raw-cap", "nul", "broken-xml", "entity", "deep", "nodes", "unknown", "forward"])
def test_payload_failures_isolated_and_diagnostics_redacted(body, code):
    issues = []
    rows = [alpha_message(1, text=body), alpha_message(2, text="[文件] 正常合同.pdf")]
    decoded = decode_messages(rows, issues=issues)
    assert len(decoded) == 1 and decoded[0].attachments[0].name == "正常合同.pdf"
    assert len(issues) == 1 and issues[0]["code"] == code
    assert set(issues[0]) == {"message_key", "sent_at", "code", "chars"}
    assert len(issues[0]["message_key"]) == 64
    assert "private" not in json.dumps(issues)
    with pytest.raises(MessageContentError) as error:
        decode_messages(rows)
    assert error.value.code == code


def test_malformed_quote_is_isolated():
    issues = []
    assert not decode_messages([alpha_message(1, quote={"content": "bad\x00"})], issues=issues)
    assert issues[0]["code"] == "invalid_text"


def test_plain_comparison_is_not_xml():
    assert decode_messages([alpha_message(1, text="比较 a < b", rawContent="比较 a < b")])[0].body == "比较 a < b"


def test_invalid_attachment_and_normalized_forward_are_not_false_successes():
    issues = []
    rows = [alpha_message(1, text="<msg><appmsg><type>6</type><title/></appmsg></msg>"),
            alpha_message(2, text="[聊天记录]", rawContent="<msg><appmsg><recorditem>嵌套记录</recorditem></appmsg></msg>")]
    assert not decode_messages(rows, issues=issues)
    assert [issue["code"] for issue in issues] == ["attachment_invalid", "forward_unsupported"]


def test_large_transport_page_shrinks_without_skipping_rows(db, monkeypatch):
    client = batch_client(1)
    client.opener.rows["chat-0"] = [alpha_message(i, conversation_id="chat-0") for i in range(6)]
    original = client.get
    attempts = []

    def get(path, **params):
        if path == "/api/v1/messages":
            attempts.append((params["offset"], params["limit"]))
            if params["limit"] > 2:
                raise WeFlowError("响应过大", code="response_limit")
        return original(path, **params)

    monkeypatch.setattr(client, "get", get)
    result = run(db, client)
    assert result["state"] == "completed" and result["received"] == 6
    assert attempts[0] == (0, 500)
    assert [offset for offset, limit in attempts if limit <= 2] == list(range(6))
    assert ChatStore(db).sources()[0]["messages"] == 6


def test_accumulated_text_size_is_bounded_before_ingestion(db, monkeypatch):
    monkeypatch.setattr(weflow, "MAX_IMPORT_BYTES", 100)
    client = batch_client(1)
    client.opener.rows["chat-0"][0]["text"] = "x" * 101
    result = run(db, client)
    assert result["state"] == "partial" and result["failed_sessions"] == 1
    assert not ChatStore(db).sources()
    assert not db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()


@pytest.mark.parametrize("extra", [{"platform_message_id": 9e18}, {"sent_at_unix_ms": "invalid"}])
def test_identity_and_timestamp_errors_are_not_swallowed(extra):
    with pytest.raises(WeFlowError):
        decode_messages([alpha_message(1, **extra)], issues=[])


def test_partial_sync_survives_restart_and_recovers_old_range(db):
    client = alpha_client()
    client.opener.rows[0]["text"] = {"private": "must-not-leak"}
    embedding = EmbeddingService(force_hash=True)
    first = sync_session(db, client, "test-chat", start=START, end=END, embedding_service=embedding)
    assert first["state"] == "partial" and first["received"] == 3
    assert first["indexed_messages"] == 2 and first["failed_messages"] == 1
    assert ChatStore(db).sources()[0]["messages"] == 2
    assert checkpoint(db)["end"] == START and checkpoint(db)["retry_start"] == START
    issue = dict(db.conn.execute("SELECT * FROM chat_sync_issues").fetchone())
    assert "must-not-leak" not in str(issue) and issue["attempts"] == 1
    reopened = LocalMindDB(db.db_path)
    try:
        repeat = sync_session(reopened, client, "test-chat", start=START, end=END, embedding_service=embedding)
        assert repeat["updated_units"] == 0 and repeat["failed_messages"] == 1
        assert reopened.conn.execute("SELECT attempts FROM chat_sync_issues").fetchone()[0] == 2
        client.opener.rows[0]["text"] = "修复后的预算审批说明"
        client.opener.calls.clear()
        fixed = sync_session(reopened, client, "test-chat", start=START, end=END + 10000, embedding_service=embedding)
        assert fixed["state"] == "completed" and fixed["failed_messages"] == 0
        assert fixed["total"] == 3
        assert not reopened.conn.execute("SELECT 1 FROM chat_sync_issues").fetchone()
        calls = [q for p, q in client.opener.calls if p == "/api/v1/messages"]
        assert calls[0]["start"] == str(START)
        assert checkpoint(reopened) == {"requested_start": START, "end": END + 10000}
        assert not reopened.conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        reopened.close()


def test_batch_partial_status_and_retry_only_incomplete(db):
    client = batch_client(2)
    client.opener.rows["chat-0"].append(alpha_message(99, conversation_id="chat-0", text={"bad": True}))
    first = run(db, client)
    assert first["state"] == "partial" and first["partial_sessions"] == 1 and first["failed_sessions"] == 0
    assert first["completed_sessions"] == 2 and first["failed_messages"] == 1
    assert first["job"]["sessions"][0]["status"] == "partial"
    assert len(ChatStore(db).attachments()) == 2
    client.opener.rows["chat-0"][1]["text"] = "正常正文"
    client.opener.calls.clear()
    retry = run(db, client)
    assert retry["resumed"] and retry["state"] == "completed" and retry["failed_messages"] == 0
    assert {q["talker"] for p, q in client.opener.calls if p == "/api/v1/messages"} == {"chat-0"}


def test_all_bad_pages_are_counted_and_clearable(db):
    client = batch_client(1)
    client.opener.rows["chat-0"] = [alpha_message(i, conversation_id="chat-0", text={"bad": True}) for i in range(1001)]
    result = run(db, client)
    assert result["state"] == "partial" and result["failed_messages"] == 1001 and result["received"] == 1001
    assert not ChatStore(db).sources()
    assert db.conn.execute("SELECT COUNT(*) FROM chat_sync_issues").fetchone()[0] == 1001
    db.clear_index()
    assert not db.conn.execute("SELECT 1 FROM chat_sync_issues").fetchone()


def test_failure_outside_range_still_aborts(db):
    client = alpha_client()
    client.opener.rows[0].update(text={"bad": True}, sent_at_unix_ms=1)
    with pytest.raises(WeFlowError, match="时间范围"):
        sync_session(db, client, "test-chat", start=START, end=END)
    assert not ChatStore(db).sources() and not db.conn.execute("SELECT 1 FROM chat_sync_issues").fetchone()


def test_conflicting_valid_and_invalid_message_id_does_not_certify_range(db):
    client = batch_client(1)
    client.opener.rows["chat-0"] = [alpha_message(1, conversation_id="chat-0"),
                                    alpha_message(1, conversation_id="chat-0", text={"bad": True})]
    result = run(db, client)
    assert result["state"] == "partial" and result["failed_sessions"] == 1
    assert not ChatStore(db).sources()
    assert not db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()


def test_failed_message_missing_on_retry_is_not_silently_resolved(db):
    client = batch_client(1)
    client.opener.rows["chat-0"][0]["text"] = {"bad": True}
    run(db, client)
    client.opener.rows["chat-0"] = []
    retry = run(db, client)
    assert retry["state"] == "partial" and retry["failed_messages"] == 1


def test_remove_source_removes_message_retry_ledger(db):
    client = alpha_client()
    client.opener.rows[0]["text"] = {"bad": True}
    sync_session(db, client, "test-chat", start=START, end=END)
    ChatStore(db).remove_source(ChatStore(db).sources()[0]["id"])
    assert not db.conn.execute("SELECT 1 FROM chat_sync_issues").fetchone()


def test_file_context_and_content_survive_neighbor_payload_failure(db, tmp_path):
    client = batch_client(1)
    path = tmp_path / "交付清单.txt"
    embedding = EmbeddingService(force_hash=True)
    file = index_file(db, path, "付款节点和质量验收条款，交付后支付尾款。", embed=embedding)
    client.opener.rows["chat-0"] = [
        alpha_message(1, conversation_id="chat-0", text={"bad": True}),
        alpha_message(2, conversation_id="chat-0", text="这份是蓝桥项目最终交付版本"),
        alpha_message(3, conversation_id="chat-0", text="[文件] 交付清单.txt"),
    ]
    assert run(db, client, embedding_service=embedding)["state"] == "partial"
    att = ChatStore(db).attachments()[0]
    ChatStore(db).bind(att["id"], file["id"])
    service = searcher(db, embedding)
    for query in ("蓝桥项目交付版本", "付款节点质量验收"):
        results = service.search(SearchQuery(query, scope="chat_files"))
        assert results and results[0].path == str(path)
    assert path.read_text(encoding="utf-8").startswith("付款节点")


def test_long_direct_reply_tail_finds_attachment_without_cross_linking(db, tmp_path):
    embedding = EmbeddingService(force_hash=True)
    file = index_file(db, tmp_path / "普通文件.txt", "例行工作安排", embed=embedding)
    other = index_file(db, tmp_path / "其他文件.txt", "其他工作的安排", embed=embedding)
    messages = [ChatMessage("file", "[文件] 普通文件.txt", "甲", normalize_time(1780000000),
                            [Attachment("普通文件.txt", sha256=file["file_hash"])]),
                ChatMessage("reply", "例行讨论。" * 4000 + "发给南宁评审组的终稿", "乙", normalize_time(1780003600), reply_to="file")]
    messages.append(ChatMessage("other-file", "[文件] 其他文件.txt", "丙", normalize_time(1780000000),
                                [Attachment("其他文件.txt", sha256=other["file_hash"])]))
    ingest(db, messages, embedding=embedding)
    for mode in ("keyword", "semantic", "hybrid"):
        found = searcher(db, embedding, mode).search(SearchQuery("南宁评审组", scope="chat_files"))
        assert found and found[0].path == file["path"]
        assert "南宁评审组" in found[0].snippet
        if mode == "keyword":
            assert all(result.path == file["path"] for result in found)
    repeat = ingest(db, messages, embedding=embedding)
    assert repeat["updated_units"] == 0
    messages[1].body = "已改发桂林评审组"
    ingest(db, messages, embedding=embedding)
    assert not db.conn.execute("SELECT 1 FROM chunks WHERE instr(chunk_text, '南宁评审组') > 0").fetchone()
    assert searcher(db, embedding, "keyword").search(SearchQuery("桂林评审组", scope="chat_files"))[0].path == file["path"]
    assert not db.conn.execute("PRAGMA foreign_key_check").fetchall()
