from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore, normalize_time, parse_export
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.indexer import IndexingService
from localmind.models import SearchQuery
from localmind.ocr import OcrService
from localmind.reranker import RerankerService
from localmind.searcher import SearchService
from localmind.vector_index import LocalVectorIndex
from localmind.weflow import SyncCancelled, WeFlowClient, WeFlowError, _xml_attachment, decode_messages, local_base_url, sync_session
from weflow_mock import MockWeFlow, message


@pytest.fixture
def db(tmp_path):
    value = LocalMindDB(tmp_path / "index.db")
    yield value
    value.close()


class CountingEmbedding(EmbeddingService):
    def __init__(self):
        super().__init__(force_hash=True)
        self.encoded = 0

    def encode_documents(self, texts):
        self.encoded += len(texts)
        return super().encode_documents(texts)


def index_file(db, path, text="付款节点百分之三十", embed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    service = IndexingService(db, embedding_service=embed or EmbeddingService(force_hash=True), ocr_service=OcrService(enabled=False))
    service.index_path(path)
    return dict(db.get_file_by_path(str(path)))


def ingest(db, messages, *, source="mock-source", conversation="模拟项目群", embedding=None):
    return ChatStore(db).ingest(platform="wechat", account="mock_owner", conversation=conversation,
                                source_key=source, messages=messages, embedding_service=embedding)


def searcher(db, embed=None, mode="hybrid"):
    return SearchService(db, embedding_service=embed or EmbeddingService(force_hash=True),
                         reranker_service=RerankerService(enabled=False), retrieval_mode=mode)


@pytest.mark.parametrize("url", ["https://127.0.0.1:5031", "http://example.com", "http://0.0.0.0:5031", "http://user:secret@localhost", "http://localhost/?access_token=x", "file:///tmp/a", "http://192.168.1.1:5031", "http://127.0.0.1:0"])
def test_only_loopback(url):
    with pytest.raises(WeFlowError):
        local_base_url(url)


def test_auth_redirect_and_proxy(monkeypatch):
    assert local_base_url("http://localhost:5031/api/v1/") == "http://127.0.0.1:5031"
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    with MockWeFlow() as mock:
        client = WeFlowClient(mock.url, mock.token)
        assert client.sessions()[0]["id"] == mock.session
        with pytest.raises(WeFlowError, match="鉴权失败") as error:
            WeFlowClient(mock.url, "private-value").sessions()
        assert "private-value" not in str(error.value)
        mock.redirect = True
        with pytest.raises(WeFlowError, match="重定向"):
            client.sessions()
        assert all("access_token" not in query for _, query in mock.calls)


def test_pagination_incremental_no_reembedding(db):
    embedding = CountingEmbedding()
    rows = [message(i, f"模拟设计记录{i}") for i in range(1, 1002)]
    with MockWeFlow(rows) as mock:
        client = WeFlowClient(mock.url, mock.token)
        result = sync_session(db, client, mock.session, start=1779990000, end=1780000050, embedding_service=embedding)
        assert result["received"] == 1001
        assert result["total"] == 1001
        count = embedding.encoded
        result = sync_session(db, client, mock.session, start=1779990000, end=1780000100, embedding_service=embedding)
        assert result["updated_units"] == 0
        assert embedding.encoded == count
        mock.rows += [message("late", "迟到的设计附件", timestamp=1780000001), message("new", "新增项目消息", timestamp=1780000110)]
        result = sync_session(db, client, mock.session, start=1779990000, end=1780000120, embedding_service=embedding)
        assert result["total"] == 1003
        assert 0 < result["updated_units"] < 20
        assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 1003
        calls = [q for p, q in mock.calls if p == "/api/v1/messages"]
        assert [int(q["offset"]) for q in calls[:3]] == [0, 500, 1000]
        assert all(q["end"] == "1780000050" for q in calls[:3])
        assert all(q["media"] == "0" for q in calls)
        assert calls[3]["start"] == "1779999750"


@pytest.mark.parametrize("failure", ["network", "cap", "account", "repeat", "cancel"])
def test_failed_sync_does_not_write_or_advance(db, failure):
    with MockWeFlow([message(i) for i in range(1, 601)]) as mock:
        if failure == "network":
            mock.fail_offset = 500
        if failure == "account":
            mock.switch_owner = True
        if failure == "repeat":
            mock.repeat_page = True
        with pytest.raises(WeFlowError):
            sync_session(db, WeFlowClient(mock.url, mock.token), mock.session,
                         start=1779990000, end=1780000010,
                         max_messages=300 if failure == "cap" else 2000,
                         should_cancel=lambda: failure == "cancel")
        assert not ChatStore(db).sources()
        assert not db.conn.execute("SELECT * FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchall()


def test_attachment_xml_and_quote():
    xml = '<msg><appmsg><title>设计方案.txt</title><type>6</type><appattach><totallen>30</totallen><filemd5>' + "a" * 32 + '</filemd5></appattach><refermsg><svrid>old-id</svrid></refermsg></appmsg></msg>'
    decoded = decode_messages([message(1234567890123456789, "[文件] 设计方案.txt", rawContent=xml)], {"mock_member": "张工"})[0]
    assert decoded.external_id == "1234567890123456789"
    assert decoded.sender == "张工" and decoded.reply_to == "old-id"
    assert decoded.attachments[0].md5 == "a" * 32
    assert decoded.attachments[0].size == 30
    assert _xml_attachment('<!DOCTYPE msg [<!ENTITY x SYSTEM "file:///private">]><msg>&x;</msg>') == ([], None)
    assert _xml_attachment(xml.replace("设计方案.txt", "设计&amp;方案.txt"))[0][0].name == "设计&方案.txt"
    quoted = decode_messages([message("q", "确认", quote={"accountName": "李工", "content": "历史预算的解释"})])[0]
    assert "引用快照" in quoted.body and "历史预算" in quoted.body


def test_context_content_scope_link_update_delete(db, tmp_path):
    embedding = CountingEmbedding()
    path = tmp_path / "docs" / "方案.txt"
    file = index_file(db, path, embed=embedding)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    messages = [ChatMessage("1", "这是星河项目的最终设计稿", "李工", normalize_time(1780000000)),
                ChatMessage("2", "[文件] 方案.txt", "张工", normalize_time(1780000001), [Attachment("方案.txt", size=file["size"], md5=md5)]),
                ChatMessage("3", "确认按照这个版本给客户", "李工", normalize_time(1780000500), reply_to="2")]
    result = ingest(db, messages, embedding=embedding)
    att = ChatStore(db).attachments()[0]
    assert att["file_id"] == file["id"] and att["link_state"] == "hash"
    service = searcher(db, embedding)
    query = SearchQuery("星河项目", scope="chat_files", conversation="模拟项目", sender="张工")
    found = service.search(query)
    assert found[0].path == str(path)
    assert "聊天上下文" in found[0].match_reason
    assert "张工" in found[0].chat_context
    found = service.search(SearchQuery("付款节点", scope="chat_files", conversation="模拟项目"))
    assert found[0].path == str(path)
    assert "文件正文" in found[0].match_reason
    assert not service.search(SearchQuery("付款节点", scope="chat_files", conversation="不匹配群"))
    found = service.search(SearchQuery("星河项目", scope="messages"))
    assert all(r.result_kind == "message" for r in found)
    index_file(db, path, "新的付款条款和金额", embed=embedding)
    service.refresh_vector_index()
    assert "本地内容已变化" in service.search(query)[0].chat_context
    assert db.remove_missing_files(tmp_path / "docs", {str(path)}) == 0
    assert ChatStore(db).sources()[0]["messages"] == 3
    ChatStore(db).remove_source(result["source_id"])
    assert path.exists() and db.get_file_by_path(str(path))
    assert not ChatStore(db).sources()
    assert not db.conn.execute("SELECT * FROM chat_units").fetchall()
    assert not db.conn.execute("PRAGMA foreign_key_check").fetchall()


def test_duplicate_name_requires_manual_confirmation(db, tmp_path):
    first = index_file(db, tmp_path / "one" / "方案.txt")
    index_file(db, tmp_path / "two" / "方案.txt", "其他内容")
    ingest(db, [ChatMessage("f", "附件", attachments=[Attachment("方案.txt")])])
    att = ChatStore(db).attachments()[0]
    assert att["file_id"] is None and att["link_state"] == "ambiguous"
    ChatStore(db).bind(att["id"], first["id"])
    assert ChatStore(db).attachments()[0]["link_state"] == "manual"
    db.clear_index()
    assert Path(first["path"]).exists()
    assert not ChatStore(db).sources()
    assert not searcher(db).search("方案")


def test_filter_before_candidate_limit_and_long_message(db):
    embedding = CountingEmbedding()
    ingest(db, [ChatMessage(str(i), "重复的项目设计", "甲") for i in range(180)], source="noise", conversation="干扰群", embedding=embedding)
    ingest(db, [ChatMessage("target", "一般背景。" * 200 + "超长末尾验收口令", "乙")], source="target", conversation="目标群", embedding=embedding)
    service = searcher(db, embedding)
    index = service.vector_index._faiss_index
    found = service.search(SearchQuery("超长末尾验收口令", scope="messages", conversation="目标群"))
    assert len(found) == 1 and "超长末尾验收口令" in found[0].snippet
    assert "目标群" in found[0].chat_context
    service.search(SearchQuery("项目设计", conversation="目标群"))
    assert service.vector_index._faiss_index is index


def test_filtered_vector_search():
    vector = LocalVectorIndex()
    vector.rebuild([1, 2, 3], np.eye(3, dtype="float32"))
    assert vector.search(np.array([1, 0, 0]), allowed_ids={2, 3}, limit=1)[0][0] in {2, 3}
    assert vector.search(np.ones(3), allowed_ids=set()) == []
    assert vector.search(np.ones(3), limit=0) == []


def test_parse_chatlab_json_and_transaction_rollback(db, tmp_path):
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"meta": {"platform": "im", "ownerId": "mock_owner"}, "messages": [
        {"platformMessageId": "123", "timestamp": 1780000000, "sender": "张工", "type": 4, "content": "[文件] 合同.txt"}]}), encoding="utf-8")
    meta, messages = parse_export(path)
    assert messages[0].external_id == "123"
    assert messages[0].attachments[0].kind == "explicit"
    embedding = CountingEmbedding()
    embedding.encode_documents = lambda texts: (_ for _ in ()).throw(RuntimeError("模拟编码失败"))
    with pytest.raises(RuntimeError, match="编码失败"):
        ingest(db, messages, embedding=embedding)
    assert not ChatStore(db).sources()
    assert db.stats().chunks_total == 0


def test_incremental_failure_preserves_checkpoint(db):
    with MockWeFlow([message(1)]) as mock:
        client = WeFlowClient(mock.url, mock.token)
        sync_session(db, client, mock.session, start=1779990000, end=1780000010)
        checkpoint = db.conn.execute("SELECT value FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()[0]
        mock.fail_offset = 0
        with pytest.raises(WeFlowError):
            sync_session(db, client, mock.session, start=1779990000, end=1780000020)
        assert db.conn.execute("SELECT value FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()[0] == checkpoint
        assert ChatStore(db).sources()[0]["messages"] == 1
        ChatStore(db).remove_source(ChatStore(db).sources()[0]["id"])
        assert not db.conn.execute("SELECT * FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchall()


def test_deleted_attachment_becomes_pending_and_chat_dates(db, tmp_path):
    path = tmp_path / "docs" / "附件.txt"
    file = index_file(db, path)
    ingest(db, [ChatMessage("a", "测试交付物", "张工", "2026-10-01T16:30:00Z",
                            [Attachment(path.name, sha256=file["file_hash"])])])
    service = searcher(db)
    query = SearchQuery("测试交付物", scope="chat_files", modified_after="2026-10-02T00:00:00+08:00", modified_before="2026-10-03T00:00:00+08:00")
    assert service.search(query)[0].result_kind == "file"
    query.modified_after = "2026-10-03T00:00:00+08:00"
    assert not service.search(query)
    path.unlink()
    db.remove_missing_files(path.parent, set())
    service.refresh_vector_index()
    result = service.search(SearchQuery("测试交付物", scope="chat_files"))[0]
    assert result.result_kind == "attachment_pending"
    assert "未找到本地文件" in result.chat_context
    assert ChatStore(db).sources()[0]["messages"] == 1


def test_md5_cache_avoids_reopening_file(db, tmp_path, monkeypatch):
    path = tmp_path / "缓存.txt"
    file = index_file(db, path)
    store = ChatStore(db)
    expected = hashlib.md5(path.read_bytes()).hexdigest()
    assert store._file_md5(file["id"]) == expected
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should use cached digest")))
    assert store._file_md5(file["id"]) == expected
