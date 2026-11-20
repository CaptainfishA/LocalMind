from __future__ import annotations

from dataclasses import replace
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.models import FileRecord, SearchQuery, TextChunk, utc_now_iso
from localmind.reranker import RerankerService
from localmind.searcher import SearchService
from localmind.vector_index import LocalVectorIndex


class FixedQuery:
    active_model_name = "snapshot-test"
    load_error = None

    def encode_query(self, text):
        return np.array([1.0, 0.0], dtype="float32")


class CountedIndex(LocalVectorIndex):
    def __init__(self):
        super().__init__()
        self.rebuilds = 0

    def rebuild(self, ids, vectors):
        self.rebuilds += 1
        super().rebuild(ids, vectors)


def document(db, path, text, vector=(1.0, 0.0)):
    path.write_text(text, encoding="utf-8")
    now = utc_now_iso()
    record = FileRecord(str(path), path.name, "txt", path.stat().st_size, now, now, text)
    fid, ids = db.replace_file_index(
        record, [TextChunk(text, "txt")], parse_status="done", ocr_status="not_required",
        embeddings=np.array([vector], dtype="float32"), model_name=FixedQuery.active_model_name,
    )
    return fid, ids[0], record


@pytest.fixture
def setup(tmp_path):
    reader = LocalMindDB(tmp_path / "index.db")
    fid, cid, record = document(reader, tmp_path / "original.txt", "付款条款")
    index = CountedIndex()
    service = SearchService(reader, embedding_service=FixedQuery(), vector_index=index,
                            reranker_service=RerankerService(enabled=False), retrieval_mode="semantic")
    writer = LocalMindDB(reader.db_path)
    try:
        yield reader, writer, service, index, fid, cid, record
    finally:
        writer.close()
        reader.close()


def test_reused_chunk_id_never_uses_previous_file_vector(setup, tmp_path):
    reader, writer, service, index, fid, cid, record = setup
    assert service.search("付款")[0].semantic_score == pytest.approx(1)
    writer.delete_file_by_path(record.path)
    new_fid, new_cid, _ = document(writer, tmp_path / "replacement.txt", "安装说明", (0.0, 1.0))
    assert new_cid == cid and new_fid == fid  # SQLite IDs can be reused after deletion.
    results = service.search("付款")
    assert results[0].name == "replacement.txt"
    assert results[0].semantic_score == pytest.approx(0)
    assert index.rebuilds == 2


def test_writer_commit_between_vector_recall_and_text_fetch_uses_one_snapshot(setup, monkeypatch):
    reader, writer, service, index, fid, cid, record = setup
    original = service._semantic_candidates
    committed = []

    def update_after_recall(*args):
        candidates = original(*args)
        writer.replace_file_index(
            replace(record, file_hash="new"), [TextChunk("修改后的安装说明", "txt")],
            parse_status="done", ocr_status="not_required",
            embeddings=np.array([[0.0, 1.0]], dtype="float32"), model_name=FixedQuery.active_model_name,
        )
        committed.append(True)
        return candidates

    monkeypatch.setattr(service, "_semantic_candidates", update_after_recall)
    result = service.search(SearchQuery("付款", scope="files"))[0]
    assert committed
    assert "付款条款" in result.snippet
    assert result.semantic_score == pytest.approx(1)
    assert not service.last_diagnostics.index_current
    monkeypatch.setattr(service, "_semantic_candidates", original)
    result = service.search(SearchQuery("付款", scope="files"))[0]
    assert "修改后的安装说明" in result.snippet
    assert result.semantic_score == pytest.approx(0)
    assert not reader.conn.in_transaction
    assert service.last_diagnostics.index_current


def test_metadata_rebind_invalidates_scope_without_rebuilding_vectors(setup, tmp_path):
    reader, writer, service, index, fid, _, record = setup
    other, _, _ = document(writer, tmp_path / "other.txt", "安装说明", (0.0, 1.0))
    ChatStore(writer).ingest(platform="wechat", account="test", conversation="合同群", source_key="sample",
                            messages=[ChatMessage("1", "合同讨论", attachments=[Attachment(record.name)])])
    aid = ChatStore(writer).attachments()[0]["id"]
    ChatStore(writer).bind(aid, fid)
    query = SearchQuery("付款", scope="chat_files", conversation="合同群", evidence="content")
    assert service.search(query)[0].file_id == fid
    rebuilds = index.rebuilds
    ChatStore(writer).bind(aid, other)
    results = service.search(query)
    assert len(results) == 1 and results[0].file_id == other
    assert results[0].semantic_score == pytest.approx(0)
    assert index.rebuilds == rebuilds
    with writer.conn:
        writer.conn.execute("UPDATE chat_sources SET conversation='方案群'")
    assert service.search(query) == []
    assert index.rebuilds == rebuilds


def test_settings_and_unchanged_reads_do_not_rebuild_index(setup):
    reader, writer, service, index, _, _, _ = setup
    before = reader.index_revisions()
    for i in range(8):
        writer.set_setting("search_retrieval_mode", "semantic" if i % 2 else "hybrid")
        writer.set_setting("reranker_enabled", "0")
        service.search("付款")
    assert index.rebuilds == 1
    assert reader.index_revisions() == before
    writer.set_setting("embedding_model", FixedQuery.active_model_name)
    service.search("付款")
    assert index.rebuilds == 2
    before = reader.index_revisions()
    writer.set_setting("embedding_model", FixedQuery.active_model_name)
    service.search("付款")
    assert index.rebuilds == 2 and reader.index_revisions() == before


def test_external_process_vector_update_is_seen_without_ui_notification(setup):
    reader, writer, service, index, _, cid, _ = setup
    subprocess.run([
        sys.executable, "-c",
        "import sqlite3,struct,sys; c=sqlite3.connect(sys.argv[1]); "
        "c.execute('UPDATE vectors SET embedding=? WHERE chunk_id=?',"
        "(struct.pack('<ff',0.,1.),int(sys.argv[2]))); c.commit(); c.close()",
        str(reader.db_path), str(cid),
    ], check=True, capture_output=True, timeout=15)
    assert service.search("付款")[0].semantic_score == pytest.approx(0)
    assert index.rebuilds == 2


def test_rolled_back_updates_do_not_invalidate_committed_cache(setup):
    reader, writer, service, index, _, cid, _ = setup
    before = reader.index_revisions()
    writer.conn.execute("UPDATE vectors SET embedding=? WHERE chunk_id=?",
                        (np.array([0.0, 1.0], dtype="float32").tobytes(), cid))
    assert writer.index_revisions() != before
    assert service.search("付款")[0].semantic_score == pytest.approx(1)
    writer.conn.rollback()
    assert reader.index_revisions() == writer.index_revisions() == before
    service.search("付款")
    assert index.rebuilds == 1


def test_search_inside_callers_transaction_does_not_commit_or_reuse_rolled_back_vectors(setup):
    reader, writer, service, index, _, cid, _ = setup
    before = writer.index_revisions()
    reader.conn.execute("UPDATE vectors SET embedding=? WHERE chunk_id=?",
                        (np.array([0.0, 1.0], dtype="float32").tobytes(), cid))
    uncommitted_revision = reader.index_revisions()
    assert service.search(SearchQuery("付款", scope="files"))[0].semantic_score == pytest.approx(0)
    assert reader.conn.in_transaction and writer.index_revisions() == before
    reader.conn.rollback()
    # A different write can reuse exactly the revision number from the rolled-back transaction.
    with writer.conn:
        writer.conn.execute("UPDATE vectors SET embedding=? WHERE chunk_id=?",
                            (np.array([0.6, 0.8], dtype="float32").tobytes(), cid))
    assert writer.index_revisions() == uncommitted_revision
    assert service.search(SearchQuery("付款", scope="files"))[0].semantic_score == pytest.approx(0.6)
    assert not reader.conn.in_transaction


def test_failed_search_releases_snapshot_and_invalidates_rolled_back_temp_scope(setup, monkeypatch):
    reader, writer, service, index, fid, cid, record = setup
    query = SearchQuery("付款", scope="files", path_contains="original")
    assert service.search(query)
    original = reader.fetch_chunks_by_ids

    def fail(_ids):
        raise RuntimeError("synthetic fetch error")

    monkeypatch.setattr(reader, "fetch_chunks_by_ids", fail)
    with pytest.raises(RuntimeError, match="synthetic fetch"):
        service.search(replace(query, path_contains="absent"))
    assert not reader.conn.in_transaction
    monkeypatch.setattr(reader, "fetch_chunks_by_ids", original)
    assert service.search(replace(query, path_contains="absent")) == []
    assert service.search(query)[0].file_id == fid
    assert not reader.conn.in_transaction


def test_clear_and_new_document_are_visible_without_explicit_refresh(setup, tmp_path):
    reader, writer, service, index, _, _, _ = setup
    writer.clear_index()
    assert service.search("付款") == []
    assert not service.vector_index.ids
    new_id, _, _ = document(writer, tmp_path / "new.txt", "新的付款条款")
    assert service.search("付款")[0].file_id == new_id


def test_later_file_index_can_resolve_existing_pending_attachment(setup, tmp_path):
    reader, writer, service, index, _, _, _ = setup
    path = tmp_path / "later.txt"
    ChatStore(writer).ingest(platform="wechat", account="test", conversation="合同群", source_key="sample",
                            messages=[ChatMessage("1", "附件", attachments=[Attachment(path.name, str(path))])])
    query = SearchQuery("付款", scope="chat_files", evidence="content")
    assert not service.search(query)
    fid, _, _ = document(writer, path, "后来建立索引的付款条款")
    assert service.search(query)[0].file_id == fid


def test_schema_upgrade_preserves_existing_data_and_initializes_revisions(tmp_path):
    path = tmp_path / "old.db"
    db = LocalMindDB(path)
    fid, cid, _ = document(db, tmp_path / "old.txt", "旧版本数据")
    with db.conn:
        triggers = db.conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'lm_revision_%'").fetchall()
        for row in triggers:
            db.conn.execute(f'DROP TRIGGER "{row[0]}"')
        db.conn.execute("DROP TABLE index_revisions")
    db.close()
    upgraded = LocalMindDB(path)
    try:
        assert upgraded.index_revisions() == (0, 0)
        assert upgraded.get_file(fid)["name"] == "old.txt"
        assert upgraded.fetch_chunks_by_ids([cid])[cid]["chunk_text"] == "旧版本数据"
        assert upgraded.load_vectors()[0] == [cid]
        with upgraded.conn:
            upgraded.conn.execute("UPDATE files SET name='renamed.txt' WHERE id=?", (fid,))
        assert upgraded.index_revisions() == (1, 0)
        reopened = LocalMindDB(path)
        try:
            assert reopened.index_revisions() == (1, 0)
        finally:
            reopened.close()
    finally:
        upgraded.close()


def test_revision_changes_rollback_atomically_with_cascade_deletes(setup):
    reader, writer, service, index, fid, _, _ = setup
    before = reader.index_revisions()
    writer.conn.execute("DELETE FROM files WHERE id=?", (fid,))
    assert writer.index_revisions()[0] > before[0]
    assert writer.index_revisions()[1] > before[1]
    writer.conn.rollback()
    assert reader.index_revisions() == before
    assert service.search("付款")[0].file_id == fid


def test_missing_revision_record_fails_loudly(setup):
    reader, writer, service, index, _, _, _ = setup
    with writer.conn:
        writer.conn.execute("DELETE FROM index_revisions")
    with pytest.raises(sqlite3.DatabaseError, match="Missing index revision"):
        service.search("付款")
