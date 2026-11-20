import os
from pathlib import Path
import sqlite3
from threading import Event
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QPoint

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.chunker import CHUNK_MAX_CHARS, TEXT_INDEX_VERSION, split_text
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService, SentenceTransformerBackend
from localmind.indexer import IndexingService
from localmind.models import TextChunk
from localmind.ocr import OcrService
from localmind.scanner import build_file_record, hash_file
from localmind.semantic_maintenance import update_wechat_semantics
from test_performance_v02 import CountingEmbedding
from test_result_lifecycle import screen  # noqa: F401


def old_file(db, path, embedding):
    text = "团队日常会议讨论进展与人员排期。" * 35 + "备用电源持续供电九十分钟。"
    path.write_text(text, encoding="utf-8")
    record = build_file_record(path)
    record.file_hash = hash_file(path)
    fid, _ = db.replace_file_index(record, [TextChunk(text, "ocr", page_no=3, start_offset=0, end_offset=len(text))],
        parse_status="parsed", ocr_status="done", embeddings=np.ones((1, 8), dtype="float32"),
        model_name=embedding.active_model_name)
    db.set_setting("embedding_model", embedding.active_model_name)
    with db.conn:
        db.conn.execute("UPDATE files SET text_index_version=0 WHERE id=?", (fid,))
    db.insert_ocr_result(fid, page_no=3, text=text, confidence=0.95, status="done")
    return fid, text


def ingest(db, message, embedding=None, key="synthetic-longtext"):
    return ChatStore(db).ingest(platform="wechat", account="synthetic", conversation="合成资料群",
        source_key=key, messages=[message], embedding_service=embedding)


def test_chunks_cover_entire_text_with_bounded_inputs():
    text = "甲乙丙丁" * 500 + "末尾的独立线索"
    spans = split_text(text)
    assert len(spans) > 1 and all(len(part) <= CHUNK_MAX_CHARS for part, _, _ in spans)
    assert spans[0][1] == 0 and spans[-1][2] == len(text)
    assert all(left[2] >= right[1] for left, right in zip(spans, spans[1:]))
    assert all(part == text[start:end] for part, start, end in spans)
    assert "末尾的独立线索" in spans[-1][0]


@pytest.mark.parametrize("options", [dict(overlap=160), dict(max_size=0), dict(min_size=500)])
def test_bad_chunk_parameters_cannot_loop_forever(options):
    with pytest.raises(ValueError):
        split_text("甲" * 1000, **options)


def test_encoder_refuses_silent_document_truncation():
    class Model:
        max_seq_length = 8
        calls = []
        def tokenizer(self, items, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [list(range(len(item) + 2)) for item in items]}
        def encode(self, items, **kwargs):
            self.calls.append(items)
            return np.ones((len(items), 8), dtype="float32")
    model = Model()
    backend = SentenceTransformerBackend(model, "synthetic")
    with pytest.raises(ValueError, match="上限"):
        backend.encode(["头部内容和尾部关键信息"])
    assert model.calls == []
    assert backend.encode(["短文"]).shape == (1, 8)


def test_upgrade_reuses_ocr_text_preserves_identity_and_is_incremental(tmp_path, monkeypatch):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        embedding = CountingEmbedding()
        path = tmp_path / "old.txt"
        fid, text = old_file(db, path, embedding)
        ingest(db, ChatMessage("one", "这是断电保障资料", attachments=[Attachment(path.name, str(path))]))
        attachment = ChatStore(db).attachments()[0]
        service = IndexingService(db, embedding_service=embedding, ocr_service=OcrService(enabled=False))
        monkeypatch.setattr(service.parser, "parse", lambda _: pytest.fail("Unchanged OCR must not be reparsed"))
        assert service.index_path(path)
        row = db.get_file(fid)
        assert row["text_index_version"] == TEXT_INDEX_VERSION and row["ocr_status"] == "done"
        chunks = db.conn.execute("SELECT * FROM chunks WHERE file_id=?", (fid,)).fetchall()
        assert len(chunks) > 1 and all(len(row["chunk_text"]) <= CHUNK_MAX_CHARS for row in chunks)
        assert all(row["page_no"] == 3 and row["source_type"] == "ocr" for row in chunks)
        assert any("九十分钟" in row["chunk_text"] for row in chunks)
        assert db.conn.execute("SELECT text FROM ocr_results WHERE file_id=?", (fid,)).fetchone()[0] == text
        assert ChatStore(db).attachments()[0]["id"] == attachment["id"]
        assert ChatStore(db).attachments()[0]["file_id"] == fid
        calls = list(embedding.batch_sizes)
        assert not service.index_path(path)
        assert embedding.batch_sizes == calls and path.read_text(encoding="utf-8") == text
    finally:
        db.close()


def test_chat_full_input_includes_neighbor_and_reply_tails_without_truncation(tmp_path):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        embedding = CountingEmbedding()
        messages = [ChatMessage("file", "日常事项和排期。" * 60 + "附件末尾线索", sender="甲" * 190,
                               sent_at="2026-10-20T00:00:00Z", attachments=[Attachment("资料.txt")]),
                    ChatMessage("reply", "此前讨论。" * 120 + "长回复末尾线索", sender="乙",
                                sent_at="2026-10-20T00:01:00Z", reply_to="file")]
        store = ChatStore(db)
        args = dict(platform="wechat", account="synthetic", conversation="合成群" * 30,
                    source_key="long", messages=messages, embedding_service=embedding)
        store.ingest(**args)
        rows = db.conn.execute("SELECT c.chunk_text,u.attachment_id FROM chunks c JOIN chat_units u ON u.chunk_id=c.id").fetchall()
        assert rows and all(len(row[0]) <= CHUNK_MAX_CHARS for row in rows)
        assert any("附件末尾线索" in row[0] and row[1] for row in rows)
        assert any("长回复末尾线索" in row[0] and row[1] for row in rows)
        assert store.ingest(**args)["updated_units"] == 0
        before = db.stats().vectors_total
        args["embedding_service"] = None
        assert store.ingest(**args)["updated_units"] == 0
        assert db.stats().vectors_total == before
    finally:
        db.close()


def test_old_schema_upgrade_is_non_destructive(tmp_path):
    path = tmp_path / "db.sqlite"
    db = LocalMindDB(path)
    fid, _ = old_file(db, tmp_path / "old.txt", CountingEmbedding())
    before = db.stats().chunks_total
    db.close()
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE files DROP COLUMN text_index_version")
    reopened = LocalMindDB(path)
    try:
        assert reopened.get_file(fid)["text_index_version"] == 0
        assert reopened.stats().chunks_total == before
    finally:
        reopened.close()


def test_local_maintenance_rebuilds_both_sources_and_reuses_second_run(tmp_path):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        embedding = CountingEmbedding()
        path = tmp_path / "old.txt"
        fid, _ = old_file(db, path, embedding)
        ingest(db, ChatMessage("file", "例会背景。" * 100 + "聊天末尾线索", attachments=[Attachment(path.name, str(path))]))
        ids = [row["id"] for row in ChatStore(db).attachments()]
        messages = list(db.conn.execute("SELECT id,body,observed_at FROM chat_messages"))
        output = update_wechat_semantics(db, embedding_service=embedding, ocr_service=OcrService(enabled=False))
        assert output["sources_done"] == 1 and output["updated_units"] > 0 and not output["issues"]
        assert output["file_summary"]["files_indexed"] == 1
        assert [row["id"] for row in ChatStore(db).attachments()] == ids
        assert [tuple(row) for row in db.conn.execute("SELECT id,body,observed_at FROM chat_messages")] == [tuple(row) for row in messages]
        repeat = update_wechat_semantics(db, embedding_service=embedding, ocr_service=OcrService(enabled=False))
        assert not repeat["changed"] and repeat["updated_units"] == 0 and repeat["file_summary"]["files_indexed"] == 0
        assert db.get_file(fid)
    finally:
        db.close()


def test_maintenance_refuses_hash_or_model_mismatch_without_erasing_vectors(tmp_path):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        old_file(db, tmp_path / "old.txt", CountingEmbedding())
        ingest(db, ChatMessage("file", "合成资料"))
        before = db.index_revisions()
        with pytest.raises(ValueError, match="未就绪"):
            update_wechat_semantics(db, embedding_service=EmbeddingService(force_hash=True), ocr_service=OcrService(enabled=False))
        embedding = CountingEmbedding()
        embedding.active_model_name = "other"
        with pytest.raises(ValueError, match="不一致"):
            update_wechat_semantics(db, embedding_service=embedding, ocr_service=OcrService(enabled=False))
        assert db.index_revisions() == before
    finally:
        db.close()


def test_maintenance_cancellation_keeps_committed_sources(tmp_path):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        ingest(db, ChatMessage("one", "第一条消息"), key="one")
        ingest(db, ChatMessage("two", "第二条消息"), key="two")
        embedding = CountingEmbedding()
        cancelled = False
        def progress(message):
            nonlocal cancelled
            if message == "更新本地会话 2/2":
                cancelled = True
        output = update_wechat_semantics(db, embedding_service=embedding, ocr_service=OcrService(enabled=False),
                                        should_cancel=lambda: cancelled, progress=progress)
        assert output["cancelled"] and output["sources_done"] == 1
        assert db.stats().vectors_total == 1
        assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 2
    finally:
        db.close()


def test_local_maintenance_ui_confirmation_and_no_token_payload(screen, monkeypatch):
    app, window, *_ = screen
    panel = window.chat_panel
    window.resize(980, 650)
    window.tabs.setCurrentIndex(1)
    panel.data_tabs.setCurrentIndex(1)
    window.show()
    app.processEvents()
    button = panel.semantic_update_button
    bottom = button.mapTo(panel.viewport(), QPoint(button.width(), button.height()))
    assert 0 < bottom.y() <= panel.viewport().height()
    assert 0 < bottom.x() <= panel.viewport().width()
    payloads = []
    panel.requested.disconnect(panel.worker.process)
    panel.requested.connect(payloads.append)
    panel.token.setText("synthetic-do-not-forward")
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.No)
    panel.update_semantic_index()
    assert not payloads
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    panel.update_semantic_index()
    app.processEvents()
    assert payloads[0]["kind"] == "semantic_update" and payloads[0]["token"] == payloads[0]["url"] == ""
    assert panel.busy and not panel.semantic_update_button.isEnabled() and panel.cancel_button.isEnabled()
    panel.finished(dict(kind="semantic_update", seconds=0.1, sources_done=1, sources_total=1, updated_units=3))
    assert not panel.busy and panel.semantic_update_button.isEnabled()
    assert "本地语义索引已更新" in panel.status.text()


def test_existing_long_chat_windows_upgrade_without_reimport(tmp_path):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        embedding = CountingEmbedding()
        message = ChatMessage("one", "旧版例会背景。" * 75 + "聊天尾部独立线索", attachments=[Attachment("资料.txt")])
        with patch("localmind.chats.split_text", lambda text, **kwargs:
                   split_text(text, **kwargs) if kwargs else [(text, 0, len(text))]):
            ingest(db, message, embedding)
        old = [tuple(row) for row in db.conn.execute("SELECT id,body,observed_at FROM chat_messages")]
        attachment_id = ChatStore(db).attachments()[0]["id"]
        assert db.conn.execute("SELECT max(length(chunk_text)) FROM chunks").fetchone()[0] > CHUNK_MAX_CHARS
        output = update_wechat_semantics(db, embedding_service=embedding, ocr_service=OcrService(enabled=False))
        assert output["updated_units"] > 0 and not output["issues"]
        assert db.conn.execute("SELECT max(length(chunk_text)) FROM chunks").fetchone()[0] <= CHUNK_MAX_CHARS
        assert [tuple(row) for row in db.conn.execute("SELECT id,body,observed_at FROM chat_messages")] == old
        assert ChatStore(db).attachments()[0]["id"] == attachment_id
        assert db.conn.execute("SELECT count(*) FROM chunks c LEFT JOIN chat_units u ON u.chunk_id=c.id WHERE u.chunk_id IS NULL").fetchone()[0] == 0
    finally:
        db.close()


def test_failed_embedding_keeps_old_file_and_chat_vectors(tmp_path, monkeypatch):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        embedding = CountingEmbedding()
        path = tmp_path / "old.txt"
        fid, _ = old_file(db, path, embedding)
        ingest(db, ChatMessage("one", "已保存的聊天", attachments=[Attachment(path.name, str(path))]), embedding)
        before = [tuple(row) for row in db.conn.execute("SELECT * FROM vectors ORDER BY chunk_id")]
        def fail(_texts):
            raise ValueError("synthetic encoder failure")
        monkeypatch.setattr(embedding, "encode_documents", fail)
        output = update_wechat_semantics(db, embedding_service=embedding, ocr_service=OcrService(enabled=False))
        assert output["issues"] and output["file_summary"]["errors"] == 1
        assert db.get_file(fid)["text_index_version"] == 0
        assert [tuple(row) for row in db.conn.execute("SELECT * FROM vectors ORDER BY chunk_id")] == before
    finally:
        db.close()


def test_missing_file_reports_partial_and_keeps_original_index(tmp_path):
    db = LocalMindDB(tmp_path / "db.sqlite")
    try:
        embedding = CountingEmbedding()
        path = tmp_path / "old.txt"
        fid, _ = old_file(db, path, embedding)
        ingest(db, ChatMessage("one", "合成消息", attachments=[Attachment(path.name, str(path))]), embedding)
        path.unlink()
        output = update_wechat_semantics(db, embedding_service=embedding, ocr_service=OcrService(enabled=False))
        assert output["issues"] and "不可读取" in output["issues"][0]
        assert output["sources_done"] == 1 and db.get_file(fid)["text_index_version"] == 0
        assert db.conn.execute("SELECT count(*) FROM chunks WHERE file_id=?", (fid,)).fetchone()[0] == 1
    finally:
        db.close()


def test_worker_local_maintenance_never_connects_alpha(tmp_path, monkeypatch):
    from localmind.chat_ui import WeFlowWorker
    path = tmp_path / "db.sqlite"
    db = LocalMindDB(path)
    ingest(db, ChatMessage("one", "合成消息"))
    db.close()
    configs = []
    embedding = CountingEmbedding()
    embedding.load_source = "local_cache"
    embedding.device = "cpu"
    embedding.runtime_note = ""
    def factory(**kwargs):
        configs.append(kwargs)
        return embedding
    monkeypatch.setattr("localmind.chat_ui.EmbeddingService", factory)
    monkeypatch.setattr("localmind.chat_ui.WeFlowClient", lambda *_args, **_kwargs: pytest.fail("Must not connect Alpha"))
    worker = WeFlowWorker(str(path), Event())
    results = []
    worker.finished.connect(results.append)
    worker.process(dict(kind="semantic_update", cache=str(tmp_path / "cache"), device="cpu", embeddings=False,
                        ocr_enabled=False, token="", url=""))
    assert configs[0]["local_files_only"] is True
    assert len(results) == 1 and "error" not in results[0] and results[0]["sources_done"] == 1
