import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.desktop.detail_text import evidence_sources, evidence_text
from localmind.models import SearchQuery
from localmind.reranker import RerankerService
from localmind.searcher import SearchService
from localmind.ui import MainWindow
from test_clause_semantics import ClauseEmbedding
from test_result_lifecycle import wait_for
from test_search_snapshot import FixedQuery, document


QUERY = "聊天里说杭州轮岗交接时要用的材料；聊天里说广州客户验收时讨论的安排"


@pytest.fixture
def evidence_corpus(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    path = tmp_path / "共享资料.txt"
    fid, _, _ = document(db, path, "机房备用电源的操作说明", (.1, 0, 0, .99))
    for key, group, body, vector, date in (
        ("first", "杭州交接群", "杭州轮岗交接时按这份资料安排", (.2, .97, .1, 0), "2026-10-20T00:00:00Z"),
        ("second", "广州验收群", "广州客户验收时讨论这份操作材料", (.2, .1, .97, 0), "2026-10-21T00:00:00Z"),
        ("newer", "日常行政群", "这份只作一般存档", (.99, .01, .01, 0), "2026-10-22T00:00:00Z"),
    ):
        ChatStore(db).ingest(platform="wechat", account="合成账号", conversation=group, source_key=key,
            messages=[ChatMessage(key, body, sender="联系人" + key, sent_at=date,
                                  attachments=[Attachment(path.name, str(path))])])
        cid = db.conn.execute("""SELECT u.chunk_id FROM chat_units u JOIN chat_messages m ON m.id=u.message_id
            JOIN chat_sources s ON s.id=m.source_id WHERE s.source_key=?""", (key,)).fetchone()[0]
        db.insert_vectors([cid], np.array([vector], dtype="float32"), model_name=FixedQuery.active_model_name)
    service = SearchService(db, embedding_service=ClauseEmbedding(db), reranker_service=RerankerService(enabled=False))
    results = service.search(SearchQuery(QUERY, scope="chat_files"))
    assert results[0].file_id == fid
    try:
        yield db, service, results, path
    finally:
        db.close()


def test_each_semantic_fragment_retains_its_own_chat_source(evidence_corpus):
    _, _, results, _ = evidence_corpus
    parts = [part for part in results[0].supporting_evidence if part.source_type == "chat" and part.query_clause]
    assert len(parts) == 2
    assert {part.chat_source.conversation for part in parts} == {"杭州交接群", "广州验收群"}
    assert len({part.chat_source.message_id for part in parts}) == 2
    assert all(part.chat_source.attachment_id for part in parts)
    assert {part.chat_source.source_key for part in parts} == {"first", "second"}
    assert all(part.chat_source.account == "合成账号" for part in parts)
    assert results[0].chat_source == parts[0].chat_source
    assert "日常行政群" not in evidence_text(results[0])[0]
    assert "杭州交接群" in evidence_text(results[0])[0]
    assert "广州验收群" in evidence_text(results[0])[0]
    assert "关联消息：" in evidence_text(results[0], expanded=True)[0]


def test_identical_snippets_from_different_sources_are_not_collapsed(evidence_corpus):
    _, _, results, _ = evidence_corpus
    parts = [replace(part, snippet="同一句上下文", query_clause="") for part in results[0].supporting_evidence]
    result = replace(results[0], supporting_evidence=tuple(parts + [parts[0]]))
    text, _ = evidence_text(result)
    assert text.count("同一句上下文") == 2
    assert {source.conversation for source in evidence_sources(result)} == {"杭州交接群", "广州验收群"}


@pytest.fixture
def evidence_screen(evidence_corpus):
    db, service, results, path = evidence_corpus
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(path.parent / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    window = MainWindow(db)
    window.search_input.setText(QUERY)
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    window._search_finished(results, service.last_diagnostics, {"total_ms": 20, "index_refresh_ms": 0})
    window.show()
    window.preview_tabs.setCurrentIndex(1)
    app.processEvents()
    try:
        yield app, window, db, results[0], path
    finally:
        wait_for(app, lambda: not window.search_busy and not window.chat_panel.busy)
        window.close()
        app.processEvents()


def test_source_switch_changes_only_context_not_file_actions(evidence_screen, monkeypatch):
    app, window, _, original, path = evidence_screen
    assert window.source_selector.count() == 2
    assert window.chat_timeline.isVisible()
    window.context_expand.click()
    assert window.source_selector.isVisible()
    assert "杭州轮岗交接" in window.context_preview.toPlainText()
    window.source_selector.setCurrentIndex(1)
    assert "广州客户验收" in window.context_preview.toPlainText()
    assert "杭州轮岗交接" not in window.context_preview.toPlainText()
    window.copy_chat_context()
    assert QApplication.clipboard().text() == window.context_preview.toPlainText()
    assert window.source_selector.currentIndex() == 1
    window.context_expand.click()
    window.search_index_dirty.emit()
    assert window.source_selector.currentIndex() == 1
    assert "广州客户验收" in window.context_preview.toPlainText()
    opened, bound, indexed = [], [], []
    monkeypatch.setattr(window, "perform_file_action", lambda *args: opened.append(args))
    monkeypatch.setattr(window.chat_panel, "bind_attachment", bound.append)
    monkeypatch.setattr(window.chat_panel, "index_attachment_contents", indexed.append)
    for action in ("open", "folder", "copy"):
        window.selected_file_action(action)
    window.bind_selected_attachment()
    window.index_selected_attachment()
    assert opened == [(str(path), action) for action in ("open", "folder", "copy")]
    assert bound == [original.attachment_id]
    assert indexed == [[original.attachment_id]]
    assert window.selected_result() is original
    assert original.chat_source.source_key == "first"
    wait_for(app, lambda: window.file_preview._payload is not None)
    assert window.file_preview._target[0] == str(path)


@pytest.mark.parametrize("operation", ["remove", "rebind", "source_key", "external_id", "account", "name", "origin", "filter"])
def test_unannounced_secondary_source_change_cannot_copy_wrong_context(evidence_screen, operation):
    _, window, db, original, path = evidence_screen
    window.source_selector.setCurrentIndex(1)
    source = window.source_selector.currentData()
    if operation == "remove":
        sid = db.conn.execute("SELECT id FROM chat_sources WHERE source_key=?", (source.source_key,)).fetchone()[0]
        ChatStore(db).remove_source(sid)
    elif operation == "rebind":
        fid, _, _ = document(db, path.parent / "另一份.txt", "不应显示的内容", (1, 0, 0, 0))
        ChatStore(db).bind(source.attachment_id, fid)
    else:
        if operation == "filter":
            original.source_query.sender = "联系人"
            table, column, key, value = "chat_messages", "sender", "id", source.message_id
        elif operation in {"source_key", "account"}:
            table, column, key, value = "chat_sources", operation, "source_key", source.source_key
        elif operation in {"name", "origin"}:
            table, column, key, value = "chat_attachments", operation, "id", source.attachment_id
        else:
            table, column, key, value = "chat_messages", operation, "id", source.message_id
        with db.conn:
            db.conn.execute(f"UPDATE {table} SET {column}=? WHERE {key}=?", ("不应显示的新值", value))
    QApplication.clipboard().setText("原来的剪贴板")
    window.copy_chat_context()
    assert QApplication.clipboard().text() == "原来的剪贴板"
    assert not window.context_preview.toPlainText()
    assert not window.source_copy_button.isEnabled()
    assert "所选聊天线索" in window.detail_warning.text()
    assert window.source_selector.currentData() == source
    assert window.preview_tabs.isTabEnabled(1)
    assert window.selected_result() is original and original.path == str(path)
    assert window.file_actions[0].isEnabled()
    window.source_selector.setCurrentIndex(0)
    assert "杭州轮岗交接" in window.context_preview.toPlainText()
    assert window.source_copy_button.isEnabled()


def test_primary_identity_is_validated_even_when_row_ids_and_file_are_unchanged(evidence_screen):
    _, window, db, original, _ = evidence_screen
    with db.conn:
        db.conn.execute("UPDATE chat_messages SET external_id='another-message' WHERE id=?", (original.chat_message_id,))
    window.show_selected_result()
    assert not window.context_preview.toPlainText()
    assert not window.index_result_button.isEnabled()
    assert not window.bind_result_button.isEnabled()
    assert window.file_actions[0].isEnabled()
    assert window.result_notice.isVisible()
    window.source_selector.setCurrentIndex(1)
    assert "广州客户验收" in window.context_preview.toPlainText()
    assert not window.index_result_button.isEnabled()


def test_source_change_between_metadata_and_context_reads_is_rejected(evidence_screen, monkeypatch):
    _, window, db, _, _ = evidence_screen
    window.source_selector.setCurrentIndex(1)
    source = window.source_selector.currentData()
    original = ChatStore.attachments

    def changed_after_read(store, *args, **kwargs):
        rows = original(store, *args, **kwargs)
        if kwargs.get("attachment_id") == source.attachment_id:
            with db.conn:
                db.conn.execute("UPDATE chat_messages SET external_id='reused-id' WHERE id=?", (source.message_id,))
        return rows

    monkeypatch.setattr(ChatStore, "attachments", changed_after_read)
    QApplication.clipboard().setText("保持不变")
    window.copy_chat_context()
    assert QApplication.clipboard().text() == "保持不变"
    assert not window.context_preview.toPlainText()
    assert not window.source_copy_button.isEnabled()


def test_source_history_starts_from_selected_source_without_mutating_target(evidence_screen, monkeypatch):
    import localmind.source_history_ui as history

    _, window, _, original, _ = evidence_screen
    received = []

    class Dialog:
        def __init__(self, _store, result, _parent):
            received.append(result)

        def exec(self):
            pass

        def deleteLater(self):
            pass

    monkeypatch.setattr(history, "SourceHistoryDialog", Dialog)
    window.source_selector.setCurrentIndex(1)
    selected = window.source_selector.currentData()
    window.source_history_button.click()
    assert received[0].chat_source == selected
    assert received[0].attachment_id == selected.attachment_id
    assert received[0].source_query is original.source_query
    assert received[0].path == original.path and received[0].file_id == original.file_id
    assert window.selected_result() is original
    assert original.chat_source.source_key == "first"


def test_source_selection_clear_and_layout(evidence_screen):
    app, window, _, original, _ = evidence_screen
    wait_for(app, lambda: window.file_preview._payload is not None)
    window.source_selector.setCurrentIndex(1)
    window.context_expand.click()
    for width, height in ((980, 650), (1240, 800)):
        window.resize(width, height)
        app.processEvents()
        for widget in (window.source_selector, window.context_preview, window.file_preview, window.file_actions[0]):
            corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
            assert corner.x() <= width and corner.y() <= height
        assert window.context_preview.height() >= 40
        assert window.source_selector.geometry().bottom() < window.context_preview.geometry().top()
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(Path(output) / f"evidence-sources-{width}.png"))
            window.preview_tabs.setCurrentIndex(0)
            app.processEvents()
            assert window.grab().save(str(Path(output) / f"evidence-labels-{width}.png"))
            window.preview_tabs.setCurrentIndex(1)
    window._clear_result_details()
    assert window.source_selector.count() == 0 and not window.source_selector.isVisible()
    assert not window.source_selector.toolTip()
    assert not window.context_preview.toPlainText()
    window.show_selected_result()
    assert window.source_selector.currentIndex() == 0
    assert window.selected_result() is original


def test_long_untrusted_source_labels_are_plain_text_and_do_not_resize_layout(evidence_screen):
    app, window, _, original, _ = evidence_screen
    label = '<b>研究资料</b>\n' + '很长的会话名称' * 40
    second = replace(original.supporting_evidence[1], chat_source=replace(
        original.supporting_evidence[1].chat_source, conversation=label))
    original.supporting_evidence = (original.supporting_evidence[0], second)
    window.resize(980, 650)
    window.show_selected_result()
    window.context_expand.click()
    window.source_selector.setCurrentIndex(1)
    app.processEvents()
    assert "\n" not in window.source_selector.currentText()
    assert window.width() == 980
    assert window.source_selector.width() < 600
    assert "<b>研究资料</b>" in window.preview.toPlainText()
    assert "…" in window.preview.toPlainText()
