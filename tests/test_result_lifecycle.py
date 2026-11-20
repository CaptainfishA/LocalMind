from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.models import SearchQuery
from localmind.searcher import SearchService
from localmind.ui import MainWindow
from test_weflow_v04 import index_file, ingest, searcher


def wait_for(app, predicate):
    deadline = time.monotonic() + 15
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


@pytest.fixture
def screen(tmp_path):
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db = LocalMindDB(tmp_path / "index.db")
    for key in ("privacy_accepted_v02", "weflow_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("search_retrieval_mode", "keyword")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    first = index_file(db, tmp_path / "原始文档.txt", "验收合同的付款节点")
    other = index_file(db, tmp_path / "其他文档.txt", "技术方案")
    ingest(db, [ChatMessage("1", "客户确认的验收合同", sender="测试客户",
                           attachments=[Attachment("原始文档.txt")])])
    attachment = ChatStore(db).attachments()[0]["id"]
    ChatStore(db).bind(attachment, first["id"])
    window = MainWindow(db)
    window.search_input.setText("验收合同")
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    service = searcher(db, mode="keyword")
    results = service.search(SearchQuery("验收合同", scope="chat_files", conversation="模拟项目群"))
    window._search_finished(results, service.last_diagnostics, {"total_ms": 10, "index_refresh_ms": 0})
    window.show()
    app.processEvents()
    assert window.selected_result().file_id == first["id"]
    try:
        yield app, window, db, first, other, attachment
    finally:
        wait_for(app, lambda: not window.search_busy and not window.chat_panel.busy)
        window.close()
        app.processEvents()


@pytest.mark.parametrize("operation", ["rebind", "remove", "filter"])
def test_stale_source_cannot_retarget_file_or_attachment_actions(screen, monkeypatch, operation):
    app, window, db, first, other, attachment = screen
    original = window.selected_result()
    if operation == "rebind":
        ChatStore(db).bind(attachment, other["id"])
    elif operation == "remove":
        ChatStore(db).remove_source(ChatStore(db).sources()[0]["id"])
    else:
        with db.conn:
            db.conn.execute("UPDATE chat_sources SET conversation='另一个群'")
    window.search_index_dirty.emit()
    app.processEvents()
    opened, indexed, bound, located = [], [], [], []
    monkeypatch.setattr(window, "perform_file_action", lambda *args: opened.append(args))
    monkeypatch.setattr(window.chat_panel, "index_attachment_contents", indexed.append)
    monkeypatch.setattr(window.chat_panel, "bind_attachment", bound.append)
    monkeypatch.setattr(window.chat_panel, "locate_attachment", lambda *args: located.append(args))
    QApplication.clipboard().setText("保持剪贴板不变")
    window.copy_chat_context()
    window.index_selected_attachment()
    window.bind_selected_attachment()
    for action in ("open", "folder", "copy"):
        window.selected_file_action(action)
    assert not indexed and not bound and not located
    assert opened == [(first["path"], action) for action in ("open", "folder", "copy")]
    assert QApplication.clipboard().text() == "保持剪贴板不变"
    assert not window.context_preview.toPlainText()
    assert not window.source_copy_button.isEnabled()
    assert not window.bind_result_button.isEnabled()
    assert not window.index_result_button.isEnabled()
    assert window.result_notice.isVisible()
    assert original is window.selected_result()
    assert original.path == first["path"] and original.chat_context
    assert other["path"] not in window.preview.toPlainText()
    assert other["path"] not in window.info_preview.toPlainText()
    assert window.detail_warning.isVisible() and "上次搜索快照" in window.detail_warning.text()
    assert "上次搜索快照" in window.info_preview.toPlainText()
    for width, height in ((980, 650), (1240, 800)):
        window.resize(width, height)
        app.processEvents()
        for widget in (window.result_notice, window.refresh_results_button, window.preview, window.search_button):
            corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
            assert corner.x() <= width and corner.y() <= height
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output and operation == "rebind":
            Path(output).mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(Path(output) / f"stale-result-{width}.png"))


def test_unannounced_rebind_is_checked_at_action_time(screen, monkeypatch):
    _, window, db, first, other, attachment = screen
    ChatStore(db).bind(attachment, other["id"])
    called = []
    monkeypatch.setattr(window.chat_panel, "index_attachment_contents", called.append)
    window.index_selected_attachment()
    assert not called and window.result_notice.isVisible()
    assert window.selected_result().path == first["path"]
    assert window._index_revision > 0


def test_valid_source_actions_still_work_after_unrelated_index_update(screen, monkeypatch):
    _, window, _, _, _, attachment = screen
    window.search_index_dirty.emit()
    indexed, bound = [], []
    monkeypatch.setattr(window.chat_panel, "index_attachment_contents", indexed.append)
    monkeypatch.setattr(window.chat_panel, "bind_attachment", bound.append)
    window.index_selected_attachment()
    window.bind_selected_attachment()
    window.copy_chat_context()
    assert indexed == [[attachment]] and bound == [attachment]
    assert QApplication.clipboard().text() == window.context_preview.toPlainText()
    assert window.index_result_button.isEnabled() and window.source_copy_button.isEnabled()


@pytest.mark.parametrize("fail", [False, True])
def test_inflight_results_from_previous_revision_are_discarded_and_service_reused(screen, monkeypatch, fail):
    app, window, _, _, _, _ = screen
    entered, release = Event(), Event()
    original_search = SearchService.search
    errors = []
    old_results = window.results

    def blocked(service, query):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("test worker wait timed out")
        if fail:
            raise RuntimeError("obsolete failure")
        return original_search(service, query)

    monkeypatch.setattr(SearchService, "search", blocked)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    window.run_search()
    try:
        wait_for(app, entered.is_set)
        assert not window.refresh_results_button.isEnabled()
        window.search_index_dirty.emit()
    finally:
        release.set()
    wait_for(app, lambda: not window.search_busy)
    assert not errors
    assert window.results is old_results
    assert "本次结果未采用" in window.search_status.text()
    assert window.result_notice.isVisible()
    service = window.search_worker.service
    assert service is not None
    monkeypatch.setattr(SearchService, "search", original_search)
    window.refresh_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert window.results is not old_results
    assert window.search_worker.service is service
    assert not window.result_notice.isVisible()
    assert not window.search_worker.index_dirty
    assert "耗时" in window.search_status.text()


def test_empty_result_refresh_and_clear_reset_notice(screen, monkeypatch):
    app, window, _, first, _, _ = screen
    window.search_input.setText("绝无匹配的火星文")
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert not window.results
    window.search_index_dirty.emit()
    assert window.result_notice.isVisible()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.clear_index()
    assert not window.results and not window.result_notice.isVisible()
    assert not window.context_preview.toPlainText()
    assert not window.source_copy_button.isEnabled()
    assert not window.index_result_button.isEnabled()
    assert window.search_status.text() == "索引已清空"
    assert Path(first["path"]).read_text(encoding="utf-8") == "验收合同的付款节点"


def test_completed_snapshot_is_marked_stale_when_database_changed_before_delivery(screen):
    _, window, db, first, other, attachment = screen
    service = searcher(db, mode="keyword")
    results = service.search(SearchQuery("验收合同", scope="chat_files"))
    diagnostics = service.last_diagnostics
    assert diagnostics.index_current
    ChatStore(db).bind(attachment, other["id"])
    window._search_finished(results, diagnostics, {"total_ms": 10, "index_refresh_ms": 0})
    assert window.result_notice.isVisible()
    assert window.selected_result().path == first["path"]
    assert not window.index_result_button.isEnabled()
    assert not window.context_preview.toPlainText()
