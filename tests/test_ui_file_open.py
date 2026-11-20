import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.models import SearchQuery
from localmind.ui import MainWindow
from test_weflow_v04 import searcher


def test_pending_attachment_opens_without_full_index_and_reuses_path(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db = LocalMindDB(tmp_path / "index.db")
    for key in ("privacy_accepted_v02", "weflow_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    root = tmp_path / "wechat"
    root.mkdir()
    document = root / "项目清单.txt"
    document.write_text("测试文档", encoding="utf-8")
    db.set_setting("wechat_file_root", str(root))
    ChatStore(db).ingest(platform="wechat", account="test", conversation="项目群", source_key="test",
        messages=[ChatMessage("1", "这是项目清单", attachments=[Attachment(document.name)])])
    window = MainWindow(db)
    opened, errors, confirmations = [], [], []
    monkeypatch.setattr(os, "startfile", lambda path: opened.append(path))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    monkeypatch.setattr(QMessageBox, "exec", lambda box: errors.append(box.text()) or QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: (confirmations.append(args[3]) or str(document), True))
    window.show()

    def wait():
        deadline = time.monotonic() + 15
        while (window.chat_panel.busy or window.search_busy) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        assert not window.chat_panel.busy
        assert not window.search_busy

    try:
        service = searcher(db, mode="keyword")
        results = service.search(SearchQuery("项目清单", scope="chat_files"))
        window._search_finished(results, service.last_diagnostics, {"total_ms": 12, "index_refresh_ms": 0})
        assert results[0].result_kind == "attachment_pending"
        assert window.file_preview.isHidden() and "待定位" in window.detail_meta.text()
        assert window.file_actions[0].text() == "定位并打开"
        assert not window.source_history_button.isEnabled()
        assert all(button.isEnabled() for button in window.file_actions)
        window.file_actions[0].click()
        wait()
        assert opened == [str(document)] and len(confirmations) == 1
        assert not window.file_preview.isHidden() and "已记录路径" in window.detail_meta.text()
        assert window.file_actions[0].text() == "打开文件"
        assert db.get_file_by_path(str(document)) is None
        assert str(document) in window.info_preview.toPlainText()
        window.copy_selected_path()
        wait()
        assert QApplication.clipboard().text() == str(document)
        assert len(confirmations) == 1
        assert window.index_result_button.isEnabled()
        window.index_result_button.click()
        wait()
        assert db.get_file_by_path(str(document)) is not None
        assert window.result_notice.isVisible()
        assert "关联已变化" in window.info_preview.toPlainText()
        assert window.detail_warning.isVisible() and "关联已变化" in window.detail_warning.text()
        assert not window.index_result_button.isEnabled()
        assert not window.source_copy_button.isEnabled()
        assert window.chat_panel.attachment_table.item(0, 4).text() == "关键词正文"
        assert db.conn.execute("SELECT COUNT(*) FROM watched_folders").fetchone()[0] == 0
        window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
        window.search_input.setText("测试文档")
        window.run_search()
        wait()
        assert window.results and window.results[0].path == str(document)
        assert window.results[0].source_type == "txt"
        assert window.source_history_button.isEnabled()
        from localmind.source_history_ui import SourceHistoryDialog
        histories = []
        def inspect_history(dialog):
            histories.append((dialog.file_id, len(dialog.rows), dialog.preview.toPlainText()))
            return 0
        monkeypatch.setattr(SourceHistoryDialog, "exec", inspect_history)
        selected_id = window.results[0].attachment_id
        window.source_history_button.click()
        assert histories[0][0] == window.results[0].file_id and histories[0][1] == 1
        assert "项目群" in histories[0][2] and window.results[0].attachment_id == selected_id
        assert "项目群" in window.results[0].chat_context
        assert "测试文档" in window.preview.toPlainText() and "路径：" not in window.preview.toPlainText()
        assert "路径：" in window.info_preview.toPlainText()
        assert window.preview_tabs.isTabEnabled(2)
        assert window.evidence_filter.isVisible()
        window.evidence_filter.setCurrentIndex(window.evidence_filter.findData("content"))
        window.run_search()
        wait()
        assert window.results[0].source_type == "txt"
        assert db.get_setting("chat_search_evidence") == "content"
        window.preview_tabs.setCurrentIndex(1)
        assert "项目群" in window.context_preview.toPlainText()
        window.source_copy_button.click()
        assert QApplication.clipboard().text() == window.context_preview.toPlainText()
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            window.grab().save(str(Path(output) / "chat-source-detail.png"))
        window.evidence_filter.setCurrentIndex(window.evidence_filter.findData("chat"))
        window.search_input.setText("项目清单")
        window.run_search()
        wait()
        assert window.results[0].source_type == "chat"
        window.search_input.setText("测试文档")
        window.run_search()
        wait()
        assert not window.results and not window.context_preview.toPlainText()
        assert not window.source_copy_button.isEnabled()
        assert not window.source_history_button.isEnabled()
        assert not window.preview_tabs.isTabEnabled(1)
        window.evidence_filter.setCurrentIndex(window.evidence_filter.findData("content"))
        window.run_search()
        wait()
        assert window.results[0].source_type == "txt"
        window.preview_tabs.setCurrentIndex(0)
        service = window.search_worker.service
        window.retrieval_buttons["semantic"].click()
        window.run_search()
        wait()
        assert not window.results and "无语义结果" in window.search_status.text()
        window.retrieval_buttons["keyword"].click()
        window.run_search()
        wait()
        assert window.results and window.search_worker.service is service
        assert db.get_setting("search_retrieval_mode") == "keyword"
        assert window.result_table.horizontalHeaderItem(3).text() == "排序分"
        assert "正确概率" in window.result_table.horizontalHeaderItem(3).toolTip()
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                window.grab().save(str(Path(output) / f"file-open-{width}.png"))
        document.write_text("已改变的测试文档正文", encoding="utf-8")
        window.run_search()
        wait()
        assert window.results[0].file_state == "changed"
        assert "索引待核对 1 项" in window.search_status.text()
        assert "文件已变化" in window.detail_warning.text()
        assert not window.result_table.item(0, 0).icon().isNull()
        assert "文件已变化" in window.result_table.item(0, 0).toolTip()
        assert window.detail_warning.isVisible() and "文件已变化" in window.detail_warning.text()
        assert "file_check:" in window.search_status.toolTip()
        assert "embedding:" in window.search_status.toolTip()
        assert "数据库读取区间：" in window.search_status.toolTip()
        window.scope_filter.setCurrentIndex(window.scope_filter.findData("all"))
        assert not window.evidence_filter.isVisible()
        window.run_search()
        wait()
        assert window.search_worker.service.last_diagnostics.evidence == "all"
        window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
        assert window.evidence_filter.currentData() == "content"
        window.resize(980, 650)
        app.processEvents()
        if output:
            window.grab().save(str(Path(output) / "file-state-warning-980.png"))
        window.tabs.setCurrentWidget(window.chat_panel)
        window.chat_panel.advanced_button.setChecked(True)
        window.chat_panel.attachment_table.selectRow(0)
        app.processEvents()
        if output:
            window.grab().save(str(Path(output) / "file-open-settings.png"))
        assert not errors
        monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
        window.clear_index()
        app.processEvents()
        assert not window.results
        assert not window.preview.toPlainText() and not window.context_preview.toPlainText()
        assert not window.info_preview.toPlainText() and not window.detail_warning.isVisible()
        assert not window.preview_tabs.isTabEnabled(1) and not window.source_copy_button.isEnabled()
        assert not window.source_history_button.isEnabled()
        assert document.read_text(encoding="utf-8") == "已改变的测试文档正文"
    finally:
        window.chat_panel.cancel.set()
        wait()
        window.close()
        app.processEvents()
        assert not window.chat_panel.thread.isRunning()


def test_open_errors_and_risky_types_are_not_silent(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    db = LocalMindDB(tmp_path / "index.db")
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    db.set_setting("startup_scan", "0")
    db.set_setting("monitor_enabled", "0")
    window = MainWindow(db)
    errors, opened = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    monkeypatch.setattr(os, "startfile", lambda path: opened.append(path))
    try:
        window.perform_file_action(str(tmp_path / "missing.txt"), "open")
        assert errors and not opened
        file = tmp_path / "test.cmd"
        file.write_text("echo test")
        window.perform_file_action(str(file), "open")
        assert not opened
    finally:
        window.close()
        app.processEvents()
