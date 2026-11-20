from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.indexer import IndexingService
from localmind.ocr import OcrService
from localmind.ui import MainWindow
from weflow_mock import MockWeFlow, message


def test_weflow_ui_mock_sync_privacy_and_screenshots(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    for font in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"]:
        QFontDatabase.addApplicationFont(font)
    db = LocalMindDB(tmp_path / "index.db")
    for key, value in {"model_cache_dir": str(tmp_path / "cache"), "privacy_accepted_v02": "1",
                       "weflow_privacy_accepted": "1", "weflow_bulk_privacy_accepted": "1", "startup_scan": "0", "monitor_enabled": "0",
                       "embedding_enabled": "0", "ocr_enabled": "0", "reranker_enabled": "0"}.items():
        db.set_setting(key, value)
    errors = []
    attachment = tmp_path / "设计方案.txt"
    attachment.write_text("星河方案的付款信息", encoding="utf-8")
    IndexingService(db, embedding_service=EmbeddingService(force_hash=True), ocr_service=OcrService(enabled=False)).index_path(attachment, include_embeddings=False)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(str(args)))
    window = MainWindow(db)
    panel = window.chat_panel
    panel.single_session.setChecked(True)
    window.show()

    def wait_task():
        end = time.monotonic() + 20
        while (panel.busy or window.search_busy) and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert not panel.busy
        assert not window.search_busy
        app.processEvents()

    try:
        with MockWeFlow([message("1", "星河方案已确认", timestamp=int(time.time()) - 100),
                         message("2", "[文件] 设计方案.txt", timestamp=int(time.time()) - 95)]) as mock:
            panel.url.setText(mock.url)
            panel.token.setText(mock.token)
            panel.start_date.setDate(QDate.currentDate().addDays(-2))
            panel.connect_button.click()
            wait_task()
            assert panel.sessions.count() == 1, panel.status.text()
            panel.sync_button.click()
            wait_task()
            assert ChatStore(db).sources()[0]["messages"] == 2, panel.status.text()
            assert panel.attachment_table.rowCount() == 1
            panel.auto.setChecked(True)
            assert panel.timer.isActive()
            panel.token.setText("changed-token")
            assert not panel.timer.isActive()
            values = [str(r[0]) for r in db.conn.execute("SELECT value FROM settings")]
            assert not any(mock.token in v or "changed-token" in v for v in values)
            window.tabs.setCurrentWidget(panel)
            for width, height in [(1240, 800), (980, 650)]:
                window.resize(width, height)
                app.processEvents()
                assert panel.sync_button.width() >= 60
                assert panel.url.width() > 100
                output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
                if output:
                    target = Path(output)
                    target.mkdir(parents=True, exist_ok=True)
                    assert window.grab().save(str(target / f"v04-weflow-{width}.png"))
            window.tabs.setCurrentIndex(0)
            window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
            window.conversation_filter.setText("模拟项目群")
            window.sender_filter.setText("张工")
            window.search_input.setText("星河方案")
            window.run_search()
            wait_task()
            assert window.results and window.results[0].result_kind == "attachment_pending"
            assert window.bind_result_button.isEnabled()
            assert all(button.isEnabled() for button in window.file_actions)
            monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(attachment), ""))
            window.bind_result_button.click()
            wait_task()
            window.run_search()
            wait_task()
            assert window.results[0].path == str(attachment)
            assert all(button.isEnabled() for button in window.file_actions)
            assert window.preview_tabs.isTabEnabled(1)
            window.preview_tabs.setCurrentIndex(1)
            assert "张工" in window.context_preview.toPlainText()
            for width, height in [(1240, 800), (980, 650)]:
                window.resize(width, height)
                app.processEvents()
                if output:
                    assert window.grab().save(str(target / f"v04-search-{width}.png"))
            assert not errors
    finally:
        panel.cancel.set()
        wait_task()
        window.close()
        app.processEvents()
        assert not panel.thread.isRunning()


def test_alpha_account_selection_and_connection_invalidation(tmp_path, monkeypatch):
    from test_weflow_alpha import alpha_client, alpha_message
    from localmind import chat_ui

    app = QApplication.instance() or QApplication([])
    db = LocalMindDB(tmp_path / "alpha.db")
    for key, value in {"privacy_accepted_v02": "1", "weflow_privacy_accepted": "1",
                       "startup_scan": "0", "monitor_enabled": "0", "embedding_enabled": "0",
                       "ocr_enabled": "0", "reranker_enabled": "0"}.items():
        db.set_setting(key, value)
    client = alpha_client()
    client.opener.rows[1] = alpha_message(1, text="[文件] 设计附件.pdf")
    monkeypatch.setattr(chat_ui, "WeFlowClient", lambda *_a, **_k: client)
    window = MainWindow(db)
    panel = window.chat_panel
    window.show()

    def wait_task():
        end = time.monotonic() + 20
        while panel.busy and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert not panel.busy
        app.processEvents()

    try:
        panel.token.setText("synthetic-token")
        panel.single_session.setChecked(True)
        panel.start_date.setDate(QDate(2026, 5, 1))
        panel.load_sessions()
        wait_task()
        assert panel.accounts.currentData() == "test-account", panel.status.text()
        assert panel.sessions.count() == 2
        panel.sync()
        wait_task()
        assert ChatStore(db).sources()[0]["messages"] == 3, panel.status.text()
        assert panel.attachment_table.rowCount() == 1
        panel.auto.setChecked(True)
        assert panel.timer.isActive()
        window.tabs.setCurrentWidget(panel)
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        for width, height in [(1240, 800), (980, 650)]:
            window.resize(width, height)
            app.processEvents()
            assert panel.accounts.width() > 100
            assert panel.sync_button.width() >= 60
            if output:
                path = Path(output)
                path.mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(path / f"v041-alpha-{width}.png"))
        panel.accounts.setCurrentIndex(0)
        assert panel.sessions.count() == 0 and not panel.timer.isActive()
        panel.load_sessions()
        wait_task()
        panel.url.setText("http://127.0.0.1:5032")
        assert panel.sessions.count() == 0 and panel.accounts.count() == 1
        assert panel.confirmed_target is None
    finally:
        panel.cancel.set()
        wait_task()
        window.close()
        app.processEvents()
        assert not panel.thread.isRunning()
