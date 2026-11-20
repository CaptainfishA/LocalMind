from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate, QPoint
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from localmind import chat_ui
from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.ui import MainWindow
from localmind.weflow import WeFlowClient
from localmind.weflow_batch import load_batch
from test_weflow_batch import BatchTransport


def test_one_click_batch_ui_without_picking_sessions(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db = LocalMindDB(tmp_path / "index.db")
    for key, value in {"model_cache_dir": str(tmp_path / "cache"), "privacy_accepted_v02": "1",
                       "weflow_privacy_accepted": "1", "startup_scan": "0", "monitor_enabled": "0",
                       "embedding_enabled": "0", "ocr_enabled": "0", "reranker_enabled": "0"}.items():
        db.set_setting(key, value)
    transport = BatchTransport(4)
    transport.sessions[0]["display_name"] = "一个用于校验窄窗口省略与完整提示的较长会话名称" * 2
    transport.errors["chat-1"] = 400

    def client(url, token="", **kwargs):
        value = WeFlowClient(url, token, **kwargs)
        value.opener = transport
        return value

    monkeypatch.setattr(chat_ui, "WeFlowClient", client)
    questions = []

    def consent(*args):
        questions.append(args[2])
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", consent)
    window = MainWindow(db)
    panel = window.chat_panel
    window.show()
    window.tabs.setCurrentWidget(panel)

    def wait_task():
        end = time.monotonic() + 20
        while (panel.busy or panel.scope_dialog is not None) and time.monotonic() < end:
            app.processEvents()
            if panel.scope_dialog is not None:
                panel.scope_dialog.accept()
            time.sleep(0.01)
        assert not panel.busy
        app.processEvents()

    def screenshot(name):
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        for width, height in [(1240, 800), (980, 650)]:
            window.resize(width, height)
            app.processEvents()
            assert panel.sync_button.width() >= 160
            assert panel.accounts.width() >= 155
            for widget in [panel.token, panel.sync_button, panel.cancel_button, panel.start_date]:
                right = widget.mapTo(panel.viewport(), QPoint(widget.width(), 0)).x()
                assert right <= panel.viewport().width()
            if output:
                directory = Path(output)
                directory.mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(directory / f"v05-{name}-{width}.png"))

    try:
        assert not panel.single_session.isChecked() and panel.sessions.count() == 0
        assert not panel.advanced.isVisible()
        panel.busy = True
        window.search_input.setText("项目资料")
        window.run_search()
        assert not window.search_busy and "搜索暂不可用" in window.search_status.text()
        panel.busy = False
        panel.token.setText("synthetic-batch-token")
        panel.start_date.setDate(QDate(2026, 5, 1))
        panel.sync_button.click()  # No connect button or conversation selection first.
        wait_task()
        assert len(questions) == 1 and "确认的会话" in questions[0]
        assert len(ChatStore(db).sources()) == 3
        assert panel.task_table.rowCount() == 4
        assert "部分会话失败" in panel.status.text()
        assert panel.sync_button.text() == "继续建立索引"
        assert panel.accounts.currentData() == "test-account"
        assert not panel.timer.isActive()
        screenshot("partial")
        transport.errors.clear()
        transport.calls.clear()
        panel.sync_button.click()
        wait_task()
        assert len(ChatStore(db).sources()) == 4 and panel.attachment_table.rowCount() == 4
        assert [q["talker"] for p, q in transport.calls if p == "/api/v1/messages"] == ["chat-1"]
        assert panel.confirmed_target == panel.target()
        assert panel.sync_button.text() == "更新所选索引"
        assert len(questions) == 1
        screenshot("complete")
        panel.show_all_attachments()
        screenshot("attachments")
        panel.advanced_button.setChecked(True)
        screenshot("advanced")
        panel.auto.setChecked(True)
        assert panel.timer.isActive() and panel.timer.interval() == 300000
        panel.auto_sync()
        wait_task()
        assert panel.timer.isActive() and panel.auto.isChecked()
        panel.single_session.setChecked(True)
        assert not panel.timer.isActive() and panel.sessions.isEnabled()
        panel.keyword.setText("项目会话1")
        assert panel.sessions.count() == 1
        panel.single_session.setChecked(False)
        assert not panel.sessions.isEnabled()
        panel.start_date.setDate(QDate(2026, 5, 2))
        assert panel.confirmed_target is None
        panel.token.setText("new-token")
        assert panel.sessions.count() == 0 and panel.accounts.count() == 1
        stored = "\n".join(str(row[0]) for row in db.conn.execute("SELECT value FROM settings"))
        assert "synthetic-batch-token" not in stored and "new-token" not in stored
    finally:
        panel.cancel.set()
        wait_task()
        window.close()
        app.processEvents()
        assert not panel.thread.isRunning()


def test_bulk_consent_decline_does_not_start_or_store_token(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    db = LocalMindDB(tmp_path / "index.db")
    for key in ["privacy_accepted_v02", "weflow_privacy_accepted"]:
        db.set_setting(key, "1")
    for key in ["startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"]:
        db.set_setting(key, "0")
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.No)
    window = MainWindow(db)
    panel = window.chat_panel
    try:
        panel.token.setText("synthetic-token")
        panel.sync()
        assert not panel.busy and not panel.timer.isActive()
        assert db.get_setting("weflow_bulk_privacy_accepted") != "1"
        assert not db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_batch:%'").fetchone()
    finally:
        window.close()
        app.processEvents()
