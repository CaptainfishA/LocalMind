from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate, QPoint
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from localmind import chat_ui
from localmind.db import LocalMindDB
from localmind.ui import MainWindow
from localmind.weflow import WeFlowClient
from test_weflow_batch import BatchTransport
from test_weflow_alpha import alpha_message


def test_partial_payload_visible_and_retryable(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db = LocalMindDB(tmp_path / "index.db")
    for key in ("privacy_accepted_v02", "weflow_bulk_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    transport = BatchTransport(3)
    transport.rows["chat-0"].append(alpha_message(99, conversation_id="chat-0", text={"bad": True}))
    transport.sessions[0]["display_name"] = "蓝桥项目设计与交付资料讨论群"

    def client(url, token="", **kwargs):
        value = WeFlowClient(url, token, **kwargs)
        value.opener = transport
        return value

    monkeypatch.setattr(chat_ui, "WeFlowClient", client)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    window = MainWindow(db)
    panel = window.chat_panel

    def wait():
        end = time.monotonic() + 20
        while (panel.busy or panel.scope_dialog is not None) and time.monotonic() < end:
            app.processEvents()
            if panel.scope_dialog is not None:
                panel.scope_dialog.accept()
            time.sleep(0.01)
        assert not panel.busy
        app.processEvents()

    try:
        window.show()
        window.tabs.setCurrentWidget(panel)
        panel.token.setText("synthetic-token")
        panel.start_date.setDate(QDate(2026, 5, 1))
        panel.sync()
        wait()
        assert panel.task_table.item(0, 1).text() == "部分完成"
        assert panel.task_table.item(0, 2).text() == "2 / 1"
        assert "正文不是文本" in panel.task_table.item(0, 4).toolTip()
        assert "待重试消息 1" in panel.batch_counts.text()
        assert panel.sync_button.text() == "继续建立索引" and panel.confirmed_target is None
        assert not panel.auto.isChecked()
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            for widget in (panel.sync_button, panel.batch_counts, panel.snapshot_time, panel.task_table):
                assert widget.mapTo(panel.viewport(), QPoint(widget.width(), 0)).x() <= panel.viewport().width()
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                directory = Path(output)
                directory.mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(directory / f"partial-retry-{width}.png"))
        transport.rows["chat-0"][1]["text"] = "修复后的正常消息"
        panel.sync()
        wait()
        assert panel.task_table.item(0, 1).text() == "已索引"
        assert "待重试消息 0" in panel.batch_counts.text()
        assert panel.confirmed_target == panel.target()
    finally:
        panel.cancel.set()
        wait()
        window.close()
        app.processEvents()
        assert not panel.thread.isRunning()
