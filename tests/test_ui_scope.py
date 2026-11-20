from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate, QPoint, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from localmind import chat_ui
from localmind.chat_scope_ui import ConversationScopeDialog
from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.ui import APP_STYLESHEET, MainWindow
from localmind.weflow import WeFlowClient
from localmind.weflow_scope import CATEGORIES
from test_weflow_batch import BatchTransport
from test_weflow_alpha import alpha_message


def rows():
    return [{"id": "person-a", "name": "王同事", "category": "private", "aliases": ["技术负责人"]},
            {"id": "person-b", "name": "王同事", "category": "private"},
            {"id": "group-a", "name": "设计讨论群", "category": "group"},
            {"id": "group-b", "name": "验收讨论群", "category": "group"},
            {"id": "company", "name": "企业用户", "category": "enterprise"},
            {"id": "official", "name": "资讯公众号", "category": "official"},
            {"id": "mystery", "name": "未知会话", "category": "unknown"}]


def test_two_level_selection_actions_and_geometry():
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    dialog = ConversationScopeDialog(rows(), base_url="http://127.0.0.1:5031", account_id="synthetic", account_name="模拟账号", period="2026-10-01 至今")
    dialog.setStyleSheet(APP_STYLESHEET)
    dialog.show()
    try:
        assert dialog.model.selected_ids == {"person-a", "person-b", "group-a", "group-b"}
        dialog.search.setText("技术负责人")
        assert dialog.visible_ids() == {"person-a"}
        dialog.select_none.click()
        assert dialog.model.selected_ids == {"person-b", "group-a", "group-b"}
        dialog.invert.click()
        assert "person-a" in dialog.model.selected_ids
        dialog.categories.setCurrentRow(list(CATEGORIES).index("group"))
        dialog.search.setText("设计")
        dialog.select_none.click()
        assert "group-a" not in dialog.model.selected_ids and "group-b" in dialog.model.selected_ids
        dialog.select_all.click()
        dialog.search.clear()
        item = dialog.categories.item(list(CATEGORIES).index("official"))
        item.setCheckState(Qt.CheckState.Checked)
        assert "official" in dialog.model.selected_ids
        item.setCheckState(Qt.CheckState.Unchecked)
        assert "official" not in dialog.model.selected_ids
        dialog.model.select(set(dialog.model.selected_ids), "none")
        assert not dialog.confirm_button.isEnabled()
        dialog.reset_button.click()
        assert dialog.confirm_button.isEnabled() and len(dialog.selection()["ids"]) == 4
        for width, height in ((740, 480), (900, 600)):
            dialog.resize(width, height)
            app.processEvents()
            for widget in (dialog.categories, dialog.table, dialog.confirm_button, dialog.total, dialog.search):
                assert widget.mapTo(dialog, QPoint(widget.width(), widget.height())).x() <= dialog.width()
                assert widget.mapTo(dialog, QPoint(0, widget.height())).y() <= dialog.height()
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                target = Path(output)
                target.mkdir(parents=True, exist_ok=True)
                assert dialog.grab().save(str(target / f"conversation-scope-{width}.png"))
    finally:
        dialog.close()
        app.processEvents()


def test_metadata_cancel_confirm_and_automatic_scope_are_separate(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    db = LocalMindDB(tmp_path / "index.db")
    for key in ("privacy_accepted_v02", "weflow_bulk_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    transport = BatchTransport(4)
    transport.sessions[1]["sessionType"] = "group"
    transport.sessions[2]["sessionType"] = "official"
    transport.sessions[3]["sessionType"] = "enterprise"

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
        while panel.busy and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert not panel.busy
        app.processEvents()

    try:
        window.show()
        panel.token.setText("synthetic-token")
        panel.start_date.setDate(QDate(2026, 5, 1))
        panel.sync_button.click()
        wait()
        assert panel.scope_dialog is not None
        assert panel.scope_dialog.model.selected_ids == {"chat-0", "chat-1"}
        assert not any(p == "/api/v1/messages" or p.startswith("/api/v1/sessions/") for p, _ in transport.calls)
        panel.scope_dialog.reject()
        assert not ChatStore(db).sources()
        assert not db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_batch:%' OR key LIKE 'weflow_scope:%'").fetchone()
        panel.sync()
        wait()
        panel.scope_dialog.model.select({"chat-0"}, "none")
        panel.scope_dialog.confirm_button.click()
        wait()
        assert len(ChatStore(db).sources()) == 1
        assert panel.active_scope["ids"] == ["chat-1"]
        assert panel.confirmed_target == panel.target()
        assert panel.scope_label.text() == "已选 1 个"
        assert not any(q.get("talker") in {"chat-0", "chat-2", "chat-3"} for p, q in transport.calls if p == "/api/v1/messages")
        transport.sessions.append({"id": "new-group", "display_name": "新建群", "sessionType": "group"})
        transport.rows["new-group"] = [alpha_message(99, conversation_id="new-group")]
        transport.calls.clear()
        panel.auto.setChecked(True)
        panel.auto_sync()
        wait()
        assert panel.scope_dialog is None and panel.auto.isChecked()
        assert [q["talker"] for p, q in transport.calls if p == "/api/v1/messages"] == ["chat-1"]
        assert "新发现 1" in panel.status.text()
        panel.sync()
        wait()
        assert panel.scope_dialog.model.selected_ids == {"chat-1"}
        assert "新增 1" in panel.scope_dialog.notice.text()
        panel.scope_dialog.reject()
        assert len(ChatStore(db).sources()) == 1
        panel.token.setText("changed-token")
        assert panel.active_scope is None and not panel.auto.isChecked()
        stored = " ".join(row[0] for row in db.conn.execute("SELECT value FROM settings"))
        assert "synthetic-token" not in stored and "changed-token" not in stored
    finally:
        if panel.scope_dialog:
            panel.scope_dialog.reject()
        panel.cancel.set()
        wait()
        window.close()
        app.processEvents()
        assert not panel.thread.isRunning()
