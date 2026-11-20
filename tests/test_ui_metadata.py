from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from localmind import chat_ui
from localmind.chat_scope_ui import ConversationScopeDialog
from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.ui import APP_STYLESHEET, MainWindow
from localmind.weflow import WeFlowClient
from localmind.weflow_scope import CATEGORIES
from test_weflow_metadata import MetadataTransport
from test_weflow_alpha import alpha_message


def rows():
    return [{"id": "chat-0", "name": "王同事", "category": "private"},
            {"id": "chat-1", "name": "李同事", "category": "private"},
            {"id": "chat-2", "name": "王同事", "category": "private"},
            {"id": "group-a", "name": "研发项目讨论群", "category": "group"},
            {"id": "group-b", "name": "验收项目讨论群", "category": "group"}]


def screenshot(dialog, name):
    output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
    if output:
        target = Path(output)
        target.mkdir(parents=True, exist_ok=True)
        assert dialog.grab().save(str(target / name))


def test_label_union_name_intersection_and_unknown_not_untagged():
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    dialog = ConversationScopeDialog(rows(), base_url="http://127.0.0.1:5031", account_id="mock",
                                     account_name="模拟账号", period="2026-10-01 至今", metadata_available=True)
    dialog.setStyleSheet(APP_STYLESHEET)
    requests = []
    dialog.metadata_requested.connect(requests.append)
    dialog.show()
    try:
        dialog.read_labels.click()
        assert requests[-1]["ids"] == ["chat-0", "chat-1", "chat-2"] and not dialog.confirm_button.isEnabled()
        dialog.receive_metadata({"operation": "labels", "value": {"values": {"chat-0": {"工作", "研发"}, "chat-1": {"工作"}},
                                "unknown": {"chat-2": "标签接口未返回标签"}, "cached": 0, "state": "partial"}})
        assert "未确认 1" in dialog.metadata_status.text()
        dialog.toggle_label("研发", True)
        assert dialog.visible_ids() == {"chat-0"}
        dialog.toggle_label("工作", True)
        assert dialog.visible_ids() == {"chat-0", "chat-1"}
        dialog.search.setText("王")
        assert dialog.visible_ids() == {"chat-0"}
        dialog.only_results.click()
        assert dialog.model.selected_ids == {"chat-0", "group-a", "group-b"}
        dialog.search.clear()
        for width, height in ((740, 480), (900, 600)):
            dialog.resize(width, height)
            app.processEvents()
            assert dialog.width() == width and dialog.height() == height
            for widget in (dialog.read_labels, dialog.label_button, dialog.only_results, dialog.confirm_button, dialog.metadata_status):
                corner = widget.mapTo(dialog, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            screenshot(dialog, f"labels-{width}.png")
        dialog.clear_labels()
        assert dialog.visible_ids() == {"chat-0", "chat-1", "chat-2"}
    finally:
        dialog.close()
        app.processEvents()


def test_member_preview_requires_explicit_identity_and_preserves_unknown_groups():
    app = QApplication.instance() or QApplication([])
    scope = ConversationScopeDialog(rows(), base_url="http://127.0.0.1:5031", account_id="mock",
                                    account_name="模拟账号", period="近 30 天", metadata_available=True)
    scope.setStyleSheet(APP_STYLESHEET)
    scope.categories.setCurrentRow(list(CATEGORIES).index("group"))
    scope.model.select({"group-a", "group-b"}, "none")
    scope.show()
    scope.open_members()
    dialog = scope.member_dialog
    try:
        scope.receive_metadata({"operation": "resolve", "value": [{"id": "person-a", "name": "王同事"}, {"id": "person-b", "name": "王同事"}]})
        assert dialog.contacts.currentData() is None and not dialog.scan.isEnabled()
        dialog.contacts.setCurrentIndex(0)
        assert dialog.scan.isEnabled()
        value = {"matches": {"group-a"}, "nonmatches": set(), "unknown": {"group-b": "未返回成员列表"},
                 "timestamps": {"group-a": time.time()}, "cached": 1, "state": "partial"}
        scope.receive_metadata({"operation": "groups", "value": value})
        assert dialog.append.isEnabled() and not dialog.replace.isEnabled()
        assert "group-a" not in scope.model.selected_ids
        for width, height in ((640, 460), (760, 560)):
            dialog.resize(width, height)
            app.processEvents()
            assert dialog.width() == width and dialog.height() == height
            for widget in (dialog.contacts, dialog.scan, dialog.append, dialog.replace, dialog.status):
                corner = widget.mapTo(dialog, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            screenshot(dialog, f"group-members-{width}.png")
        dialog.append.click()
        assert scope.member_dialog is None and "group-a" in scope.model.selected_ids and "group-b" not in scope.model.selected_ids
        assert {"chat-0", "chat-1", "chat-2"} <= scope.model.selected_ids
    finally:
        scope.reject()
        app.processEvents()


def test_metadata_worker_to_scope_to_index_without_preconfirmation_messages(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    db = LocalMindDB(tmp_path / "index.db")
    for key in ("privacy_accepted_v02", "weflow_bulk_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    transport = MetadataTransport()
    transport.sessions = [{"id": row["id"], "display_name": row["name"], "sessionType": row["category"]} for row in rows()]
    transport.rows = {row["id"]: [alpha_message(i + 1, conversation_id=row["id"])] for i, row in enumerate(rows())}

    def client(url, token="", **kwargs):
        client = WeFlowClient(url, token, **kwargs)
        client.opener = transport
        return client

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
        panel.period.setCurrentIndex(2)
        panel.sync()
        wait()
        scope = panel.scope_dialog
        scope.read_labels.click()
        scope.reject()
        wait()
        assert panel.scope_dialog is None and not ChatStore(db).sources()
        panel.sync()
        wait()
        scope = panel.scope_dialog
        scope.read_labels.click()
        wait()
        assert not ChatStore(db).sources() and scope.read_labels.text() == "刷新标签"
        scope.toggle_label("研发", True)
        scope.only_results.click()
        scope.categories.setCurrentRow(list(CATEGORIES).index("group"))
        scope.open_members()
        members = scope.member_dialog
        members.query.setText("王同事")
        members.lookup.click()
        wait()
        members.contacts.setCurrentIndex(0)
        members.scan.click()
        wait()
        assert members.replace.isEnabled() and members.result["matches"] == {"group-a"}
        assert not any(path == "/api/v1/messages" or path.startswith("/api/v1/sessions/") for path, _ in transport.calls)
        members.replace.click()
        assert scope.model.selected_ids == {"chat-0", "group-a"}
        scope.confirm_button.click()
        wait()
        assert len(ChatStore(db).sources()) == 2
        assert panel.active_scope["ids"] == ["chat-0", "group-a"]
        persisted = " ".join(row[0] for row in db.conn.execute("SELECT value FROM settings"))
        assert "synthetic-token" not in persisted and "person-a" not in persisted and '"labels":' not in persisted
        panel.token.clear()
        for _ in range(10):
            app.processEvents()
            time.sleep(0.01)
        assert not panel.worker.metadata_cache.entries
    finally:
        if panel.scope_dialog:
            panel.scope_dialog.reject()
        panel.cancel.set()
        wait()
        window.close()
        app.processEvents()
        assert not panel.thread.isRunning()
