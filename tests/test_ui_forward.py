from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.models import SearchQuery
from localmind.ui import MainWindow
from test_weflow_forward import item, message
from test_weflow_v04 import index_file, ingest, searcher


def test_forward_attachment_origin_preview_and_pending_actions(tmp_path):
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db = LocalMindDB(tmp_path / "index.db")
    for key in ("privacy_accepted_v02", "weflow_bulk_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    target = index_file(db, tmp_path / "项目交付清单.txt", "质量验收通过后支付尾款")
    ingest(db, [message([item(1, "发给南宁评审组的竣工报告"), item(8, "项目交付清单.txt", name="模拟原作者"),
                        item(8, "未下载的附件.pdf")])])
    attachment = next(a for a in ChatStore(db).attachments() if a["name"] == "项目交付清单.txt")
    ChatStore(db).bind(attachment["id"], target["id"])
    window = MainWindow(db)
    window.show()
    try:
        service = searcher(db, mode="keyword")
        window.search_input.setText("南宁评审组")
        window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
        results = service.search(SearchQuery("南宁评审组", scope="chat_files"))
        window._search_finished(results, service.last_diagnostics, {"total_ms": 14, "index_refresh_ms": 0})
        assert results and results[0].path == target["path"]
        assert "合并转发附件上下文" in window.info_preview.toPlainText()
        assert window.preview_tabs.isTabEnabled(1)
        window.preview_tabs.setCurrentIndex(1)
        window.context_expand.click()
        assert "原发送者: 模拟原作者" in window.context_preview.toPlainText()
        assert "第 2 项" in window.context_preview.toPlainText()
        assert all(button.isEnabled() for button in window.file_actions)
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            assert window.width() == width and window.height() == height
            for widget in (window.context_preview, window.search_button, window.bind_result_button):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            window.context_preview.verticalScrollBar().setValue(window.context_preview.verticalScrollBar().maximum())
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"forward-result-{width}.png"))
        window.tabs.setCurrentWidget(window.chat_panel)
        window.chat_panel.refresh_attachments()
        table = window.chat_panel.attachment_table
        assert table.rowCount() == 2
        assert all(table.item(row, 2).text().startswith("转发者:") for row in range(2))
        assert any("原发送者: 模拟原作者" in table.item(row, 0).toolTip() for row in range(2))
        app.processEvents()
        if output:
            assert window.grab().save(str(Path(output) / "forward-attachments.png"))
        missing = service.search(SearchQuery("未下载的附件", scope="chat_files"))
        window._search_finished(missing, service.last_diagnostics, {"total_ms": 12, "index_refresh_ms": 0})
        assert missing[0].result_kind == "attachment_pending"
        assert all(button.isEnabled() for button in window.file_actions)
        assert window.bind_result_button.isEnabled()
    finally:
        window.close()
        app.processEvents()
        assert not window.chat_panel.thread.isRunning()
