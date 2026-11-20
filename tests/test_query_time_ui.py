import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMessageBox

from localmind.query_processing import process_query
from localmind import searcher as search_module
from test_result_lifecycle import screen, wait_for  # noqa: F401


def test_date_controls_display_the_effective_range_and_can_remove_it(screen, monkeypatch):
    app, window, db, first, *_ = screen
    with db.conn:
        db.conn.execute("UPDATE chat_messages SET sent_at='2026-09-12T04:00:00Z'")
    monkeypatch.setattr(search_module, "process_query",
                        lambda text, **kw: process_query(text, now=datetime(2026, 10, 21), **kw))
    window.search_input.setText("上个月验收合同")
    assert window.date_mode_filter.currentData() is True
    window.run_search()
    assert not window.date_mode_filter.isEnabled()
    wait_for(app, lambda: not window.search_busy)
    assert window.selected_result().file_id == first["id"]
    assert window.date_mode_filter.isEnabled()
    assert window._displayed_request.query.modified_after.startswith("2026-09")
    assert window.search_time_status.isVisible()
    assert window.search_time_status.text().startswith("发送时间：")
    assert "2026-09-01" in window.search_time_status.text() and "2026-10-01" in window.search_time_status.text()
    assert "终点不包含" in window.search_time_status.toolTip()
    assert window.expand_results_button.isEnabled()
    for width, height in ((980, 650), (1240, 800)):
        window.resize(width, height)
        app.processEvents()
        assert window.width() == width and window.height() == height
        for widget in (window.date_mode_filter, window.search_time_status, window.sender_filter,
                       window.search_button, window.file_actions[-1]):
            origin = widget.mapTo(window, QPoint(0, 0))
            corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
            assert origin.x() >= 0 and origin.y() >= 0
            assert corner.x() <= width and corner.y() <= height
            assert widget.width() >= (36 if widget.objectName() == "iconButton" else 50)
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(Path(output) / f"query-time-{width}.png"))
    window.date_mode_filter.setCurrentIndex(1)
    assert not window.expand_results_button.isEnabled()
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert window.selected_result().file_id == first["id"]
    assert window.search_time_status.text() == "发送时间：不限"
    assert window._displayed_request.query.modified_after is None
    assert window._displayed_request.query.modified_before is None
    assert not window._displayed_request.query.infer_dates
    assert db.get_setting("search_infer_dates") == "0"
    window.expand_search()
    wait_for(app, lambda: not window.search_busy)
    assert window._displayed_request.query.limit == 50
    assert window._displayed_request.query.modified_after is None
    assert not window._displayed_request.query.infer_dates
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.clear_index()
    assert not window.search_time_status.isVisible() and not window.search_time_status.text()


def test_time_label_tracks_the_completed_scope_not_unsubmitted_control_changes(screen):
    app, window, *_ = screen
    previous = window.search_time_status.text()
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("files"))
    assert window.search_time_status.text() == previous
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert window.search_time_status.text() == "修改时间：不限"
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("all"))
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert window.search_time_status.text() == "检索时间：不限"
    window.conversation_filter.setText("模拟项目群")
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert window.search_time_status.text() == "发送时间：不限"
