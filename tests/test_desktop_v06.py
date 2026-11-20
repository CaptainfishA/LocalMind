"""Presentation contracts using only synthetic data and temporary indexes."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QStackedWidget

from localmind.desktop.widgets import icon
from test_result_lifecycle import screen  # noqa: F401


def test_workspace_navigation_and_local_results_without_alpha(screen):
    app, window, _, first, *_ = screen
    assert isinstance(window.tabs, QStackedWidget)
    assert window.tabs.count() == 5
    for index, nav in enumerate(window.nav_buttons):
        nav.click()
        app.processEvents()
        assert window.tabs.currentIndex() == index
        assert window.page_title.text() == nav.text()
        assert nav.isChecked()
    window.nav_buttons[0].click()
    assert not window.chat_panel.token.text()
    assert window.selected_result().file_id == first["id"]
    assert window.detail_title.text() == first["name"]
    assert window.result_table.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert window.result_table.isColumnHidden(3)
    assert "排序分" in window.info_preview.toPlainText()


def test_lucide_assets_render_at_multiple_pixel_ratios(screen):
    app, window, *_ = screen
    for name in ("search", "folder", "message-circle", "scan-search", "copy", "activity"):
        glyph = icon(name)
        assert not glyph.isNull()
        image = glyph.pixmap(24, 24).toImage()
        assert any(image.pixelColor(x, y).alpha() for x in range(image.width()) for y in range(image.height()))


def test_workspace_geometry_and_screenshots_all_pages(screen):
    app, window, *_ = screen
    output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
    for width, height in ((980, 650), (1380, 860)):
        window.resize(width, height)
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
            app.processEvents()
            assert window.width() == width and window.height() == height
            assert window.tabs.geometry().right() <= width
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"workspace-{index}-{width}.png"))
        window.tabs.setCurrentIndex(0)
        window.filter_toggle.setChecked(True)
        app.processEvents()
        for field in (window.conversation_filter, window.sender_filter, window.date_mode_filter,
                      window.search_button, window.preview_tabs, *window.file_actions):
            point = field.mapTo(window, QPoint(field.width(), field.height()))
            assert point.x() <= width and point.y() <= height
        window.filter_toggle.setChecked(False)


def test_filter_panel_keeps_controls_and_state(screen):
    app, window, *_ = screen
    assert not window.filter_panel.isVisible()
    window.filter_toggle.click()
    window.sender_filter.setText("演示联系人")
    window.date_mode_filter.setCurrentIndex(1)
    window.filter_toggle.click()
    assert window._current_search_request().query.sender == "演示联系人"
    assert window._current_search_request().query.infer_dates is False
    window.filter_toggle.click()
    assert window.sender_filter.text() == "演示联系人"
