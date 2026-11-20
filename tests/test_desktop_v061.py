import os
import sqlite3

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QStyle, QStyleOptionButton, QStyleOptionComboBox

from localmind.db import LocalMindDB
from localmind.ui import MainWindow
from test_result_lifecycle import screen  # noqa: F401


def test_responsive_sidebar_and_focus_shortcuts(screen):
    app, window, *_ = screen
    window.resize(980, 650)
    app.processEvents()
    assert window.sidebar.width() == 68
    assert all(not button.text() for button in window.nav_buttons)
    window.nav_buttons[1].click()
    assert window.tabs.currentWidget() == window.chat_panel
    window.focus_search()
    assert window.tabs.currentIndex() == 0 and window.search_input.hasFocus()
    window.sidebar_toggle.click()
    assert window.sidebar.width() == 184
    assert window.db.get_setting("ui_sidebar_mode") == "expanded"
    window.sidebar_toggle.click()
    window.resize(1380, 860)
    app.processEvents()
    assert window.sidebar.width() == 68
    assert window.nav_buttons[1].accessibleName() == "微信资料"


def test_collapsed_filter_summary_reset_and_busy_guard(screen):
    app, window, *_ = screen
    query = window.search_input.text()
    window.conversation_filter.setText("<b>合成项目群</b>")
    window.sender_filter.setText("示例联系人")
    window.date_mode_filter.setCurrentIndex(1)
    window.enable_reranker.setChecked(True)
    app.processEvents()
    assert not window.filter_panel.isVisible() and window.filter_summary_band.isVisible()
    assert "示例联系人" in window.filter_summary.text()
    assert "<b>" not in window.filter_summary.toolTip()
    window.search_busy = True
    window._update_result_controls()
    assert not window.reset_filters_button.isEnabled()
    window.reset_search_filters()
    assert window.sender_filter.text()
    window.search_busy = False
    window._update_result_controls()
    window.reset_filters_button.click()
    assert not window.sender_filter.text() and not window.conversation_filter.text()
    assert window.date_mode_filter.currentData() is True
    assert window.scope_filter.currentData() == "all"
    assert not window.filter_summary_band.isVisible()
    assert window.search_input.text() == query


def test_double_click_and_context_menu_keep_file_identity(screen, monkeypatch):
    app, window, _, first, *_ = screen
    opened = []
    monkeypatch.setattr(window, "perform_file_action", lambda *args: opened.append(args))
    window.result_table.cellDoubleClicked.emit(0, 0)
    assert opened == [(first["path"], "open")]
    point = window.result_table.visualItemRect(window.result_table.item(0, 0)).center()
    window._show_result_menu(point)
    app.processEvents()
    menu = window._result_menu
    assert menu is not None and all(action.text() for action in menu.actions())
    menu.actions()[2].trigger()
    assert opened[-1] == (first["path"], "copy")
    menu.close()
    app.processEvents()
    window._show_result_menu(QPoint(-20, -20))
    assert window._result_menu is None


def test_layout_restores_without_persisting_query_or_token(screen):
    app, window, db, *_ = screen
    db_path = db.db_path
    window.resize(1220, 780)
    window.search_splitter.setSizes([520, 450])
    window.chat_panel.token.setText("synthetic-memory-only-token")
    window.search_input.setText("synthetic-private-query")
    app.processEvents()
    sizes = window.search_splitter.sizes()
    window.close()
    app.processEvents()
    restored_db = LocalMindDB(db_path)
    restored = MainWindow(restored_db)
    restored.show()
    app.processEvents()
    try:
        # Qt clamps restored windows to the current monitor (offscreen is 800px).
        assert 980 <= restored.width() <= 1220 and 650 <= restored.height() <= 780
        assert restored_db.get_setting("ui_window_geometry")
        restored_sizes = restored.search_splitter.sizes()
        assert abs(restored_sizes[0] / sum(restored_sizes) - sizes[0] / sum(sizes)) < 0.12
        assert not restored.search_input.text() and not restored.chat_panel.token.text()
        stored = "\n".join(str(row[0]) for row in restored_db.conn.execute("SELECT value FROM settings"))
        assert "synthetic-memory-only-token" not in stored
        assert "synthetic-private-query" not in stored
    finally:
        restored.close()
        app.processEvents()


def test_empty_library_routes_to_existing_setup_without_network(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    db = LocalMindDB(tmp_path / "index.db")
    for key in ("privacy_accepted_v02", "weflow_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    window = MainWindow(db)
    window.show()
    app.processEvents()
    folders = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: folders.append(True) or "")
    try:
        assert window.search_content.currentWidget() == window.empty_library
        window.empty_folder_button.click()
        assert folders == [True] and window.tabs.currentIndex() == 2
        window.focus_search()
        window.empty_wechat_button.click()
        assert window.tabs.currentWidget() == window.chat_panel
        assert not window.chat_panel.busy and window.chat_panel.token.hasFocus()
    finally:
        window.close()
        app.processEvents()


def test_result_status_is_concise_but_keeps_timing_diagnostics(screen):
    app, window, *_ = screen
    assert "耗时 10 ms" in window.search_status.text()
    assert "候选" not in window.search_status.text()
    assert "候选" in window.search_status.toolTip()
    assert "数据库读取区间" in window.search_status.toolTip()


def test_control_indicators_have_visible_pixels_and_state(screen):
    app, window, *_ = screen
    window.filter_toggle.setChecked(True)
    checkbox = window.enable_reranker
    checkbox.setChecked(False)
    app.processEvents()
    option = QStyleOptionButton()
    checkbox.initStyleOption(option)
    rect = checkbox.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, checkbox)
    unchecked = checkbox.grab(rect).toImage()
    assert any(unchecked.pixelColor(x, y).lightness() < 200
               for x in range(unchecked.width()) for y in range(unchecked.height()))
    checkbox.setChecked(True)
    app.processEvents()
    assert checkbox.grab(rect).toImage() != unchecked
    combo = window.scope_filter
    option = QStyleOptionComboBox()
    combo.initStyleOption(option)
    rect = combo.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                       QStyle.SubControl.SC_ComboBoxArrow, combo)
    arrow = combo.grab(rect.adjusted(3, 5, -3, -5)).toImage()
    assert any(arrow.pixelColor(x, y).lightness() < 100
               for x in range(arrow.width()) for y in range(arrow.height()))
    window.resize(980, 650)
    window.tabs.setCurrentWidget(window.chat_panel)
    app.processEvents()
    combo = window.chat_panel.accounts
    option = QStyleOptionComboBox()
    combo.initStyleOption(option)
    rect = combo.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                       QStyle.SubControl.SC_ComboBoxEditField, combo)
    assert rect.width() >= combo.fontMetrics().horizontalAdvance(combo.currentText()) + 4


def test_preference_write_failure_does_not_block_shutdown(screen, monkeypatch):
    app, window, *_ = screen
    def unavailable():
        raise sqlite3.OperationalError("synthetic preference write failure")
    monkeypatch.setattr(window, "_save_desktop_layout", unavailable)
    assert window.close()
    assert not window.search_thread.isRunning()
    assert not window.incremental_thread.isRunning()
