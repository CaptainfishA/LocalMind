"""Workspace interactions that do not change search or synchronization semantics."""

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMenu, QPlainTextEdit, QVBoxLayout

from ..ui_formatting import plain_tooltip
from .widgets import icon


class DesktopInteractions:
    def _initialize_desktop_interactions(self):
        mode = self.db.get_setting("ui_sidebar_mode", "auto")
        self._sidebar_mode = mode if mode in {"auto", "compact", "expanded"} else "auto"
        self._shortcuts = []
        self._result_menu = None
        for sequence, callback in [("Ctrl+K", self.focus_search), ("Ctrl+F", self.focus_search)]:
            self._shortcuts.append(QShortcut(QKeySequence(sequence), self, activated=callback))
        for index, nav in enumerate(self.nav_buttons):
            nav.setToolTip(f"{self.page_names[index]} (Ctrl+{index + 1})")
            self._shortcuts.append(QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self,
                                            activated=lambda n=index: self.tabs.setCurrentIndex(n)))
        self._open_shortcut = QShortcut(QKeySequence("Return"), self.result_table,
                                       activated=self.open_selected_file)
        self._open_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.result_table.cellDoubleClicked.connect(lambda *_: self.open_selected_file())
        self.result_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.result_table.customContextMenuRequested.connect(self._show_result_menu)
        for field in (self.conversation_filter, self.sender_filter):
            field.textChanged.connect(self._update_filter_summary)
        self.date_mode_filter.currentIndexChanged.connect(self._update_filter_summary)
        self.enable_reranker.toggled.connect(self._update_filter_summary)
        self._update_filter_summary()
        geometry = self.db.get_setting("ui_window_geometry", "")
        split = self.db.get_setting("ui_search_splitter", "")
        if geometry:
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii", errors="ignore")))
            if not any(screen.availableGeometry().intersects(self.frameGeometry()) for screen in QGuiApplication.screens()):
                self.move(QGuiApplication.primaryScreen().availableGeometry().topLeft())
        if split:
            self.search_splitter.restoreState(QByteArray.fromBase64(split.encode("ascii", errors="ignore")))
        self._resize_workspace()

    def focus_search(self):
        self.tabs.setCurrentIndex(0)
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _toggle_sidebar(self):
        self._sidebar_mode = "expanded" if self.sidebar.width() < 100 else "compact"
        self.db.set_setting("ui_sidebar_mode", self._sidebar_mode)
        self._resize_workspace()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_sidebar_mode"):
            self._resize_workspace()
            QTimer.singleShot(0, self._resize_workspace)

    def _resize_workspace(self):
        compact = self._sidebar_mode == "compact" or (self._sidebar_mode == "auto" and self.width() < 1180)
        self.sidebar.setFixedWidth(68 if compact else 184)
        for index, nav in enumerate(self.nav_buttons):
            nav.setText("" if compact else self.page_names[index])
        self.brand_label.setVisible(not compact)
        self.nav_caption.setVisible(not compact)
        self.version_label.setVisible(not compact)
        self.library_status.setVisible(not compact)
        self.sidebar_toggle.setIcon(icon("panel-left-open" if compact else "panel-left-close"))
        self.sidebar_toggle.setToolTip("展开侧栏" if compact else "收起侧栏")
        self.result_table.setColumnHidden(5, self.search_splitter.sizes()[0] < 380)

    def _save_desktop_layout(self):
        self.db.set_setting("ui_window_geometry", bytes(self.saveGeometry().toBase64()).decode("ascii"))
        self.db.set_setting("ui_search_splitter", bytes(self.search_splitter.saveState().toBase64()).decode("ascii"))

    def _update_filter_summary(self):
        parts = []
        if self.conversation_filter.text().strip():
            parts.append("会话：" + self.conversation_filter.text().strip())
        if self.sender_filter.text().strip():
            parts.append("发送者：" + self.sender_filter.text().strip())
        if self.date_mode_filter.currentData() is False:
            parts.append("不限时间")
        if self.enable_reranker.isChecked():
            parts.append("AI 精排")
        text = "  ·  ".join(parts)
        self.filter_summary.setText(text)
        self.filter_summary.setToolTip(plain_tooltip(text))
        self.filter_summary_band.setVisible(bool(parts))
        self.filter_toggle.setToolTip(f"更多筛选 · {len(parts)} 项已启用" if parts else "更多筛选")
        self.filter_toggle.setAccessibleName(f"更多筛选，{len(parts)} 项已启用")

    def reset_search_filters(self):
        if self.search_busy or self.chat_panel.busy:
            return
        self.conversation_filter.clear()
        self.sender_filter.clear()
        self.date_mode_filter.setCurrentIndex(0)
        self.type_filter.setCurrentIndex(0)
        self.scope_filter.setCurrentIndex(0)
        self.evidence_filter.setCurrentIndex(0)
        self.enable_reranker.setChecked(False)

    def _show_result_menu(self, position):
        index = self.result_table.indexAt(position)
        if not index.isValid():
            return
        self.result_table.selectRow(index.row())
        self.show_selected_result()
        menu = QMenu(self)
        for control in [*self.file_actions, self.bind_result_button, self.index_result_button]:
            action = menu.addAction(control.icon(), control.accessibleName())
            action.setEnabled(control.isEnabled())
            action.triggered.connect(control.click)
        self._result_menu = menu
        menu.aboutToHide.connect(lambda: setattr(self, "_result_menu", None))
        menu.aboutToHide.connect(menu.deleteLater)
        menu.popup(self.result_table.viewport().mapToGlobal(position))

    def show_search_diagnostics(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("查询诊断")
        dialog.resize(620, 420)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self.search_status.text() + "\n\n" + self.search_status.toolTip())
        layout.addWidget(text)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.exec()
        dialog.deleteLater()

    def _open_folder_setup(self):
        self.tabs.setCurrentIndex(2)
        self.add_folder()

    def _open_wechat_setup(self):
        self.tabs.setCurrentIndex(1)
        self.chat_panel.token.setFocus()
