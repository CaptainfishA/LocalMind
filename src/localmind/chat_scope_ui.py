"""Metadata-only, two-level conversation selection dialog."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSplitter, QStyle, QTableView,
    QVBoxLayout, QWidget, QMenu, QToolButton,
)

from .weflow_scope import CATEGORIES, DEFAULT_CATEGORIES, make_scope, restored_ids
from .chat_members_ui import GroupMemberDialog


class SessionSelectionModel(QAbstractTableModel):
    selection_changed = Signal()

    def __init__(self, sessions, selected_ids, parent=None):
        super().__init__(parent)
        self.rows = sessions
        self.selected_ids = set(selected_ids)
        self.search_text = [" ".join([r["name"], r["id"], *r.get("aliases", [])]).casefold() for r in sessions]

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 4

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ("", "会话", "标识", "类型")[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, column = self.rows[index.row()], index.column()
        if role == Qt.ItemDataRole.CheckStateRole and column == 0:
            return Qt.CheckState.Checked if row["id"] in self.selected_ids else Qt.CheckState.Unchecked
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            if role == Qt.ItemDataRole.ToolTipRole and column == 1:
                return "\n".join([row["name"], *row.get("aliases", [])])
            return ("", row["name"], row["id"], CATEGORIES[row.get("category", "unknown")])[column]
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or index.column() != 0 or role != Qt.ItemDataRole.CheckStateRole:
            return False
        self.select({self.rows[index.row()]["id"]}, "all" if Qt.CheckState(value) == Qt.CheckState.Checked else "none")
        return True

    def select(self, ids, mode):
        ids = set(ids)
        if mode == "all":
            self.selected_ids.update(ids)
        elif mode == "none":
            self.selected_ids.difference_update(ids)
        elif mode == "invert":
            self.selected_ids.symmetric_difference_update(ids)
        if self.rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, 0), [Qt.ItemDataRole.CheckStateRole])
        self.selection_changed.emit()


class SessionFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.category = "private"
        self.keyword = ""
        self.labels = set()
        self.contact_labels = {}

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        row = model.rows[source_row]
        return (row.get("category", "unknown") == self.category and self.keyword in model.search_text[source_row]
                and (self.category != "private" or not self.labels or bool(self.labels & self.contact_labels.get(row["id"], set()))))

    def set_filter(self, category=None, keyword=None):
        if category is not None:
            self.category = category
        if keyword is not None:
            self.keyword = keyword.strip().casefold()
        self.invalidate()


class ConversationScopeDialog(QDialog):
    metadata_requested = Signal(object)
    metadata_cancelled = Signal()

    def __init__(self, sessions, *, base_url, account_id, account_name, period, previous=None, parent=None, metadata_available=False):
        super().__init__(parent)
        self.setWindowTitle("选择索引会话")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumSize(740, 480)
        self.resize(900, 600)
        self.base_url, self.account_id = base_url, account_id
        self.metadata_available = metadata_available
        self.metadata_busy = False
        self.member_dialog = None
        self.model = SessionSelectionModel(sessions, restored_ids(sessions, previous), self)
        self.proxy = SessionFilter(self)
        self.proxy.setSourceModel(self.model)
        self.category_ids = {category: {row["id"] for row in sessions if row.get("category", "unknown") == category}
                             for category in CATEGORIES}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        heading = QLabel("索引范围")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        identity = QLabel(f"{account_name} · {period}")
        identity.setTextFormat(Qt.TextFormat.PlainText)
        identity.setWordWrap(True)
        identity.setObjectName("mutedText")
        layout.addWidget(identity)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setObjectName("mutedText")
        unknown = len(self.category_ids["unknown"])
        missing = len(set(previous["ids"]) - self.model.selected_ids) if previous else 0
        notes = []
        if unknown:
            notes.append(f"{unknown} 个会话类型未知")
        if missing:
            notes.append(f"上次选择中 {missing} 个会话已不可见或类型改变")
        if previous:
            added = len({r["id"] for r in sessions} - set(previous["known_ids"]))
            if added:
                notes.append(f"新增 {added} 个会话，未自动勾选")
        self.notice.setText(" · ".join(notes))
        self.notice.setVisible(bool(notes))
        layout.addWidget(self.notice)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.categories = QListWidget()
        self.categories.setMinimumWidth(160)
        self.categories.setMaximumWidth(220)
        self.categories.setSpacing(2)
        self.categories.setUniformItemSizes(True)
        self.categories.setAccessibleName("会话类别")
        for key, label in CATEGORIES.items():
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setSizeHint(QSize(170, 32))
            item.setToolTip(label if self.category_ids[key] else f"当前列表未识别到{label}")
            self.categories.addItem(item)
        split.addWidget(self.categories)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索名称、备注或会话 ID")
        self.search.setClearButtonEnabled(True)
        right_layout.addWidget(self.search)
        metadata_row = QHBoxLayout()
        self.label_button = QToolButton()
        self.label_button.setText("全部标签")
        self.label_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.label_menu = QMenu(self)
        self.label_button.setMenu(self.label_menu)
        self.read_labels = self.action("读取标签", QStyle.StandardPixmap.SP_BrowserReload, self.load_labels)
        self.by_member = self.action("按成员选群", QStyle.StandardPixmap.SP_FileDialogContentsView, self.open_members)
        self.stop_metadata = self.action("取消读取", QStyle.StandardPixmap.SP_BrowserStop, self.metadata_cancelled.emit)
        for widget in (self.label_button, self.read_labels, self.by_member, self.stop_metadata):
            metadata_row.addWidget(widget)
        metadata_row.addStretch()
        right_layout.addLayout(metadata_row)
        self.metadata_status = QLabel()
        self.metadata_status.setWordWrap(True)
        self.metadata_status.setTextFormat(Qt.TextFormat.PlainText)
        self.metadata_status.setObjectName("mutedText")
        self.metadata_status.hide()
        right_layout.addWidget(self.metadata_status)
        actions = QHBoxLayout()
        self.select_all = self.action("全选当前结果", QStyle.StandardPixmap.SP_DialogApplyButton, lambda: self.select_visible("all"))
        self.select_none = self.action("取消全选", QStyle.StandardPixmap.SP_DialogCancelButton, lambda: self.select_visible("none"))
        self.invert = self.action("反选", QStyle.StandardPixmap.SP_BrowserReload, lambda: self.select_visible("invert"))
        self.only_results = self.action("仅选结果", QStyle.StandardPixmap.SP_DialogApplyButton, self.select_only_visible)
        for button in (self.select_all, self.select_none, self.invert):
            button.setToolTip(button.text() + "（仅当前类别的搜索结果）")
            actions.addWidget(button)
        actions.addStretch()
        self.only_results.setToolTip("用当前搜索结果替换当前类别的选择，其它类别不变")
        actions.addWidget(self.only_results)
        right_layout.addLayout(actions)
        self.table = QTableView()
        self.table.setAccessibleName("具体会话")
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 36)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(2, 160)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 90)
        right_layout.addWidget(self.table, 1)
        self.visible_count = QLabel()
        self.visible_count.setObjectName("mutedText")
        right_layout.addWidget(self.visible_count)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([190, 650])
        layout.addWidget(split, 1)
        footer = QHBoxLayout()
        self.reset_button = self.action("恢复默认", QStyle.StandardPixmap.SP_BrowserReload, self.reset_defaults)
        self.reset_button.setToolTip("仅勾选当前可见的全部单人会话与群聊")
        footer.addWidget(self.reset_button)
        self.total = QLabel()
        self.total.setWordWrap(True)
        footer.addWidget(self.total, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.confirm_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.confirm_button.setText("确认并建立索引")
        self.confirm_button.setObjectName("primaryButton")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        footer.addWidget(self.buttons)
        layout.addLayout(footer)
        self.categories.currentItemChanged.connect(self.category_changed)
        self.categories.itemChanged.connect(self.category_toggled)
        self.search.textChanged.connect(self.search_changed)
        self.model.selection_changed.connect(self.refresh_counts)
        self.categories.setCurrentRow(next((index for index, key in enumerate(CATEGORIES) if self.category_ids[key]), 0))
        self.refresh_counts()

    def action(self, text, icon, slot):
        button = QPushButton(text)
        button.setIcon(self.style().standardIcon(icon))
        button.clicked.connect(slot)
        return button

    def category_changed(self, current, _previous):
        if current is not None:
            self.proxy.set_filter(category=current.data(Qt.ItemDataRole.UserRole))
            self.refresh_counts()

    def category_toggled(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        self.model.select(self.category_ids[key], "none" if item.checkState() == Qt.CheckState.Unchecked else "all")

    def search_changed(self, text):
        self.proxy.set_filter(keyword=text)
        self.refresh_counts()

    def visible_ids(self):
        return {self.model.rows[self.proxy.mapToSource(self.proxy.index(i, 0)).row()]["id"] for i in range(self.proxy.rowCount())}

    def select_visible(self, mode):
        self.model.select(self.visible_ids(), mode)

    def select_only_visible(self):
        ids = self.visible_ids()
        self.model.select(self.category_ids[self.proxy.category], "none")
        self.model.select(ids, "all")

    def reset_defaults(self):
        self.model.selected_ids = {row["id"] for row in self.model.rows if row.get("category") in DEFAULT_CATEGORIES}
        self.model.select(set(), "all")

    def refresh_counts(self):
        self.categories.blockSignals(True)
        for index, (key, label) in enumerate(CATEGORIES.items()):
            ids = self.category_ids[key]
            count = len(ids & self.model.selected_ids)
            item = self.categories.item(index)
            item.setText(f"{label}  {count}/{len(ids)}")
            item.setCheckState(Qt.CheckState.Checked if count and count == len(ids) else Qt.CheckState.PartiallyChecked if count else Qt.CheckState.Unchecked)
        self.categories.blockSignals(False)
        visible = self.visible_ids()
        self.visible_count.setText(f"当前结果 {len(visible)} 个 · 已选 {len(visible & self.model.selected_ids)} 个")
        self.total.setText(f"已选 {len(self.model.selected_ids)} / {len(self.model.rows)} 个会话")
        self.confirm_button.setEnabled(bool(self.model.selected_ids) and not self.metadata_busy)
        self.update_metadata_controls()

    def update_metadata_controls(self):
        private, group = self.proxy.category == "private", self.proxy.category == "group"
        self.label_button.setVisible(private)
        self.read_labels.setVisible(private)
        self.by_member.setVisible(group)
        self.stop_metadata.setVisible(self.metadata_busy)
        for button in (self.read_labels, self.by_member):
            button.setEnabled(self.metadata_available and not self.metadata_busy)
            button.setToolTip("" if self.metadata_available else "需要支持固定账号和联系人 MCP 的 Alpha 接口")
        if self.metadata_available:
            self.by_member.setToolTip("核对当前账号可见群的当前成员；不包含已退出或不可见的群")
        self.label_button.setEnabled(bool(self.label_menu.actions()) and not self.metadata_busy)
        self.confirm_button.setEnabled(bool(self.model.selected_ids) and not self.metadata_busy)

    def load_labels(self):
        if self.metadata_busy or not self.metadata_available:
            return
        self.request_metadata({"operation": "labels", "ids": sorted(self.category_ids["private"]),
                               "force": self.read_labels.text() == "刷新标签"})

    def request_metadata(self, request):
        if self.metadata_busy:
            return
        self.metadata_busy = True
        self.set_metadata_progress("正在读取联系人元数据")
        self.update_metadata_controls()
        self.metadata_requested.emit(request)

    def set_metadata_progress(self, text):
        self.metadata_status.setText(text)
        self.metadata_status.show()
        if self.member_dialog and self.member_dialog.busy:
            self.member_dialog.status.setText(text)

    def receive_metadata(self, result):
        self.metadata_busy = False
        if result["operation"] != "labels":
            if self.member_dialog:
                self.member_dialog.receive(result)
            self.set_metadata_progress(result.get("error", "成员查询已结束"))
        elif result.get("error"):
            self.set_metadata_progress(result["error"])
        else:
            value = result["value"]
            self.proxy.contact_labels = value["values"]
            available = set().union(*value["values"].values()) if value["values"] else set()
            self.proxy.labels.intersection_update(available)
            self.label_menu.clear()
            clear = self.label_menu.addAction("全部标签")
            clear.triggered.connect(self.clear_labels)
            self.label_menu.addSeparator()
            for label in sorted(available):
                action = QAction(label, self.label_menu)
                action.setCheckable(True)
                action.setChecked(label in self.proxy.labels)
                action.toggled.connect(lambda checked, text=label: self.toggle_label(text, checked))
                self.label_menu.addAction(action)
            self.read_labels.setText("刷新标签")
            stamps = value.get("timestamps", {}).values()
            stamp = datetime.fromtimestamp(min(stamps)).strftime("%H:%M:%S") if stamps else "--"
            self.set_metadata_progress(f"标签已确认 {len(value['values'])} · 未确认 {len(value['unknown'])} · 缓存 {value['cached']}"
                                       f" · 最早核对 {stamp}"
                                       + (" · 已取消" if value["state"] == "cancelled" else ""))
            if value["unknown"]:
                self.metadata_status.setToolTip("\n".join(sorted(set(value["unknown"].values()))))
            else:
                self.metadata_status.setToolTip("")
            if value["state"] == "unavailable":
                reason = next((reason for reason in value["unknown"].values() if reason != "未读取"), "标签能力不可用")
                self.set_metadata_progress(reason)
            self.filter_labels()
        self.update_metadata_controls()

    def clear_labels(self):
        self.proxy.labels.clear()
        for action in self.label_menu.actions():
            action.setChecked(False)
        self.filter_labels()

    def toggle_label(self, label, checked):
        self.proxy.labels.add(label) if checked else self.proxy.labels.discard(label)
        self.filter_labels()

    def filter_labels(self):
        self.label_button.setText(f"标签 ({len(self.proxy.labels)})" if self.proxy.labels else "全部标签")
        self.proxy.invalidate()
        self.refresh_counts()

    def open_members(self):
        if self.metadata_busy or self.member_dialog or not self.metadata_available:
            return
        dialog = GroupMemberDialog([row for row in self.model.rows if row.get("category") == "group"], self)
        self.member_dialog = dialog
        dialog.requested.connect(self.request_metadata)
        dialog.cancelled.connect(self.metadata_cancelled.emit)
        dialog.apply_matches.connect(self.apply_groups)

        def closed(_result):
            self.member_dialog = None
            dialog.deleteLater()

        dialog.finished.connect(closed)
        dialog.open()

    def apply_groups(self, ids, mode):
        if mode == "replace":
            self.model.select(self.category_ids["group"], "none")
        self.model.select(set(ids) & self.category_ids["group"], "all")

    def reject(self):
        if self.member_dialog:
            self.member_dialog.reject()
        if self.metadata_busy:
            self.metadata_cancelled.emit()
        super().reject()

    def selection(self):
        return make_scope(self.base_url, self.account_id, self.model.rows, self.model.selected_ids)
