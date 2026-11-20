"""Member lookup preview; applying matches is always an explicit action."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPushButton, QStyle, QTableView, QVBoxLayout,
)


class GroupMatchModel(QAbstractTableModel):
    def __init__(self, sessions, parent=None):
        super().__init__(parent)
        self.sessions, self.rows = sessions, []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ("群聊", "标识", "成员核对")[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return self.rows[index.row()][index.column()]

    def set_result(self, result):
        self.beginResetModel()
        self.rows = []
        for row in self.sessions:
            uid = row["id"]
            status = "匹配" if uid in result["matches"] else "不在成员列表" if uid in result["nonmatches"] else result["unknown"].get(uid, "未确认")
            self.rows.append((row["name"], uid, status))
        self.rows.sort(key=lambda row: (row[1] not in result["matches"], row[1] in result["nonmatches"], row[0]))
        self.endResetModel()


class GroupMemberDialog(QDialog):
    requested = Signal(object)
    cancelled = Signal()
    apply_matches = Signal(object, str)

    def __init__(self, sessions, parent=None):
        super().__init__(parent)
        self.sessions, self.result = sessions, None
        self.busy = False
        self.setWindowTitle("按成员选择群聊")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumSize(640, 460)
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("联系人名称或微信内部 ID")
        self.query.setMaxLength(256)
        self.lookup = QPushButton("查找联系人")
        self.lookup.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        row.addWidget(self.query, 1)
        row.addWidget(self.lookup)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.contacts = QComboBox()
        self.contacts.setMinimumWidth(200)
        self.contacts.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.contacts.setMinimumContentsLength(16)
        self.scan = QPushButton("查询参与的群")
        self.scan.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView))
        row.addWidget(self.contacts, 1)
        row.addWidget(self.scan)
        layout.addLayout(row)
        self.force = QCheckBox("刷新成员缓存")
        self.force.setToolTip("本机内存缓存保留 10 分钟；刷新时重新读取当前账号可见群的成员")
        layout.addWidget(self.force)
        self.status = QLabel("尚未查询")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.model = GroupMatchModel(sessions, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(1, 180)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        footer = QHBoxLayout()
        self.stop = QPushButton("取消查询")
        self.append = QPushButton("追加匹配群")
        self.replace = QPushButton("替换已选群")
        self.close_button = QPushButton("关闭")
        self.replace.setToolTip("仅在所有群均完成成员核对后可替换；不会改变单人会话选择")
        for button in (self.stop, self.append, self.replace, self.close_button):
            footer.addWidget(button)
        layout.addLayout(footer)
        self.lookup.clicked.connect(self.resolve)
        self.query.returnPressed.connect(self.resolve)
        self.query.textEdited.connect(self.invalidate_candidates)
        self.contacts.currentIndexChanged.connect(self.invalidate_result)
        self.scan.clicked.connect(self.find_groups)
        self.stop.clicked.connect(self.cancelled.emit)
        self.close_button.clicked.connect(self.reject)
        self.append.clicked.connect(lambda: self.apply("append"))
        self.replace.clicked.connect(lambda: self.apply("replace"))
        self.update_controls()

    def invalidate_candidates(self, *_):
        self.contacts.clear()
        self.invalidate_result()

    def invalidate_result(self, *_):
        self.result = None
        self.model.set_result({"matches": set(), "nonmatches": set(), "unknown": {}})
        self.update_controls()

    def update_controls(self):
        for widget in (self.query, self.contacts, self.force):
            widget.setEnabled(not self.busy)
        self.lookup.setEnabled(not self.busy)
        self.scan.setEnabled(not self.busy and self.contacts.currentData() is not None)
        self.stop.setEnabled(self.busy)
        self.append.setEnabled(not self.busy and bool(self.result and self.result["matches"]))
        self.replace.setEnabled(not self.busy and bool(self.result and self.result["matches"] and not self.result["unknown"]))

    def start(self, request):
        self.busy = True
        self.invalidate_result()
        self.status.setText("正在读取联系人" if request["operation"] == "resolve" else f"正在核对 {len(self.sessions)} 个群")
        self.requested.emit(request)

    def resolve(self):
        if not self.busy and self.query.text().strip():
            self.contacts.clear()
            self.start({"operation": "resolve", "query": self.query.text()})

    def find_groups(self):
        if not self.busy and self.contacts.currentData():
            self.start({"operation": "groups", "person": self.contacts.currentData(),
                        "ids": [row["id"] for row in self.sessions], "force": self.force.isChecked()})

    def receive(self, result):
        self.busy = False
        if result.get("error"):
            self.status.setText(result["error"])
        elif result["operation"] == "resolve":
            self.contacts.clear()
            for row in result["value"]:
                self.contacts.addItem(f"{row['name']} ({row['id']})", row["id"])
            self.contacts.setCurrentIndex(-1)
            self.status.setText(f"找到 {len(result['value'])} 个候选" + (" · 请选择联系人" if result["value"] else ""))
        else:
            self.result = result["value"]
            self.model.set_result(self.result)
            times = self.result["timestamps"].values()
            stamp = datetime.fromtimestamp(min(times)).strftime("%H:%M:%S") if times else "--"
            self.status.setText(f"匹配 {len(self.result['matches'])} · 未匹配 {len(self.result['nonmatches'])} · 未确认 {len(self.result['unknown'])}"
                                f" · 缓存 {self.result['cached']} · 最早核对 {stamp}" + (" · 已取消" if self.result["state"] == "cancelled" else ""))
        self.update_controls()

    def apply(self, mode):
        if self.busy or not self.result or not self.result["matches"] or (mode == "replace" and self.result["unknown"]):
            return
        self.apply_matches.emit(self.result["matches"], mode)
        self.accept()

    def reject(self):
        if self.busy:
            self.cancelled.emit()
        super().reject()
