from __future__ import annotations

import sqlite3
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSplitter, QStyle, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from .chats import ChatStore
from .models import SearchResult
from .ui_formatting import display_time as _display_time


class SourceHistoryDialog(QDialog):
    PAGE_SIZE = 25

    def __init__(self, store: ChatStore, result: SearchResult, parent=None):
        super().__init__(parent)
        self.store = store
        self.file_id = result.file_id
        self.query = replace(result.source_query) if result.source_query else None
        self.original_attachment = result.attachment_id
        self.cursors = [None]
        self.next_cursor = None
        self.rows = []
        self.setWindowTitle("聊天来源记录")
        self.resize(900, 660)
        self.setMinimumSize(640, 480)
        layout = QVBoxLayout(self)
        name = QLabel(result.name)
        name.setWordWrap(True)
        name.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(name)
        filters = []
        if self.query:
            for key, label in (("chat_platform", "平台"), ("conversation", "会话"), ("sender", "发送者")):
                if value := getattr(self.query, key):
                    filters.append(f"{label}：{value}")
            chat_dates = (self.query.scope in {"chat_files", "messages"}
                          or self.query.chat_platform or self.query.conversation or self.query.sender)
            if chat_dates:
                if self.query.modified_after:
                    filters.append("发送时间 >= " + self.query.modified_after)
                if self.query.modified_before:
                    filters.append("发送时间 < " + self.query.modified_before)
        self.filter_summary = QLineEdit("；".join(filters) or "关联此文件的全部已索引来源")
        self.filter_summary.setReadOnly(True)
        self.filter_summary.setToolTip(self.filter_summary.text())
        self.filter_summary.setCursorPosition(0)
        layout.addWidget(self.filter_summary)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索会话或发送者")
        self.search.setMaxLength(200)
        layout.addWidget(self.search)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["发送时间（本地）", "会话", "发送者", "账号"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 168), (2, 90), (3, 124)):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(column, width)
        self.table.setMinimumHeight(120)
        splitter.addWidget(self.table)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(100)
        splitter.addWidget(self.preview)
        splitter.setSizes([260, 220])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        footer = QHBoxLayout()
        self.previous = self._icon_button(QStyle.StandardPixmap.SP_ArrowBack, "上一页")
        self.next = self._icon_button(QStyle.StandardPixmap.SP_ArrowForward, "下一页")
        footer.addWidget(self.previous)
        footer.addWidget(self.next)
        self.status = QLabel()
        footer.addWidget(self.status, 1)
        self.copy_button = QPushButton("复制来源")
        self.copy_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.copy_button.setAutoDefault(False)
        footer.addWidget(self.copy_button)
        close = QPushButton("关闭")
        close.setAutoDefault(False)
        close.clicked.connect(self.reject)
        footer.addWidget(close)
        layout.addLayout(footer)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.reset_pages)
        self.search.textChanged.connect(self.schedule_filter)
        self.search.returnPressed.connect(self.reset_pages)
        self.previous.clicked.connect(self.previous_page)
        self.next.clicked.connect(self.next_page)
        self.table.itemSelectionChanged.connect(self.show_source)
        self.copy_button.clicked.connect(self.copy_source)
        self.load_page()

    def _icon_button(self, icon, name):
        button = QPushButton()
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(name)
        button.setAccessibleName(name)
        button.setFixedSize(32, 30)
        button.setAutoDefault(False)
        return button

    def schedule_filter(self):
        self.previous.setEnabled(False)
        self.next.setEnabled(False)
        self.preview.clear()
        self.copy_button.setEnabled(False)
        self.table.setEnabled(False)
        self.timer.start()

    def reset_pages(self):
        self.timer.stop()
        self.cursors = [None]
        self.load_page()

    def load_page(self):
        self.rows = []
        self.next_cursor = None
        self.table.setRowCount(0)
        self.preview.clear()
        self.copy_button.setEnabled(False)
        try:
            self.rows, self.next_cursor = self.store.file_sources(
                self.file_id, self.query, cursor=self.cursors[-1],
                limit=self.PAGE_SIZE, text=self.search.text(),
            )
        except (sqlite3.Error, ValueError):
            self.status.setText("来源读取失败")
        else:
            self.table.setRowCount(len(self.rows))
            for index, row in enumerate(self.rows):
                for col, key in enumerate(("sent_at", "conversation", "sender", "account")):
                    text = row[key] or "未知"
                    shown = _display_time(row["sent_at"]) if key == "sent_at" else text
                    item = QTableWidgetItem(shown)
                    item.setToolTip(text)
                    if col == 0 and row["attachment_id"] == self.original_attachment:
                        item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
                        item.setToolTip(text + "\n当前结果来源")
                    self.table.setItem(index, col, item)
            self.status.setText(f"第 {len(self.cursors)} 页 · {len(self.rows)} 条" if self.rows else "没有符合条件的来源")
            if self.rows:
                selected = next((i for i, row in enumerate(self.rows)
                                 if row["attachment_id"] == self.original_attachment), 0)
                self.table.selectRow(selected)
        self.table.setEnabled(True)
        self.previous.setEnabled(len(self.cursors) > 1)
        self.next.setEnabled(self.next_cursor is not None)

    def next_page(self):
        if self.next_cursor is not None:
            self.cursors.append(self.next_cursor)
            self.load_page()

    def previous_page(self):
        if len(self.cursors) > 1:
            self.cursors.pop()
            self.load_page()

    def show_source(self):
        self.preview.clear()
        self.copy_button.setEnabled(False)
        index = self.table.currentRow()
        if not 0 <= index < len(self.rows):
            return
        try:
            text = self.store.attachment_context(
                self.rows[index]["attachment_id"], file_id=self.file_id, query=self.query,
            )
        except (sqlite3.Error, ValueError):
            self.status.setText("来源读取失败")
            return
        self.preview.setPlainText(text or "该来源已更新或不再关联此文件")
        self.copy_button.setEnabled(bool(text))

    def copy_source(self):
        # Revalidate the association before copying after a background index update.
        self.show_source()
        if self.copy_button.isEnabled():
            QApplication.clipboard().setText(self.preview.toPlainText())

    def done(self, result):
        self.timer.stop()
        super().done(result)
