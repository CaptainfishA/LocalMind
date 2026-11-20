"""Isolated read-only transcripts for forward archives and bounded Alpha windows."""

from threading import Event

from PySide6.QtCore import QThread, QTimer, QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QAbstractItemView, QApplication, QDialog, QHBoxLayout, QPlainTextEdit, QVBoxLayout

from ..alpha_chat_context import AlphaChatContext, ContextCache
from ..chat_timeline import TimelineError
from ..weflow import SyncCancelled, WeFlowError
from .chat_timeline import BubbleDelegate, MessageModel, MessageView
from .widgets import button, label


def forward_rows(outer):
    attachments = {att["ordinal"]: att for att in outer["attachments"]}
    rows = []
    for child in outer["forward_items"]:
        attachment = attachments.get(child.attachment_ordinal)
        rows.append({"id": (outer["id"], child.path), "sender": child.sender,
            "sent_at": None, "time_text": child.time_text or "原时间未知", "body": child.body,
            "display_body": None, "from_me": None, "quote_body": "", "quote_sender": "",
            "forwarded": child.kind == "nested", "forward_items": (), "forward_path": child.path,
            "attachments": [attachment] if attachment is not None else [],
            "gap_before": False, "hit": False, "source_anchor": False, "day_heading": "",
            "archive_child": True})
    return tuple(rows)


class TranscriptDialog(QDialog):
    def __init__(self, title, rows=(), *, notice="", guard=lambda: True, attachment=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 680)
        self.guard, self.attachment = guard, attachment
        self.rows, self.branch = tuple(rows), ()
        layout = QVBoxLayout(self)
        self.toolbar = QHBoxLayout()
        self.back = button("返回上层转发", "chevron-left", self.go_back, compact=True)
        self.toolbar.addWidget(self.back)
        self.breadcrumb = label("")
        self.toolbar.addWidget(self.breadcrumb, 1)
        self.toolbar.addWidget(button("复制消息", "copy", self.copy_message, compact=True))
        self.toolbar.addWidget(button("查看完整消息", "maximize-2", self.inspect, compact=True))
        layout.addLayout(self.toolbar)
        self.notice = label(notice, wrap=True)
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.notice)
        self.view = MessageView()
        self.view.setAccessibleName("只读转发或上下文消息")
        self.view.setStyleSheet("QListView { background: #f1f5f3; border: none; }")
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setResizeMode(MessageView.ResizeMode.Adjust)
        self.view.setLayoutMode(MessageView.LayoutMode.Batched)
        self.view.setBatchSize(20)
        self.model = MessageModel(self)
        self.view.setModel(self.model)
        delegate = BubbleDelegate(self.view)
        self.view.setItemDelegate(delegate)
        delegate.attachmentRequested.connect(self.open_attachment)
        self.view.doubleClicked.connect(self.inspect)
        self.view.inspectRequested.connect(self.inspect)
        self.view.copyRequested.connect(self.copy_message)
        layout.addWidget(self.view, 1)
        self.footer = QHBoxLayout()
        self.footer.addStretch(1)
        self.footer.addWidget(button("关闭", "x", self.reject))
        layout.addLayout(self.footer)
        self.set_rows(rows)

    def set_rows(self, rows):
        self.rows, self.branch = tuple(rows), ()
        self.show_branch()

    def show_branch(self):
        nested = any("forward_path" in row for row in self.rows)
        rows = [r for r in self.rows if not nested or r.get("forward_path", ())[:-1] == self.branch]
        self.model.replace(rows)
        self.back.setVisible(bool(self.branch))
        self.breadcrumb.setText(("转发记录" + (" / " + ".".join(map(str, self.branch)) if self.branch else "")) if nested else "消息上下文")
        if rows:
            self.view.setCurrentIndex(self.model.index(0))

    def valid(self):
        try:
            if self.guard() is False:
                raise TimelineError("来源已变化")
            return True
        except Exception:
            self.rows = ()
            self.model.replace(())
            self.notice.setText("来源已更新或移除，请关闭窗口后重新打开。")
            return False

    def selected(self):
        return self.view.currentIndex().data(Qt.ItemDataRole.UserRole) if self.valid() else None

    def go_back(self):
        if self.valid():
            self.branch = self.branch[:-1]
            self.show_branch()

    def copy_message(self):
        row = self.selected()
        if row:
            QApplication.clipboard().setText(row["sender"] + "\n" + row["body"])

    def inspect(self, *_):
        row = self.selected()
        if row is None:
            return
        path = row.get("forward_path")
        if path and any(r.get("forward_path", ())[:-1] == path for r in self.rows):
            self.branch = path
            self.show_branch()
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("原始消息")
        dialog.resize(620, 480)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("font-size: 16px;")
        text.setPlainText(row["body"])
        layout.addWidget(text)
        layout.addWidget(button("关闭", "x", dialog.accept))
        dialog.exec()

    def open_attachment(self, row, attachment):
        if self.valid() and self.attachment and row in self.model.rows and attachment in row["attachments"]:
            self.attachment(row, attachment)


class ContextWorker(QObject):
    finished = Signal(object, str)

    def __init__(self, client, anchor, options, cancel):
        super().__init__()
        self.client, self.anchor, self.options, self.cancel = client, anchor, options, cancel
        self.cache = ContextCache()

    @Slot()
    def run(self):
        try:
            result = AlphaChatContext(self.client, cache=self.cache, cancelled=self.cancel.is_set).fetch(self.anchor, **self.options)
            error = ""
        except (WeFlowError, SyncCancelled) as exc:
            result, error = None, str(exc)
        except Exception:
            result, error = None, "Alpha 聊天读取失败；本地索引未改变"
        finally:
            self.client._token = ""
            self.cache.clear()
        self.finished.emit(result, error)


class AlphaContextDialog(TranscriptDialog):
    def __init__(self, client, anchor, *, options, guard, enrich=lambda value: value, attachment=None, parent=None):
        super().__init__("Alpha · 只读聊天窗口", notice="正在读取选定消息上下文…", guard=guard, attachment=attachment, parent=parent)
        self.enrich = enrich
        self.cancel = Event()
        self.closing = False
        self.thread = QThread(self)
        self.worker = ContextWorker(client, anchor, options, self.cancel)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.loaded)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    def loaded(self, result, error):
        if self.cancel.is_set() or not self.valid():
            return
        if error:
            self.notice.setText(error)
        else:
            try:
                result = self.enrich(result)
                self.set_rows(result.rows)
                self.notice.setText(result.notice + "；仅已核对本地身份的附件可定位")
            except Exception:
                self.notice.setText("本地来源核对未通过，请关闭后重新读取。")

    def done(self, result):
        self.cancel.set()
        if self.thread.isRunning():
            self.closing = True
            self.notice.setText("正在结束读取并释放连接…")
            QTimer.singleShot(100, lambda: self.done(result))
            return
        self.rows = ()
        self.model.replace(())
        super().done(result)

    def closeEvent(self, event):  # noqa: N802
        if self.thread.isRunning():
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)
