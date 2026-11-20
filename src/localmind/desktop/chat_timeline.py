"""Native, read-only message bubbles backed by bounded local snapshots."""

from threading import Event
import sqlite3
import weakref

from PySide6.QtCore import QAbstractListModel, QEvent, QModelIndex, QObject, QRectF, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QKeySequence, QTextDocument, QTextOption
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QHBoxLayout, QListView, QMenu,
    QPlainTextEdit, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from ..chat_timeline import TimelineError, discussions, verify_message, window
from ..ui_formatting import display_time, plain_tooltip
from .widgets import ElidedLabel, button, icon, label


class TimelineWorker(QObject):
    finished = Signal(int, object, str)

    @Slot(object)
    def run(self, request):
        generation, kind, path, result, options, cancel = request
        if cancel.is_set():
            return
        try:
            data = (discussions(path, result, **options, cancelled=cancel.is_set) if kind == "groups" else
                    window(path, result, **options, cancelled=cancel.is_set))
            error = ""
        except TimelineError as exc:
            data, error = None, str(exc)
        except Exception:
            data, error = None, "聊天记录读取失败，可刷新重试"
        if not cancel.is_set():
            self.finished.emit(generation, data, error)


class MessageModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return row
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return f"{row['sender']} {row.get('time_text') or display_time(row['sent_at'])}\n{row['body']}"

    def replace(self, rows):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()


class BubbleDelegate(QStyledItemDelegate):
    attachmentRequested = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.cache = {}

    def layout(self, row, width):
        key = (row["id"], width, row["body"], row.get("display_body"), row.get("quote_body"), row.get("quote_sender"), row["forwarded"],
               row.get("forward_json"), row.get("archive_child"),
               tuple((a["id"], a["name"]) for a in row["attachments"]))
        if key in self.cache:
            return self.cache[key]
        body = row["display_body"] if row.get("display_body") is not None else row["body"]
        body = body or "[非文本消息]"
        if row["forwarded"]:
            children = row.get("forward_items")
            if children:
                roots = [child for child in children if len(child.path) == 1]
                preview = "\n".join((child.sender or "未知发送者") + "：" + " ".join(child.body.split())[:60] for child in roots[:2])
                body = f"合并转发 · {len(roots)} 条记录\n" + preview
            else:
                body = ("嵌套转发记录\n" if row.get("archive_child") else "合并转发 · 归档节选\n") + body
        if row.get("quote_body"):
            body += f"\n\n引用 · {row.get('quote_sender') or '未知发送者'}\n{row['quote_body']}"
        if len(body) > 360:
            body = body[:360] + "\n…"
        doc = QTextDocument()
        font = QFont("Microsoft YaHei")
        font.setPixelSize(16)
        doc.setDefaultFont(font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        doc.setDefaultTextOption(option)
        doc.setDocumentMargin(0)
        doc.setPlainText(body)
        doc.setTextWidth(max(90, min(560, width * .78) - 24))
        value = doc, min(doc.size().height(), 650)
        if len(self.cache) > 80:
            self.cache.clear()
        self.cache[key] = value
        return value

    def sizeHint(self, option, index):  # noqa: N802
        row = index.data(Qt.ItemDataRole.UserRole)
        _, height = self.layout(row, self.parent().viewport().width())
        return QSize(0, int(height + self.files_height(row) + 77 + (30 if row["gap_before"] else 0)
                            + (28 if row.get("day_heading") else 0)))

    @staticmethod
    def files_height(row):
        return min(3, len(row["attachments"])) * 50 + (22 if len(row["attachments"]) > 3 else 0)

    def file_rects(self, rect, row):
        doc, height = self.layout(row, rect.width())
        width = doc.textWidth() + 24
        x = rect.right() - 8 - width if row.get("from_me") == 1 else rect.x() + 8
        y = rect.y() + 31 + height + 18 + (30 if row["gap_before"] else 0) + (28 if row.get("day_heading") else 0)
        return [QRectF(x + 10, y + i * 50, width - 20, 48) for i, _ in enumerate(row["attachments"][:3])]

    def editorEvent(self, event, model, option, index):  # noqa: N802
        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            row = index.data(Qt.ItemDataRole.UserRole)
            for rect, att in zip(self.file_rects(option.rect, row), row["attachments"]):
                if rect.contains(event.position()):
                    self.attachmentRequested.emit(row, att)
                    return True
        return super().editorEvent(event, model, option, index)

    def paint(self, painter, option, index):
        row = index.data(Qt.ItemDataRole.UserRole)
        doc, height = self.layout(row, option.rect.width())
        painter.save()
        box = QRectF(option.rect).adjusted(8, 0, -8, 0)
        own = row.get("from_me") == 1
        font = QFont("Microsoft YaHei")
        font.setPixelSize(13)
        painter.setFont(font)
        painter.setPen(QColor("#7b8782"))
        if row.get("day_heading"):
            painter.drawText(box, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, row["day_heading"])
            box.adjust(0, 28, 0, 0)
        if row["gap_before"]:
            painter.drawText(box.adjusted(0, 0, 0, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
                             "··· 中间有其它已索引消息 ···")
            box.adjust(0, 30, 0, 0)
        suffix = (" · 转发原记录" if row.get("archive_child") else " · 命中" if row["hit"]
                  else " · 文件来源" if row["source_anchor"] else " · 上下文")
        text = (row["sender"] or "未知发送者") + " · " + (row.get("time_text") or display_time(row["sent_at"])) + suffix
        text = painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(box.width()))
        painter.drawText(QRectF(box.x(), box.y() + 5, box.width(), 23),
                         Qt.AlignmentFlag.AlignRight if own else Qt.AlignmentFlag.AlignLeft, text)
        width = doc.textWidth() + 24
        bubble = QRectF(box.right() - width if own else box.x(), box.y() + 31, width, height + 24 + self.files_height(row))
        painter.setPen(QColor("#27937a") if row["hit"] or option.state & QStyle.StateFlag.State_Selected else QColor("#dce4df"))
        painter.setBrush(QColor("#dff1e8") if own else QColor("#ffffff"))
        painter.drawRoundedRect(bubble, 6, 6)
        painter.save()
        painter.setClipRect(bubble.adjusted(10, 10, -10, -10))
        painter.translate(bubble.x() + 12, bubble.y() + 12)
        doc.drawContents(painter, QRectF(0, 0, width - 24, height))
        painter.restore()
        for rect, att in zip(self.file_rects(option.rect, row), row["attachments"]):
            painter.setPen(QColor("#dce4df"))
            painter.drawLine(rect.topLeft(), rect.topRight())
            icon("file").paint(painter, int(rect.x()), int(rect.y() + 12), 22, 22)
            painter.setPen(QColor("#204c40"))
            font.setPixelSize(14)
            painter.setFont(font)
            title = painter.fontMetrics().elidedText(att["name"], Qt.TextElideMode.ElideRight, int(rect.width() - 30))
            painter.drawText(rect.adjusted(30, 3, 0, -23), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
            font.setPixelSize(12)
            painter.setFont(font)
            painter.setPen(QColor("#7b8782"))
            painter.drawText(rect.adjusted(30, 24, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             "已关联本地文件" if att["file_id"] else "待定位附件")
        if len(row["attachments"]) > 3:
            painter.drawText(bubble.adjusted(12, bubble.height() - 22, -12, 0), Qt.AlignmentFlag.AlignLeft,
                             f"另有 {len(row['attachments']) - 3} 个附件")
        painter.restore()


class MessageView(QListView):
    inspectRequested = Signal()
    copyRequested = Signal()

    def keyPressEvent(self, event):  # noqa: N802
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copyRequested.emit()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.inspectRequested.emit()
        else:
            super().keyPressEvent(event)


class ChatTimeline(QWidget):
    requested = Signal(object)
    groupsReady = Signal(object)
    attachmentRequested = Signal(object, object)
    remoteRequested = Signal(object, object, object, str)
    topicsRequested = Signal(object)

    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = str(db_path)
        self.result, self.peers, self.identity = None, (), None
        self.groups, self.payload = (), None
        self.generation, self.page = 0, 0
        self.cancel = Event()
        self.busy = False
        self.raw_mode = False
        self._restore = None
        self._window_options = {}
        self.rule_groups, self.topic_excluded = (), set()
        self.topic_epoch = 0
        self.topic_cancel = Event()
        self.topic_busy = False
        self.thread = QThread(self)
        self.worker = TimelineWorker()
        self.worker.moveToThread(self.thread)
        self.requested.connect(self.worker.run)
        self.worker.finished.connect(self._finished)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.selector = QComboBox()
        self.selector.setMinimumContentsLength(4)
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.selector.setAccessibleName("相关讨论")
        self.header = QHBoxLayout()
        self.header.setSpacing(2)
        self.header.addWidget(self.selector, 1)
        layout.addLayout(self.header)
        self.controls = QWidget()
        controls = QHBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        self.mode.addItem("相关片段", False)
        self.mode.addItem("附近消息", True)
        controls.addWidget(self.mode, 1)
        self.previous_hit = button("上一处命中", "chevron-left", lambda: self.jump_hit(-1), compact=True)
        self.next_hit = button("下一处命中", "chevron-right", lambda: self.jump_hit(1), compact=True)
        controls.addWidget(self.previous_hit)
        controls.addWidget(self.next_hit)
        controls.addWidget(button("聊天记录操作", "ellipsis", self.options_menu, compact=True))
        layout.addWidget(self.controls)
        self.notice = ElidedLabel("选择聊天来源")
        self.notice.setObjectName("mutedText")
        layout.addWidget(self.notice)
        self.view = MessageView()
        self.view.setMinimumHeight(0)
        self.view.setStyleSheet("QListView { background: #f1f5f3; border: none; } ")
        self.view.setWordWrap(True)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setLayoutMode(QListView.LayoutMode.Batched)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setBatchSize(20)
        self.model = MessageModel(self)
        self.view.setModel(self.model)
        delegate = BubbleDelegate(self.view)
        delegate.attachmentRequested.connect(self._attachment)
        self.view.setItemDelegate(delegate)
        self.view.setAccessibleName("只读聊天记录")
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._menu)
        self.view.doubleClicked.connect(self.inspect)
        self.view.inspectRequested.connect(self.inspect)
        self.view.copyRequested.connect(self.copy_message)
        layout.addWidget(self.view, 1)
        self.paging = QWidget()
        footer = QHBoxLayout(self.paging)
        footer.setContentsMargins(0, 0, 0, 0)
        self.previous_page = button("更早消息", "chevron-left", lambda: self.turn_page(-1), compact=True)
        self.next_page = button("更晚消息", "chevron-right", lambda: self.turn_page(1), compact=True)
        self.count = label("")
        footer.addWidget(self.previous_page)
        footer.addWidget(self.count, 1)
        footer.addWidget(self.next_page)
        layout.addWidget(self.paging)
        self.paging.hide()
        self.selector.currentIndexChanged.connect(self._select)
        self.mode.currentIndexChanged.connect(self._select)

    def set_raw_mode(self, enabled):
        self.raw_mode = enabled
        for widget in (self.view, self.controls, self.notice, self.selector):
            widget.setVisible(not enabled)
        self.paging.setVisible(not enabled and self.payload is not None and (self.payload.previous or self.payload.following))
        self.setMaximumHeight(42 if enabled else 16777215)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "notice"):
            self.notice.setVisible(not self.raw_mode and self.height() >= 260)
            self.mode.setToolTip(self.notice.toolTip())

    def set_result(self, result, peers=(), *, revision=None):
        identity = (id(result), tuple(id(r) for r in peers), revision)
        if identity == self.identity:
            return
        preserve = result is self.result
        self.preferred_message = result.chat_message_id if result is not self.result else None
        self.identity = identity
        self.result, self.peers = result, tuple(peers)
        self.refresh(preserve=preserve)

    def clear(self):
        self.reset_topics()
        self.cancel.set()
        self.generation += 1
        self.busy = False
        self.identity, self.result, self.payload = None, None, None
        self.groups = ()
        self._restore = None
        self._window_options = {}
        self.model.replace(())
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.blockSignals(False)
        self.previous_page.setEnabled(False)
        self.next_page.setEnabled(False)
        self.count.clear()
        self.paging.hide()
        self.notice.setText("暂无可读取的聊天记录")

    def _submit(self, kind, **options):
        if kind == "window":
            self._window_options = {k: v for k, v in options.items() if k != "discussion"}
        self.cancel.set()
        self.cancel = Event()
        self.generation += 1
        self.kind = kind
        self.busy = True
        self.payload = None
        self.model.replace(())
        self.notice.setText("正在读取本地记录…")
        self.previous_page.setEnabled(False)
        self.next_page.setEnabled(False)
        self.requested.emit((self.generation, kind, self.db_path, self.result, options, self.cancel))

    def refresh(self, *, preserve=False):
        self.reset_topics()
        self._restore = None
        if preserve and self.payload and self.selector.currentData():
            current = self.view.currentIndex().data(Qt.ItemDataRole.UserRole)
            self._restore = (self.selector.currentData().key, dict(self._window_options),
                             current["id"] if current else None, self.view.verticalScrollBar().value(),
                             tuple((r["id"], r["body"]) for r in self.model.rows))
        if self.result is not None:
            self._submit("groups", peers=self.peers)

    def _select(self):
        self._restore = None
        self.page = 0
        group = self.selector.currentData()
        self._load(anchor_id=group.anchor_id if group else None)

    def _load(self, **options):
        group = self.selector.currentData()
        if group is not None and self.result is not None:
            self._submit("window", discussion=group, nearby=self.mode.currentData(), page=self.page, **options)

    def _finished(self, generation, data, error, *, topics=False):
        if generation != self.generation:
            return
        self.busy = False
        if error:
            self.notice.setText(error)
            self.notice.setToolTip(error)
            return
        if self.kind == "groups":
            self.groups, more = data
            if not topics:
                self.rule_groups = self.groups
            selected = self.selector.currentData()
            self.selector.blockSignals(True)
            self.selector.clear()
            for group in self.groups:
                summary = " ".join(f"{group.conversation} · {group.title}".split())
                self.selector.addItem(summary, group)
                self.selector.setItemData(self.selector.count() - 1, plain_tooltip(summary), Qt.ItemDataRole.ToolTipRole)
            index = next((i for i, g in enumerate(self.groups)
                          if self.preferred_message and self.preferred_message in {r.message_id for r in g.refs}),
                         next((i for i, g in enumerate(self.groups) if selected and g.key == selected.key), 0))
            self.selector.setCurrentIndex(index if self.groups else -1)
            self.selector.blockSignals(False)
            self.selector.setToolTip(f"已整理 {len(self.groups)} 段讨论" + ("；来源未全部加载，可查看来源记录" if more else ""))
            self.groupsReady.emit(self.groups)
            if generation != self.generation:
                return
            if self.groups:
                group = self.selector.currentData()
                if self._restore and group.key == self._restore[0]:
                    self._submit("window", discussion=group, **self._restore[1])
                else:
                    self._select()
            else:
                self.notice.setText("暂无可读取的聊天来源")
            return
        self.payload = data
        self.page = data.page_index
        self.model.replace(data.rows)
        self.notice.setText(f"{len(data.rows)} 条 · {sum(r['hit'] for r in data.rows)} 条命中 · " + data.notice)
        self.notice.setToolTip(data.notice)
        self.mode.setToolTip(data.notice)
        self.view.setAccessibleDescription(data.notice)
        self.count.setText(f"{len(data.rows)} 条 · {sum(r['hit'] for r in data.rows)} 条命中")
        self.previous_page.setEnabled(data.previous)
        self.next_page.setEnabled(data.following)
        self.paging.setVisible(not self.raw_mode and (data.previous or data.following))
        target = next((i for i, row in enumerate(data.rows) if row["id"] == data.anchor_id), 0)
        restore, self._restore = self._restore, None
        if restore:
            target = next((i for i, row in enumerate(data.rows) if row["id"] == restore[2]), target)
        if data.rows:
            index = self.model.index(target)
            self.view.setCurrentIndex(index)
            self.view.scrollTo(index, QAbstractItemView.ScrollHint.PositionAtCenter)
            if restore and tuple((r["id"], r["body"]) for r in data.rows) == restore[4]:
                self.view.doItemsLayout()
                self.view.verticalScrollBar().setValue(restore[3])

    def turn_page(self, delta):
        if self.busy or self.payload is None:
            return
        if not (self.payload.previous if delta < 0 else self.payload.following):
            return
        cursor = self.payload.before_cursor if delta < 0 else self.payload.after_cursor
        self.page += delta
        self._load(cursor=cursor)

    def jump_hit(self, delta):
        group = self.selector.currentData()
        if self.busy or group is None:
            return
        ids = list(group.hit_ids) or list(dict.fromkeys(r.message_id for r in group.refs))
        selected = self.view.currentIndex().data(Qt.ItemDataRole.UserRole)
        current = selected["id"] if selected else None
        position = ids.index(current) if current in ids else (-1 if delta > 0 else 0)
        target = ids[(position + delta) % len(ids)]
        index = next((i for i, r in enumerate(self.model.rows) if r["id"] == target), None)
        if index is None:
            self.page = 0
            self._load(anchor_id=target)
        else:
            self.view.setCurrentIndex(self.model.index(index))
            self.view.scrollTo(self.view.currentIndex(), QAbstractItemView.ScrollHint.PositionAtCenter)

    def _selected(self):
        index = self.view.currentIndex().row()
        if not 0 <= index < len(self.model.rows):
            return None
        row = self.model.rows[index]
        try:
            # Recheck group identity and attachment associations before any action.
            window(self.db_path, self.result, self.selector.currentData())
            current = verify_message(self.db_path, self.selector.currentData(), row["id"], expected_body=row["body"])
            fields = ("external_id", "sender", "sent_at", "from_me", "display_body", "quote_body", "quote_sender", "forwarded", "forward_json")
            if any(current[field] != row[field] for field in fields):
                raise TimelineError("消息展示信息已变化")
        except (TimelineError, OSError, sqlite3.Error):
            self.refresh()
            return None
        return row

    def copy_message(self):
        row = self._selected()
        if row:
            QApplication.clipboard().setText(f"{row['sender']} · {display_time(row['sent_at'])}\n{row['body']}")

    def _menu(self, point):
        index = self.view.indexAt(point)
        if not index.isValid():
            return
        self.view.setCurrentIndex(index)
        menu = QMenu(self)
        menu.addAction("复制消息", self.copy_message)
        menu.addAction("查看完整消息", self.inspect)
        row = self.model.rows[index.row()]
        menu.addAction("从 Alpha 读取附近消息…", lambda: self.read_remote("around"))
        if row["forwarded"]:
            menu.addAction("从 Alpha 读取转发记录…", lambda: self.read_remote("forward"))
        for att in row["attachments"][:20]:
            menu.addAction("预览 / 定位：" + att["name"][:50], lambda a=att: self._attachment(row, a))
        menu.exec(self.view.viewport().mapToGlobal(point))

    def attachment_menu(self):
        index = self.view.currentIndex()
        if index.isValid():
            self._menu(self.view.visualRect(index).center())

    def options_menu(self):
        menu = QMenu(self)
        menu.addAction("展开聊天记录", self.expand)
        menu.addAction("刷新聊天记录", self.refresh)
        action = menu.addAction("按主题整理（实验）", self.organize_topics)
        action.setEnabled(not self.topic_busy and len(self.rule_groups) > 1)
        if self.topic_busy:
            menu.addAction("取消主题整理", self.topic_cancel.set)
        selected = self.selector.currentData()
        action = menu.addAction("拆分本次主题", self.split_topic)
        action.setEnabled(bool(selected and selected.components))
        menu.addAction("恢复引用 / 回复分组", self.restore_rules)
        menu.addSeparator()
        menu.addAction("从 Alpha 读取附近消息…", lambda: self.read_remote("around"))
        menu.addAction("查看完整消息", self.inspect)
        menu.addAction("复制消息", self.copy_message)
        menu.addAction("消息附件", self.attachment_menu)
        menu.exec(self.sender().mapToGlobal(self.sender().rect().bottomLeft()))

    def expand(self):
        if self.result is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("聊天记录")
        dialog.resize(760, 680)
        layout = QVBoxLayout(dialog)
        timeline = ChatTimeline(self.db_path, dialog)
        timeline.attachmentRequested.connect(self.attachmentRequested.emit)
        timeline.remoteRequested.connect(self.remoteRequested.emit)
        timeline.topicsRequested.connect(self.topicsRequested.emit)
        timeline.set_result(self.result, self.peers)
        selected = self.selector.currentData()
        if selected:
            timeline.preferred_message = selected.anchor_id
        layout.addWidget(timeline)
        try:
            dialog.exec()
        finally:
            timeline.shutdown()
            dialog.deleteLater()

    def inspect(self, *_):
        row = self._selected()
        if row is None:
            return
        if row["forwarded"] and row.get("forward_items"):
            self.inspect_forward(row)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("聊天记录 · 完整消息")
        dialog.resize(640, 520)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label(f"{row['sender']} · {display_time(row['sent_at'])}", wrap=True))
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("font-size: 16px;")
        text.setPlainText(row["body"])
        layout.addWidget(text, 1)
        if row["forwarded"]:
            layout.addWidget(button("从 Alpha 读取转发记录", "refresh-cw", lambda: self.read_remote("forward")))
        if row["attachments"]:
            files = QComboBox()
            files.setMinimumContentsLength(8)
            files.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            files.setAccessibleName("消息附件")
            for att in row["attachments"]:
                files.addItem(att["name"] + (" · 已关联" if att["file_id"] else " · 待定位"), att)
                files.setItemData(files.count() - 1, plain_tooltip(att["name"] + "\n" + att["origin"]), Qt.ItemDataRole.ToolTipRole)
            actions = QHBoxLayout()
            actions.addWidget(files, 1)
            actions.addWidget(button("预览 / 定位附件", "file", lambda: self._attachment(row, files.currentData()), compact=True))
            layout.addLayout(actions)
        layout.addWidget(button("关闭", "x", dialog.accept))
        dialog.exec()

    def _attachment(self, row, attachment):
        selected = self._selected()
        if selected is not None and selected["id"] == row["id"]:
            self.attachmentRequested.emit(row, attachment)

    def shutdown(self):
        self.topic_cancel.set()
        self.cancel.set()
        self.thread.quit()
        return self.thread.wait(3000)

    def read_remote(self, kind):
        row = self._selected()
        if row:
            self.remoteRequested.emit(self.result, self.selector.currentData(), row, kind)

    def inspect_forward(self, row):
        from .chat_transcript import TranscriptDialog, forward_rows
        group, result = self.selector.currentData(), self.result

        def guard():
            window(self.db_path, result, group)
            current = verify_message(self.db_path, group, row["id"], expected_body=row["body"])
            if current["forward_json"] != row["forward_json"] or current["external_id"] != row["external_id"]:
                raise TimelineError("转发记录已更新")
            return True

        dialog = TranscriptDialog("合并转发 · 原始子记录", forward_rows(row),
            notice="转发记录独立展示；发送者和时间来自转发原文，不代表外层会话。", guard=guard,
            attachment=lambda child, att: self.attachmentRequested.emit(row, att), parent=self)
        dialog.exec()
        dialog.deleteLater()

    def reset_topics(self):
        self.topic_cancel.set()
        self.topic_epoch += 1
        self.topic_busy = False
        self.topic_excluded.clear()
        self.rule_groups = ()

    def organize_topics(self):
        if self.topic_busy or len(self.rule_groups) < 2:
            return
        self.topic_cancel = Event()
        self.topic_busy = True
        self.notice.setText("正在本地整理主题…")
        self.topicsRequested.emit({"receiver": weakref.ref(self), "epoch": self.topic_epoch,
            "cancel": self.topic_cancel, "result": self.result, "groups": self.rule_groups,
            "excluded": tuple(self.topic_excluded)})

    def topics_finished(self, request, data, notice):
        if request["epoch"] != self.topic_epoch:
            return
        self.topic_busy = False
        if request["cancel"].is_set():
            self.notice.setText("主题整理已取消，保留原分组")
            return
        if data is not None:
            self.apply_groups(data)
        self.selector.setToolTip(notice)

    def apply_groups(self, groups):
        self.cancel.set()
        self.generation += 1
        self.kind = "groups"
        self._finished(self.generation, (tuple(groups), False), "", topics=True)

    def split_topic(self):
        group = self.selector.currentData()
        if group is None or not group.components:
            return
        self.topic_excluded.update(g.key for g in group.components)
        self.apply_groups(tuple(item for g in self.groups for item in (g.components if g.key == group.key else (g,))))

    def restore_rules(self):
        self.topic_cancel.set()
        self.topic_epoch += 1
        self.topic_busy = False
        if self.rule_groups:
            self.apply_groups(self.rule_groups)
