"""Asynchronous preview pane with latest-selection wins and disposable parser processes."""

import base64
import json
from pathlib import Path
import sys

from PySide6.QtCore import QPoint, QPointF, QProcess, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap, QPolygonF, QTextCursor
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QHeaderView, QPlainTextEdit, QScrollArea, QSizePolicy,
    QSpinBox, QStackedWidget, QTableWidgetItem, QVBoxLayout, QWidget, QTextEdit, QAbstractItemView,
    QAbstractSpinBox, QStyledItemDelegate,
)

from .widgets import DataTable, ElidedLabel, button, icon, label
from ..ui_formatting import plain_tooltip


class PreviewCellDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if not index.data(Qt.ItemDataRole.UserRole):
            return super().paint(painter, option, index)
        # The shared item stylesheet otherwise overrides BackgroundRole highlights.
        painter.save()
        painter.setClipRect(option.rect)
        painter.fillRect(option.rect, QColor("#fff2a8"))
        painter.setPen(QColor("#273730"))
        painter.setFont(option.font)
        rect = option.rect.adjusted(6, 3, -6, -3)
        value = painter.fontMetrics().elidedText(str(index.data() or ""), Qt.TextElideMode.ElideRight, rect.width())
        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, value)
        painter.restore()


class PreviewImage(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap()
        self.focus_polygons = []
        self.setMinimumSize(40, 40)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def sizeHint(self):  # noqa: N802
        return QSize(300, 170)

    def image_rect(self):
        if not self.pixmap.isNull():
            size = self.pixmap.size().scaled(self.size() - QSize(12, 12), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect(self.rect())
            target.setSize(size)
            target.moveCenter(self.rect().center())
            return target
        return QRect()

    def mapped_polygon(self, points):
        target = self.image_rect()
        return QPolygonF([QPointF(target.x() + x * target.width(), target.y() + y * target.height())
                          for x, y in points])

    def focus_rect(self):
        return (self.mapped_polygon(self.focus_polygons[0]).boundingRect().intersected(QRectF(self.image_rect()))
                if self.focus_polygons else QRectF())

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#edf0f2"))
        if not self.pixmap.isNull():
            target = self.image_rect()
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(target, self.pixmap)
            painter.setClipRect(target)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 226, 96, 130))
            for points in self.focus_polygons:
                painter.drawPolygon(self.mapped_polygon(points))
        painter.end()


class PreviewScroll(QScrollArea):
    """Fit raster pages to the available width, retaining readable vertical detail."""

    def __init__(self, canvas):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidget(canvas)
        self._focus_timer = QTimer(self)
        self._focus_timer.setSingleShot(True)
        self._focus_timer.timeout.connect(self._center_focus)
        self.verticalScrollBar().sliderPressed.connect(self._focus_timer.stop)
        self.verticalScrollBar().actionTriggered.connect(self._focus_timer.stop)

    def reset_focus(self):
        self._focus_timer.stop()
        self.widget().focus_polygons = []

    def focus_first(self):
        self._focus_timer.start(0)

    def _center_focus(self):
        rect = self.widget().focus_rect()
        if not rect.isEmpty():
            self.verticalScrollBar().setValue(round(rect.center().y() - self.viewport().height() / 3))

    def fit_width(self):
        canvas = self.widget()
        width = max(40, self.viewport().width())
        pixels = canvas.pixmap
        height = (round((width - 12) * pixels.height() / pixels.width()) + 12
                  if not pixels.isNull() else 40)
        canvas.resize(width, max(height, self.viewport().height()))

    def resizeEvent(self, event):  # noqa: N802
        focus = self.widget().focus_rect()
        previous = event.oldSize() if event.oldSize().isValid() else self.viewport().size()
        visible = QRectF(0, self.verticalScrollBar().value(), previous.width(), previous.height()).intersects(focus)
        super().resizeEvent(event)
        self.fit_width()
        if visible:
            self.focus_first()

    def wheelEvent(self, event):  # noqa: N802
        self._focus_timer.stop()
        super().wheelEvent(event)


class PreviewText(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self._keep_visible = QTimer(self)
        self._keep_visible.setSingleShot(True)
        self._keep_visible.timeout.connect(self.ensureCursorVisible)

    def resizeEvent(self, event):  # noqa: N802
        # Reflow can push a visible match below the pane; preserve it, not a user-scrolled-away cursor.
        visible = bool(self.extraSelections()) and QRect(QPoint(), event.oldSize()).intersects(self.cursorRect())
        super().resizeEvent(event)
        if visible:
            self._keep_visible.start(0)


class FilePreview(QWidget):
    collapsedChanged = Signal(bool)
    OUTPUT_LIMIT = 16 * 1024 * 1024
    TIMEOUT_MS = 10000

    def __init__(self, parent=None, *, expanded=False):
        super().__init__(parent)
        self.setObjectName("filePreview")
        self._target = None
        self._generation = 0
        self._process = None
        self._retired = []
        self._payload = None
        self._page = 0
        self._sheet = ""
        self._buffer = bytearray()
        self._expanded = expanded
        self._focus_mode = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._load)
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._timed_out)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(2)
        self.caption = ElidedLabel("文件预览")
        self.caption.setObjectName("previewCaption")
        self.collapse = button("收起文件预览", "chevron-down", self._toggle_collapsed, compact=True)
        self.collapse.setCheckable(True)
        self.collapse.setVisible(not expanded)
        self.previous = button("上一页", "chevron-left", lambda: self.go_page(self._page - 1), compact=True)
        self.next = button("下一页", "chevron-right", lambda: self.go_page(self._page + 1), compact=True)
        self.page_number = QSpinBox()
        self.page_number.setRange(1, 1)
        self.page_number.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.page_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_number.setFixedWidth(58)
        self.page_number.setAccessibleName("预览页码")
        self.page_number.editingFinished.connect(lambda: self.go_page(self.page_number.value() - 1))
        self.page_total = label("/ 1")
        self.refresh = button("刷新预览", "refresh-cw", self.reload, compact=True)
        self.enlarge = button("放大预览", "maximize-2", self.show_large, compact=True)
        self.enlarge.setVisible(not expanded)
        self.focus_button = button("定位索引片段", "scan-search", self._toggle_focus, compact=True)
        self.focus_button.setCheckable(True)
        self.focus_notice = ElidedLabel()
        self.focus_notice.setObjectName("previewNote")
        bar.addWidget(self.collapse)
        bar.addWidget(self.caption, 1)
        for control in (self.refresh, self.enlarge):
            bar.addWidget(control)
        layout.addLayout(bar)
        self.navigation = QWidget()
        navigation = QHBoxLayout(self.navigation)
        navigation.setContentsMargins(0, 0, 0, 0)
        navigation.setSpacing(2)
        navigation.addWidget(self.focus_button)
        navigation.addWidget(self.focus_notice, 1)
        for control in (self.previous, self.page_number, self.page_total, self.next):
            navigation.addWidget(control)
        layout.addWidget(self.navigation)
        self.stack = QStackedWidget()
        self.message = label("选择一个文件", wrap=True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setMinimumHeight(35)
        self.canvas = PreviewImage()
        self.image_scroll = PreviewScroll(self.canvas)
        self.text = PreviewText()
        self.text.setReadOnly(True)
        self.text.setObjectName("filePreviewText")
        self.text.setMinimumHeight(40)
        self.table = DataTable()
        self.table.setMinimumHeight(40)
        self.table.setObjectName("filePreviewTable")
        self.table.setItemDelegate(PreviewCellDelegate(self.table))
        self.table.verticalHeader().show()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.verticalHeader().setDefaultSectionSize(34)
        for view in (self.message, self.image_scroll, self.text, self.table):
            self.stack.addWidget(view)
        layout.addWidget(self.stack, 1)
        self.note = ElidedLabel()
        self.note.setObjectName("previewNote")
        layout.addWidget(self.note)
        self.setMinimumHeight(130)
        self.clear()

    def _toggle_collapsed(self):
        collapsed = self.collapse.isChecked()
        self.setMinimumHeight(44 if collapsed else 130)
        self.setMaximumHeight(44 if collapsed else 16777215)
        self.stack.setVisible(not collapsed)
        self.collapse.setIcon(icon("chevron-right" if collapsed else "chevron-down"))
        self.collapse.setToolTip("展开文件预览" if collapsed else "收起文件预览")
        self.collapse.setAccessibleName(self.collapse.toolTip())
        self._update_chrome()
        self.collapsedChanged.emit(collapsed)

    def _update_chrome(self):
        payload = self._payload or {}
        collapsed = self.collapse.isChecked()
        for control in (self.previous, self.next, self.page_number, self.page_total):
            control.setVisible(not collapsed and payload.get("pages", 1) > 1)
        available = self._focus_available()
        self.navigation.setVisible(not collapsed and (available or payload.get("pages", 1) > 1))
        self.focus_button.setVisible(available)
        self.focus_notice.setVisible(available or payload.get("pages", 1) > 1)
        status = payload.get("focus_status")
        pdf = self._target and Path(self._target[0]).suffix.lower() == ".pdf"
        text = "文件开头" if not self._focus_mode else "片段定位"
        if pdf and not self._focus_mode:
            text = "原页预览"
        if self._focus_mode and status == "matched":
            cells = payload.get("focus_cells", [])
            text = (f"第 {payload['row_start'] + cells[0][0]} 行附近" if cells else
                    "本页片段已标记" if pdf else "索引片段附近")
        elif self._focus_mode and status == "not_found":
            text = "本页未找到原片段" if pdf else "未找到原片段"
        elif self._focus_mode and status in {"no_text", "limited", "unavailable"}:
            text = {"no_text": "本页无可定位文字", "limited": "本页已跳过定位", "unavailable": "本页暂无法定位"}[status]
        self.focus_notice.setText(text if available else "")
        hint = ("\n只核对当前页的完整索引原文，忽略空白差异；标记匹配区域并定位首处，不跨页搜索，也不修改 PDF。"
                if pdf else "\n定位核对当前文件中的索引原文，忽略空白差异；不是再次语义搜索，重复片段取首次匹配。")
        self.focus_notice.setToolTip(plain_tooltip(payload.get("note", "") + hint))
        self.focus_button.setToolTip(("显示原页（不标记）" if self._focus_mode else "返回索引页并定位片段") if pdf else
                                    "从文件开头预览" if self._focus_mode else "定位索引片段")
        self.focus_button.setAccessibleName(self.focus_button.toolTip())
        self.note.setVisible(not collapsed and self.height() >= 180 and bool(payload.get("note")))

    def _focus_available(self):
        return bool(self._target and self._target[5] and
                    Path(self._target[0]).suffix.lower() in {".txt", ".md", ".docx", ".xlsx", ".csv", ".pdf"})

    def _toggle_focus(self):
        if not self._focus_available():
            return
        self._focus_mode = self.focus_button.isChecked()
        if self._focus_mode or Path(self._target[0]).suffix.lower() != ".pdf":
            self._page, self._sheet = self._target[1], self._target[2]
        self.reload()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "note"):
            self._update_chrome()

    def set_target(self, path, *, page=0, sheet="", revision=0, expected_hash="", anchor=""):
        target = (str(path), max(0, (page or 1) - 1), sheet or "", revision, expected_hash, anchor) if path else None
        if target == self._target:
            return
        self.clear("正在读取文件…" if target else "尚未定位到本地文件")
        self._target = target
        if target:
            self._page, self._sheet = target[1], target[2]
            self._focus_mode = self._focus_available()
            self.focus_button.setChecked(self._focus_mode)
            self._update_chrome()
            self.refresh.setEnabled(True)
            self._debounce.start()

    def _cancel_process(self):
        self._timeout.stop()
        process, self._process = self._process, None
        if process is not None:
            if process.state() != QProcess.ProcessState.NotRunning:
                self._retired.append(process)
                process.finished.connect(lambda *_: self._dispose(process))
                process.kill()
            else:
                process.deleteLater()

    def _dispose(self, process):
        if process in self._retired:
            self._retired.remove(process)
        process.deleteLater()

    def clear(self, message="选择一个文件"):
        self._generation += 1
        self._debounce.stop()
        self._cancel_process()
        self._target = None
        self._focus_mode = False
        self.focus_button.setChecked(False)
        self._payload = None
        self._update_chrome()
        self._buffer.clear()
        self.image_scroll.reset_focus()
        self.canvas.pixmap = QPixmap()
        self.canvas.update()
        self.text.clear()
        self.text.setExtraSelections([])
        self.table.clearContents()
        self.table.setRowCount(0)
        self.message.setText(message)
        self.stack.setCurrentWidget(self.message)
        self.caption.setText("文件预览")
        self.note.clear()
        self.note.hide()
        for control in (self.previous, self.next, self.page_number, self.page_total):
            control.hide()
        self.enlarge.setEnabled(False)
        self.refresh.setEnabled(False)

    def go_page(self, page):
        if not self._target or not self._payload or page == self._page:
            return
        if 0 <= page < self._payload.get("pages", 1):
            self._page, self._sheet = page, ""
            self.reload()

    def reload(self):
        if not self._target:
            return
        self._generation += 1
        self._cancel_process()
        self._payload = None
        self._update_chrome()
        self.image_scroll.reset_focus()
        self.canvas.pixmap = QPixmap()
        self.text.clear()
        self.text.setExtraSelections([])
        self.table.setRowCount(0)
        self.message.setText("正在读取文件…")
        self.stack.setCurrentWidget(self.message)
        self.enlarge.setEnabled(False)
        self._debounce.start()

    def _load(self):
        if not self._target:
            return
        self._buffer.clear()
        generation = self._generation
        process = QProcess(self)
        self._process = process
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe":
            executable = executable.with_name("python.exe")
        request = json.dumps({"path": self._target[0], "page": self._page, "sheet": self._sheet,
                              "expected_hash": self._target[4],
                              "anchor": self._target[5] if self._focus_mode else ""}, ensure_ascii=False).encode("utf-8")
        def started():
            if self._process is process:
                process.write(request)
                process.closeWriteChannel()
        process.started.connect(started)
        process.readyReadStandardOutput.connect(lambda: self._read(process, generation))
        process.readyReadStandardError.connect(lambda: process.readAllStandardError())
        process.finished.connect(lambda code, *_: self._finished(process, generation, code))
        process.errorOccurred.connect(lambda error: self._failed_start(process, generation, error))
        self._timeout.start(self.TIMEOUT_MS)
        process.start(str(executable), ["-X", "utf8", "-m", "localmind.preview_worker"])

    def _read(self, process, generation):
        data = bytes(process.readAllStandardOutput())
        if self._process is process and generation == self._generation:
            self._buffer.extend(data)
            if len(self._buffer) > self.OUTPUT_LIMIT:
                self._error("预览内容过大，请使用系统程序打开")

    def _finished(self, process, generation, code):
        if self._process is not process or generation != self._generation:
            return
        self._read(process, generation)
        if self._process is not process:
            return
        self._timeout.stop()
        self._process = None
        process.deleteLater()
        try:
            payload = json.loads(self._buffer)
            if code or not isinstance(payload, dict):
                raise ValueError()
        except (ValueError, UnicodeError):
            self._error("预览未完成，可重试或使用系统程序打开")
            return
        self._buffer.clear()
        self._display(payload)

    def _failed_start(self, process, generation, error):
        if self._process is process and generation == self._generation and error == QProcess.ProcessError.FailedToStart:
            self._error("无法启动本地预览，请使用系统程序打开")

    def _timed_out(self):
        self._error("预览超过 10 秒，已停止；可使用系统程序打开")

    def _error(self, message):
        self._generation += 1
        self._cancel_process()
        self._payload = None
        self.image_scroll.reset_focus()
        self._update_chrome()
        self._buffer.clear()
        self.message.setText(message)
        self.stack.setCurrentWidget(self.message)
        self.enlarge.setEnabled(False)
        for control in (self.previous, self.next, self.page_number, self.page_total):
            control.hide()

    def _display(self, payload):
        if payload.get("error"):
            self._error(payload["error"])
            return
        self.image_scroll.reset_focus()
        self._payload = payload
        self._page = payload["page"]
        self.caption.setText(payload["caption"])
        self.caption.setToolTip(plain_tooltip(payload["caption"] + "\n" + payload.get("note", "")))
        self.note.setText(payload.get("note", ""))
        self.note.setToolTip(plain_tooltip(payload.get("note", "")))
        self.note.setVisible(bool(payload.get("note")))
        if payload["kind"] == "image":
            pixels = QPixmap()
            pixels.loadFromData(base64.b64decode(payload["image"]), "PNG")
            self.canvas.pixmap = pixels
            self.canvas.focus_polygons = payload.get("focus_polygons", [])
            self.canvas.update()
            self.stack.setCurrentWidget(self.image_scroll)
            self.image_scroll.fit_width()
            self.image_scroll.verticalScrollBar().setValue(0)
            if self.canvas.focus_polygons:
                self.image_scroll.focus_first()
        elif payload["kind"] == "table":
            rows = payload["rows"]
            self.table.clearContents()
            self.table.setColumnCount(max(map(len, rows), default=1))
            self.table.setRowCount(len(rows))
            self.table.setHorizontalHeaderLabels([str(n + payload.get("column_start", 1))
                                                 for n in range(self.table.columnCount())])
            self.table.setVerticalHeaderLabels([str(n + payload.get("row_start", 1)) for n in range(len(rows))])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    self.table.setItem(i, j, QTableWidgetItem(value))
                    self.table.setColumnWidth(j, 160)
            self.stack.setCurrentWidget(self.table)
            for i, j in payload.get("focus_cells", []):
                cell = self.table.item(i, j)
                if cell:
                    cell.setBackground(QColor("#fff2a8"))
                    cell.setData(Qt.ItemDataRole.UserRole, True)
                    cell.setToolTip(plain_tooltip(cell.text()))
            if payload.get("focus_cells"):
                i, j = payload["focus_cells"][0]
                self.table.setCurrentCell(i, j)
                self.table.clearSelection()
                self.table.scrollToItem(self.table.item(i, j), QAbstractItemView.ScrollHint.PositionAtCenter)
        else:
            value = payload.get("text") or "本页没有可提取文字，可打开文件查看图片或图表"
            self.text.setPlainText(value)
            selections = []
            for start, end in payload.get("focus_spans", []):
                cursor = QTextCursor(self.text.document())
                cursor.setPosition(len(value[:start].encode("utf-16-le")) // 2)
                cursor.setPosition(len(value[:end].encode("utf-16-le")) // 2, QTextCursor.MoveMode.KeepAnchor)
                selected = QTextEdit.ExtraSelection()
                selected.cursor = cursor
                selected.format.setBackground(QColor("#fff2a8"))
                selections.append(selected)
            self.text.setExtraSelections(selections)
            self.stack.setCurrentWidget(self.text)
            if selections:
                cursor = QTextCursor(selections[0].cursor)
                cursor.setPosition(cursor.selectionStart())
                self.text.setTextCursor(cursor)
                self.text.centerCursor()
        count = payload["pages"]
        for control in (self.previous, self.next, self.page_number, self.page_total):
            control.setVisible(count > 1)
        self.page_number.setRange(1, count)
        self.page_number.setValue(self._page + 1)
        self.page_total.setText(f"/ {count}")
        self.page_number.setToolTip("预览" + payload.get("unit", "页"))
        self.previous.setEnabled(self._page > 0)
        self.next.setEnabled(self._page + 1 < count)
        self.enlarge.setEnabled(True)
        self._update_chrome()

    def show_large(self):
        if not self._target or not self._payload:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("文件预览 · " + Path(self._target[0]).name)
        dialog.resize(1050, 780)
        layout = QVBoxLayout(dialog)
        pane = FilePreview(dialog, expanded=True)
        pane._target = self._target
        pane._focus_mode = self._focus_mode
        pane.focus_button.setChecked(self._focus_mode)
        pane._sheet = self._sheet
        pane._display(self._payload)
        pane.refresh.setEnabled(True)
        layout.addWidget(pane)
        try:
            dialog.exec()
        finally:
            pane.shutdown()
            dialog.deleteLater()

    def shutdown(self):
        self.clear()
        for process in list(self._retired):
            process.kill()
            process.waitForFinished(1000)
