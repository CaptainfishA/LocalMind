"""Small, shared presentation primitives for the native desktop workspace."""

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QTableWidget, QVBoxLayout, QWidget,
)


@lru_cache(maxsize=160)
def icon(name: str, color: str = "#64756b", size: int = 20) -> QIcon:
    renderer = QSvgRenderer(str(Path(__file__).parent / "assets" / "icons" / f"{name}.svg"))
    result = QIcon()
    for scale in (1, 2, 3):
        image = QPixmap(size * scale, size * scale)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, size * scale, size * scale))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(image.rect(), QColor(color))
        painter.end()
        image.setDevicePixelRatio(scale)
        result.addPixmap(image)
    return result


def button(text, glyph, callback, *, primary=False, compact=False, tip=None):
    value = QPushButton(text if not compact else "")
    value.setIcon(icon(glyph, "#ffffff" if primary else "#64756b"))
    value.setIconSize(QSize(18, 18))
    value.setCursor(Qt.CursorShape.PointingHandCursor)
    value.setAccessibleName(text)
    value.setToolTip(tip or text)
    if primary:
        value.setObjectName("primaryButton")
    elif compact:
        value.setObjectName("iconButton")
        value.setFixedSize(36, 36)
    value.clicked.connect(callback)
    return value


def label(text, name="mutedText", *, wrap=False):
    value = QLabel(text)
    value.setObjectName(name)
    value.setTextFormat(Qt.TextFormat.PlainText)
    value.setWordWrap(wrap)
    return value


def divider():
    value = QFrame()
    value.setObjectName("divider")
    value.setFixedHeight(1)
    return value


def page_layout(page=None, *, margin=24):
    page = page or QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(margin, 20, margin, 20)
    layout.setSpacing(16)
    return page, layout


def section(layout, title):
    layout.addWidget(label(title, "sectionTitle"))


class ElidedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(28)

    def sizeHint(self):  # noqa: N802
        return QSize(160, 28)

    def minimumSizeHint(self):  # noqa: N802
        return QSize(0, 28)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setFont(self.font())
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(self.contentsRect(), Qt.AlignmentFlag.AlignVCenter,
                         self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width()))
        painter.end()


class ResultDelegate(QStyledItemDelegate):
    """Keep model values intact while presenting a file as a two-line list row."""

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        title, glyph = opt.text, QIcon(opt.icon)
        opt.text = ""
        opt.icon = QIcon()
        style = opt.widget.style() if opt.widget else None
        if style:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        painter.save()
        box = option.rect.adjusted(14, 0, -12, 0)
        glyph.paint(painter, QRect(box.x(), box.center().y() - 11, 22, 22))
        box.adjust(36, 0, 0, 0)
        painter.setPen(QColor("#273c31"))
        font = QFont(option.font)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(QRect(box.x(), box.y() + 13, box.width(), 23), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, box.width()))
        subtitle = index.data(Qt.ItemDataRole.UserRole) or ""
        painter.setPen(QColor("#89928d"))
        font.setWeight(QFont.Weight.Normal)
        font.setPointSizeF(max(8, option.font.pointSizeF() - 1))
        painter.setFont(font)
        painter.drawText(QRect(box.x(), box.y() + 38, box.width(), 20), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(str(subtitle), Qt.TextElideMode.ElideRight, box.width()))
        painter.setPen(QColor("#f0f3f1"))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        painter.restore()


class DataTable(QTableWidget):
    def __init__(self, rows=0, columns=0, parent=None):
        super().__init__(rows, columns, parent)
        self.empty_title = "暂无记录"
        self.empty_glyph = "database"
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().hide()
        self.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.verticalHeader().setDefaultSectionSize(48)
        self.setAlternatingRowColors(False)

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if self.rowCount() or not self.empty_title:
            return
        painter = QPainter(self.viewport())
        rect = self.viewport().rect()
        center = rect.center()
        icon(self.empty_glyph, "#bccbc2", 32).paint(painter, QRect(center.x() - 16, center.y() - 40, 32, 32))
        painter.setPen(QColor("#8a968e"))
        painter.drawText(QRect(10, center.y() + 4, rect.width() - 20, 26), Qt.AlignmentFlag.AlignCenter, self.empty_title)
        painter.end()


class TimestampDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        date, _, time = opt.text.partition(" ")
        opt.text = ""
        opt.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        painter.save()
        painter.setPen(QColor("#697e71"))
        font = QFont(option.font)
        font.setPointSizeF(max(8, font.pointSizeF() - 1))
        painter.setFont(font)
        rect = option.rect.adjusted(10, 0, -6, 0)
        painter.drawText(QRect(rect.x(), rect.y() + 14, rect.width(), 22), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(date, Qt.TextElideMode.ElideRight, rect.width()))
        painter.setPen(QColor("#97a39c"))
        painter.drawText(QRect(rect.x(), rect.y() + 38, rect.width(), 18), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(time, Qt.TextElideMode.ElideRight, rect.width()))
        painter.restore()


def result_icon(ext, kind="file"):
    if kind == "message":
        return icon("message-circle", "#339477")
    glyph, color = {
        "pdf": ("file-text", "#cd6b68"),
        "docx": ("file-text", "#5888ba"),
        "xlsx": ("file-spreadsheet", "#439a7c"),
        "csv": ("file-spreadsheet", "#439a7c"),
        "pptx": ("presentation", "#bd9564"),
        "png": ("image", "#8a7ba5"), "jpg": ("image", "#8a7ba5"), "jpeg": ("image", "#8a7ba5"),
    }.get(ext.lower().lstrip("."), ("file-text", "#7c949a"))
    return icon(glyph, color, 22)
