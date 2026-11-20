"""Shared desktop tokens and Qt widget styling."""

from pathlib import Path

from PySide6.QtCore import QDir

QDir.addSearchPath("localmindicons", str(Path(__file__).parent / "assets" / "icons"))

APP_STYLESHEET = """
QWidget { background: #ffffff; color: #252b32; font-family: 'Microsoft YaHei UI'; font-size: 13px; }
QMainWindow, QDialog { background: #ffffff; }
QWidget#workspace, QScrollArea, QStackedWidget { background: #ffffff; }
QFrame#sidebar { background: #f3f5f6; border-right: 1px solid #e2e7e8; }
QFrame#sidebar QLabel { background: transparent; }
QLabel#brand { font-family: 'Segoe UI'; font-size: 19px; font-weight: 700; color: #20362e; }
QFrame#sidebar QLabel#brandMark { background: #237b63; border-radius: 8px; }
QLabel#navCaption { color: #8b939b; font-size: 11px; padding: 8px 10px 4px; }
QLabel#versionLabel { color: #92999f; font-family: 'Segoe UI'; font-size: 11px; }
QPushButton#navButton { text-align: left; padding: 0 12px; border: 0; background: transparent; color: #58636c; font-weight: 400; }
QPushButton#navButton:hover { background: #e8edee; }
QPushButton#navButton:checked { background: #e0eee8; color: #17664e; font-weight: 600; }
QFrame#workspaceHeader { background: white; border-bottom: 1px solid #e9edef; }
QLabel#pageTitle { font-size: 19px; font-weight: 600; }
QLabel#pageContext, QLabel#mutedText, QLabel#searchStatus { color: #7b858d; }
QLabel#sectionTitle { font-size: 16px; font-weight: 600; }
QLabel#statusChip { color: #237b63; background: #eef6f1; padding: 5px 10px; border-radius: 5px; font-size: 12px; }
QFrame#divider { background: #e8edef; border: 0; max-height: 1px; }
QLineEdit, QComboBox, QDateEdit, QSpinBox {
    background: #ffffff; border: 1px solid #dce2e5; border-radius: 6px;
    min-height: 34px; padding: 0 10px; selection-background-color: #dceee5; selection-color: #20352b;
}
QLineEdit:hover, QComboBox:hover { border-color: #bac9c2; }
QLineEdit:focus, QComboBox:focus, QDateEdit:focus, QSpinBox:focus { border: 1px solid #369374; }
QLineEdit#searchInput { font-size: 15px; background: #f6f8f7; padding: 0 12px; border: 1px solid #dce5e0; }
QLineEdit#searchInput:focus { background: white; border: 1px solid #369374; }
QComboBox::drop-down { border: 0; width: 24px; }
QComboBox { padding-right: 26px; }
QComboBox::down-arrow { image: url(localmindicons:chevron-down.svg); width: 14px; height: 14px; }
QComboBox QAbstractItemView { background: white; border: 1px solid #dce2e5; padding: 4px; selection-background-color: #e0eee8; selection-color: #17664e; }
QPushButton { min-height: 32px; padding: 0 12px; border: 1px solid #dce2e5; background: white; border-radius: 6px; font-weight: 500; }
QPushButton:hover { background: #f0f5f2; border-color: #b8cec2; }
QPushButton:pressed { background: #e3eee7; }
QPushButton:focus { border-color: #369374; }
QPushButton:disabled { background: #f5f6f7; color: #a3aab0; border-color: #e8ecee; }
QPushButton#primaryButton { background: #237b63; color: white; border-color: #237b63; font-weight: 600; }
QPushButton#primaryButton:hover { background: #1b6852; border-color: #1b6852; }
QPushButton#primaryButton:pressed { background: #14533f; }
QPushButton#primaryButton:disabled { background: #edf2ef; color: #9ca9a1; border-color: #e2e8e4; }
QPushButton#dangerButton { color: #b34c4c; border-color: #e7caca; }
QPushButton#dangerButton:hover { background: #fcf0f0; }
QPushButton#iconButton { border: 0; background: transparent; padding: 0; }
QPushButton#iconButton:hover { background: #edf2ef; }
QPushButton#iconButton:checked { background: #e0eee8; }
QPushButton#iconButton:disabled { background: transparent; }
QPushButton#retrievalMode { border: 0; background: #f2f4f5; color: #707b82; padding: 0 9px; font-size: 12px; }
QPushButton#retrievalMode:checked { background: #e1eee7; color: #216a50; font-weight: 600; }
QPushButton#retrievalMode:hover { background: #e7eeea; }
QToolButton { border: 0; padding: 6px; color: #617269; border-radius: 4px; }
QToolButton:hover { background: #edf2ef; }
QCheckBox { spacing: 7px; background: transparent; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #9daeaa; border-radius: 3px; background: #ffffff; }
QCheckBox::indicator:checked { background: #dceee5; border-color: #369374; image: url(localmindicons:check.svg); }
QCheckBox::indicator:indeterminate { background: #369374; border-color: #369374; }
QCheckBox::indicator:disabled { background: #edf1ef; border-color: #cdd5d0; }
QTableWidget, QTableView, QListWidget, QPlainTextEdit, QTextEdit {
    background: white; border: 0; selection-background-color: #edf5f0; selection-color: #20382c;
    alternate-background-color: #fafbfc; gridline-color: transparent;
}
QTableView::item, QListWidget::item { border: 0; padding: 8px; }
QTableView::item:selected { background: #eaf4ed; color: #20382c; }
QTableView::item:hover, QListWidget::item:hover { background: #f3f7f4; }
QHeaderView::section { background: #f8faf9; color: #849088; padding: 10px; border: 0; border-bottom: 1px solid #e9eeeb; font-weight: 400; font-size: 12px; }
QTableWidget#resultTable { alternate-background-color: white; }
QFrame#previewPanel { background: #fbfcfc; border-left: 1px solid #e5eae7; }
QFrame#previewPanel QPlainTextEdit { background: #fbfcfc; padding: 6px 0; }
QLabel#detailTitle { font-size: 18px; font-weight: 600; background: transparent; }
QLabel#detailMeta { font-size: 13px; color: #65766c; background: transparent; }
QLabel#identifierNotice { font-size: 14px; color: #94611d; background: transparent; }
QLabel#identifierNotice[state='matched'] { color: #237b63; }
QPlainTextEdit#readableText { font-size: 16px; color: #273730; padding: 6px 2px; }
QPlainTextEdit#filePreviewText, QTableWidget#filePreviewTable { font-size: 15px; }
QTableWidget#filePreviewTable::item { padding: 3px 6px; }
QTableWidget#filePreviewTable QHeaderView::section { padding: 4px 6px; }
QLabel#previewCaption { font-size: 14px; font-weight: 600; }
QLabel#previewNote { font-size: 12px; color: #718077; }
QWidget#filePreview { border-top: 1px solid #e1e7e3; }
QTabWidget::pane { border: 0; background: transparent; }
QTabBar::tab { color: #7a8580; background: transparent; padding: 10px 13px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #236e53; border-bottom-color: #369374; font-weight: 600; }
QTabBar::tab:disabled { color: #b9c0bc; }
QTabBar::tab:hover { color: #236e53; }
QSplitter::handle { background: #edf0ee; width: 1px; }
QProgressBar { border: 0; background: #edf1ee; max-height: 5px; border-radius: 2px; }
QProgressBar::chunk { background: #369374; border-radius: 2px; }
QProgressBar[state='partial']::chunk { background: #c5963f; }
QLabel#connectionStatus { color: #86918b; font-weight: 500; }
QLabel#connectionStatus[state='ready'] { color: #237b63; }
QLabel#connectionStatus[state='error'] { color: #b34c4c; }
QLabel#scopeLabel { font-weight: 600; }
QLabel#metricValue { font-family: 'Segoe UI'; font-size: 28px; font-weight: 600; color: #253b31; }
QFrame#metricBand { border-bottom: 1px solid #e8edef; }
QFrame#filterPanel { background: #f6f8f7; border-radius: 6px; }
QFrame#filterPanel QLabel, QFrame#filterPanel QCheckBox { background: transparent; }
QFrame#noticeBand { background: #fff8e8; border-radius: 4px; }
QScrollBar:vertical { width: 8px; background: transparent; margin: 0; }
QScrollBar::handle:vertical { background: #d1d9d4; min-height: 30px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #b8c5bc; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { height: 8px; background: transparent; }
QScrollBar::handle:horizontal { background: #d1d9d4; min-width: 30px; border-radius: 4px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QMenu { background: white; border: 1px solid #dde5df; padding: 5px; }
QMenu::item { padding: 8px 25px 8px 12px; border-radius: 4px; }
QMenu::item:selected { background: #eaf4ed; }
QToolTip { color: #354139; background: #ffffff; border: 1px solid #dce4df; padding: 6px; }
"""
