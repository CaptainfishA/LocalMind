"""Keep Qt resources and deferred widget destruction within one application."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from localmind.desktop.widgets import icon


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    # Recreating QApplication while cached QPixmaps/widgets survive is unsafe.
    existing_app = QApplication.instance()
    app = existing_app or QApplication([])
    yield app
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    # Release Python-owned Qt resources before interpreter shutdown unloads Qt.
    icon.cache_clear()
    if existing_app is None:
        app.shutdown()
