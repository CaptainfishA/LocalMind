"""Keep Qt resources and deferred widget destruction within one application."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    # Recreating QApplication while cached QPixmaps/widgets survive is unsafe.
    app = QApplication.instance() or QApplication([])
    yield app
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
