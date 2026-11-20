"""Keep Qt resources and deferred widget destruction within one application."""

import gc
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
    if existing_app is None:
        app.closeAllWindows()
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    # Drop cached Qt objects while the application is still alive.
    icon.cache_clear()
