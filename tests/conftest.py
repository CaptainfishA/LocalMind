"""Keep Qt resources and deferred widget destruction within one application."""

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from localmind.desktop.widgets import icon
from localmind.ui import MainWindow


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


@pytest.fixture(autouse=True)
def release_desktop_windows(qt_application):
    existing = {
        widget for widget in qt_application.topLevelWidgets() if isinstance(widget, MainWindow)
    }
    yield
    # Destroy each test's closed windows before the next test allocates Qt objects.
    for window in qt_application.topLevelWidgets():
        if isinstance(window, MainWindow) and window not in existing and window.close():
            window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
