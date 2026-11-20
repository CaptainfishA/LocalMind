from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QFontDatabase

from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.indexer import IndexingService
from localmind.ocr import OcrService
from localmind.ui import MainWindow


def test_startup_reconcile_repeat_scan_and_device_setting(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    for font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/consola.ttf"]:
        if Path(font_path).exists():
            QFontDatabase.addApplicationFont(font_path)
    docs = tmp_path / "files"
    docs.mkdir()
    old = docs / "旧文件.txt"
    old.write_text("旧版本", encoding="utf-8")
    db = LocalMindDB(tmp_path / "index.db")
    IndexingService(db, embedding_service=EmbeddingService(force_hash=True),
                    ocr_service=OcrService(enabled=False)).scan_and_index([docs], include_embeddings=False)
    db.add_watched_folder(str(docs))
    for key, value in {"model_cache_dir": str(tmp_path / "cache"), "privacy_accepted_v02": "1",
                       "startup_scan": "1", "monitor_enabled": "0", "embedding_enabled": "0", "ocr_enabled": "0"}.items():
        db.set_setting(key, value)
    old.unlink()
    (docs / "新文件.txt").write_text("新版本合同", encoding="utf-8")
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(str(args)))
    window = MainWindow(db)
    window.show()

    def wait_scan():
        end = time.monotonic() + 20
        app.processEvents()
        while window.scan_thread is not None and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert window.scan_thread is None

    try:
        wait_scan()
        assert not errors
        assert db.get_file_by_path(str(old)) is None
        assert db.get_file_by_path(str(docs / "新文件.txt")) is not None
        assert not window.cancel_scan_button.isEnabled()
        assert window.scan_button.isEnabled()
        window.scan_button.click()
        wait_scan()
        assert not errors
        window.embedding_device_combo.setCurrentIndex(window.embedding_device_combo.findData("cpu"))
        assert db.get_setting("embedding_device") == "cpu"
        window.tabs.setCurrentIndex(2)
        app.processEvents()
        for size in [(1240, 800), (980, 650)]:
            window.resize(*size)
            app.processEvents()
            assert window.embedding_device_combo.geometry().width() > 50
            image_path = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if image_path:
                target = Path(image_path)
                target.mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(target / f"v03-status-{size[0]}.png"))
    finally:
        if window.scan_worker is not None:
            window.cancel_scan()
            wait_scan()
        window.close()
        app.processEvents()
