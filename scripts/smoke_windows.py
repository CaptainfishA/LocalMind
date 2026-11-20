"""Check the portable runtime using a synthetic database and the installed wheel."""

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import onnxruntime
from paddleocr import PaddleOCR
from PySide6.QtWidgets import QApplication
from sentence_transformers import SentenceTransformer

from localmind import __version__
from localmind.db import LocalMindDB
from localmind.ui import MainWindow


def main() -> None:
    assert __version__ == version("localmind-demo")
    assert faiss.IndexFlatL2(2).d == 2
    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
    assert callable(PaddleOCR) and callable(SentenceTransformer)

    app = QApplication([])
    with TemporaryDirectory() as directory:
        db = LocalMindDB(Path(directory) / "smoke.db")
        db.set_setting("privacy_accepted_v02", "1")
        window = MainWindow(db)
        try:
            window.show()
            app.processEvents()
            assert window.isVisible() and __version__ in window.windowTitle()
            assert not window.windowIcon().isNull()
        finally:
            window.close()
            app.processEvents()


if __name__ == "__main__":
    main()
