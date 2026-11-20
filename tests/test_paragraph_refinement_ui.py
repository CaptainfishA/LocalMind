import os
from pathlib import Path

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFontDatabase

from localmind.ui import MainWindow
from test_file_preview import preview_app, wait  # noqa: F401
from test_paragraph_refinement import BACKGROUND, QUERY, TARGET, detail_corpus  # noqa: F401


def test_refined_detail_is_readable_expandable_and_targets_current_file(preview_app, detail_corpus, tmp_path):
    app = preview_app
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db, _, service, _, _, _, _, _, _, query = detail_corpus
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    hits = service.search(query)
    window = MainWindow(db)
    window.search_input.setText(QUERY)
    window.show()
    try:
        window._search_finished(hits, service.last_diagnostics, {"total_ms": 120, "index_refresh_ms": 0})
        wait(app, lambda: window.file_preview._payload is not None)
        assert "匹配片段" in window.preview.toPlainText() and TARGET in window.preview.toPlainText()
        assert BACKGROUND not in window.preview.toPlainText()
        assert window.evidence_expand.isVisible()
        assert window.file_preview._target[5] == TARGET
        assert window.file_preview._payload["focus_status"] == "matched"
        assert "正文细化完成" in window.search_status.toolTip()
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            app.processEvents()
            text = window.file_preview.text
            assert text.extraSelections()
            assert text.viewport().rect().contains(text.cursorRect().center())
            for widget in (window.preview, window.evidence_expand, window.file_preview, window.file_actions[0]):
                point = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert point.x() <= width and point.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"refined-detail-{width}.png"))
        window.evidence_expand.click()
        assert "上下文" in window.preview.toPlainText()
        assert BACKGROUND in "".join(window.preview.toPlainText().split())
        window.result_table.clearSelection()
        assert window.file_preview._target is None and "匹配片段" not in window.preview.toPlainText()
    finally:
        window.close()
        app.processEvents()


def test_preview_resize_preserves_user_scrolled_away_position(preview_app):
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QTextEdit
    from localmind.desktop.file_preview import PreviewText

    pane = PreviewText()
    pane.setPlainText("\n".join(f"Synthetic line {n}" for n in range(100)))
    pane.resize(360, 280)
    pane.show()
    try:
        preview_app.processEvents()
        selection = QTextEdit.ExtraSelection()
        selection.cursor = QTextCursor(pane.document().findBlockByNumber(70))
        pane.setExtraSelections([selection])
        pane.setTextCursor(selection.cursor)
        pane.centerCursor()
        pane.verticalScrollBar().setValue(0)
        assert not pane.viewport().rect().intersects(pane.cursorRect())
        pane.resize(320, 220)
        preview_app.processEvents()
        preview_app.processEvents()
        assert pane.verticalScrollBar().value() == 0
    finally:
        pane.close()
