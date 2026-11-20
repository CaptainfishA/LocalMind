import base64
import io
import os
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from localmind.desktop.file_preview import FilePreview
from localmind.preview_worker import preview_file
from test_file_preview import preview_app, wait  # noqa: F401


LINES = ("低温环境下，备用电池每季度进行放电测试。", "记录容量衰减并在复盘会上核对。")
ANCHOR = "\n".join(LINES)


def pdf_fixture(path, *, rotation=0, crop=False, text_rotation=0):
    with pymupdf.open() as document:
        first = document.new_page(width=600, height=900)
        first.insert_text((55, 80), "资料封面", fontname="china-s", fontsize=24)
        page = document.new_page(width=600, height=900)
        page.insert_text((55, 90), "设备维护记录", fontname="china-s", fontsize=24, color=(.1, .35, .25))
        for y in (180, 230, 280):
            page.insert_text((55, y), "日常检查按值班安排执行，完成后记录结果。", fontname="china-s", fontsize=14)
        if text_rotation:
            page.insert_text((400, 700), LINES[0], fontname="china-s", fontsize=16, rotate=text_rotation)
        else:
            for n, line in enumerate(LINES):
                page.insert_text((55, 660 + n * 32), line, fontname="china-s", fontsize=16)
        if crop:
            page.set_cropbox(pymupdf.Rect(20, 30, 580, 850))
        page.set_rotation(rotation)
        document.save(path)
    return path


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [False, True])
def test_pdf_regions_align_with_actual_pixels_after_rotation_and_crop(tmp_path, rotation, crop):
    path = pdf_fixture(tmp_path / "设备资料.pdf", rotation=rotation, crop=crop)
    before = path.read_bytes()
    result = preview_file(str(path), page=1, anchor=ANCHOR)
    assert result["focus_status"] == "matched" and len(result["focus_polygons"]) == 2
    assert result["page"] == 1 and result["pages"] == 2
    assert result["image"] == preview_file(str(path), page=1)["image"]
    pixels = Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB")
    for polygon in result["focus_polygons"]:
        assert len(polygon) == 4 and all(0 <= value <= 1 for point in polygon for value in point)
        xs, ys = zip(*polygon)
        box = (round(min(xs) * pixels.width), round(min(ys) * pixels.height),
               round(max(xs) * pixels.width), round(max(ys) * pixels.height))
        assert pixels.crop(box).convert("L").getextrema()[0] < 80
    assert path.read_bytes() == before
    with pymupdf.open(path) as document:
        assert not list(document[1].annots() or ())


def test_rotated_text_keeps_polygon_direction(tmp_path):
    path = pdf_fixture(tmp_path / "vertical.pdf", text_rotation=90)
    result = preview_file(str(path), page=1, anchor=LINES[0])
    assert result["focus_status"] == "matched"
    points = result["focus_polygons"][0]
    assert abs(points[0][0] - points[1][0]) < .01
    assert abs(points[0][1] - points[1][1]) > .1


def test_missing_partial_or_other_page_text_never_gets_fake_highlights(tmp_path):
    path = pdf_fixture(tmp_path / "pages.pdf")
    for page, anchor in ((0, ANCHOR), (1, LINES[0] + "不存在的第二条条件")):
        result = preview_file(str(path), page=page, anchor=anchor)
        assert result["focus_status"] == "not_found" and not result["focus_polygons"]
        assert result["page"] == page
    # Recover the current line break, even if the stored anchor used different whitespace.
    result = preview_file(str(path), page=1, anchor="".join(LINES))
    assert result["focus_status"] == "matched" and len(result["focus_polygons"]) == 2


def test_image_only_pdf_is_rendered_without_inventing_ocr_coordinates(tmp_path):
    path = tmp_path / "scan.pdf"
    image = io.BytesIO()
    Image.new("RGB", (100, 100), "#167d61").save(image, format="PNG")
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_image(pymupdf.Rect(50, 100, 450, 500), stream=image.getvalue())
        document.save(path)
    result = preview_file(str(path), anchor=ANCHOR)
    assert result["kind"] == "image" and result["image"]
    assert result["focus_status"] == "no_text" and not result["focus_polygons"]


def test_pdf_locating_is_one_page_bounded_and_optional(tmp_path, monkeypatch):
    import localmind.pdf_focus as focus
    path = pdf_fixture(tmp_path / "budget.pdf")
    loads = []
    original = pymupdf.Document.load_page
    monkeypatch.setattr(pymupdf.Document, "load_page", lambda self, page: loads.append(page) or original(self, page))
    monkeypatch.setattr(focus, "MAX_PAGE_CHARS", 10)
    result = preview_file(str(path), page=1, anchor=ANCHOR)
    assert result["focus_status"] == "limited" and result["image"] and not result["focus_polygons"]
    assert loads == [1]
    monkeypatch.setattr(focus, "MAX_PAGE_CHARS", 100000)
    monkeypatch.setattr(pymupdf.Page, "get_textpage", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")))
    result = preview_file(str(path), page=1, anchor=ANCHOR)
    assert result["focus_status"] == "unavailable" and result["image"]
    assert preview_file(str(path), page=1)["focus_status"] == ""


def test_repeated_full_passages_have_a_bounded_number_of_regions(tmp_path, monkeypatch):
    import localmind.pdf_focus as focus
    path = tmp_path / "repeated.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        for y in (100, 180, 260):
            page.insert_text((40, y), LINES[0], fontname="china-s", fontsize=14)
        document.save(path)
    monkeypatch.setattr(focus, "MAX_REGIONS", 2)
    result = preview_file(str(path), anchor=LINES[0])
    assert result["focus_status"] == "matched" and len(result["focus_polygons"]) == 2
    assert "前 2 个区域" in result["note"]


def visible_focus(pane):
    from PySide6.QtCore import QRectF
    rect = pane.canvas.focus_rect()
    viewport = QRectF(0, pane.image_scroll.verticalScrollBar().value(),
                      pane.image_scroll.viewport().width(), pane.image_scroll.viewport().height())
    return not rect.isEmpty() and viewport.intersects(rect)


def test_pdf_real_worker_highlight_scroll_resize_toggle_and_clear(preview_app, tmp_path):
    from PySide6.QtGui import QFontDatabase
    from localmind.desktop.theme import APP_STYLESHEET

    app = preview_app
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    path = pdf_fixture(tmp_path / "设备维护.pdf")
    pane = FilePreview()
    pane.setStyleSheet(APP_STYLESHEET)
    pane.resize(420, 340)
    pane.show()
    try:
        pane.set_target(str(path), page=2, anchor=ANCHOR)
        wait(app, lambda: pane._payload is not None and visible_focus(pane))
        assert pane._page == 1 and pane.focus_button.isChecked() and pane.focus_button.isVisible()
        assert "已标记" in pane.focus_notice.text() and pane.image_scroll.verticalScrollBar().value() > 0
        pixels = pane.canvas.grab().toImage()
        ratio = pixels.devicePixelRatio()
        bounds = pane.canvas.focus_rect().toAlignedRect().intersected(pane.canvas.rect())
        assert sum(pixels.pixelColor(round(x * ratio), round(y * ratio)).red() > 220
                   and pixels.pixelColor(round(x * ratio), round(y * ratio)).blue() < 200
                   for y in range(bounds.top(), bounds.bottom(), 2)
                   for x in range(bounds.left(), bounds.right(), 2)) > 100
        for width in (340, 720, 420):
            pane.resize(width, 300)
            wait(app, lambda: visible_focus(pane))
        pane.resize(420, 650)
        wait(app, lambda: visible_focus(pane))
        pane.resize(420, 200)
        wait(app, lambda: visible_focus(pane))
        # Resizing must not drag the reader back after they intentionally scroll away.
        pane.image_scroll.verticalScrollBar().setValue(0)
        assert not visible_focus(pane)
        pane.resize(380, 280)
        app.processEvents()
        assert pane.image_scroll.verticalScrollBar().value() == 0 and not visible_focus(pane)
        pane.focus_button.click()
        wait(app, lambda: pane._payload is not None)
        assert pane._page == 1 and not pane.canvas.focus_polygons and pane.focus_notice.text() == "原页预览"
        pane.focus_button.click()
        wait(app, lambda: pane._payload is not None and visible_focus(pane))
        pane.previous.click()
        wait(app, lambda: pane._payload is not None and pane._page == 0)
        assert not pane.canvas.focus_polygons and "未找到" in pane.focus_notice.text()
        pane.focus_button.click()
        wait(app, lambda: pane._payload is not None)
        assert pane._page == 0 and not pane._focus_mode
        pane.focus_button.click()
        wait(app, lambda: pane._payload is not None and visible_focus(pane))
        assert pane._page == 1
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            assert pane.grab().save(str(Path(output) / "pdf-focus-pane.png"))
        pane.clear()
        app.processEvents()
        assert not pane.canvas.focus_polygons and not pane.image_scroll._focus_timer.isActive()
        assert pane.navigation.isHidden()
    finally:
        pane.shutdown()
        pane.close()


def test_pdf_enlargement_keeps_current_page_and_read_only_highlight(preview_app, tmp_path):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog
    from localmind.desktop.theme import APP_STYLESHEET

    path = pdf_fixture(tmp_path / "enlarge.pdf")
    before = path.read_bytes()
    pane = FilePreview()
    pane.setStyleSheet(APP_STYLESHEET)
    pane.show()
    observed, errors = [], []
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(lambda: QApplication.activeModalWidget().reject()
                          if QApplication.activeModalWidget() else None)
    try:
        pane.set_target(str(path), page=2, anchor=ANCHOR)
        wait(preview_app, lambda: pane._payload is not None and visible_focus(pane))
        def inspect():
            dialog = QApplication.activeModalWidget()
            try:
                assert isinstance(dialog, QDialog)
                child = dialog.findChild(FilePreview)
                wait(preview_app, lambda: visible_focus(child))
                assert child._page == 1 and child._target == pane._target and child._focus_mode
                child.refresh.click()
                wait(preview_app, lambda: child._payload is not None and visible_focus(child))
                assert child._page == 1
                output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
                if output:
                    Path(output).mkdir(parents=True, exist_ok=True)
                    assert dialog.grab().save(str(Path(output) / "pdf-focus-large.png"))
                observed.append(True)
            except Exception as error:
                errors.append(error)
            finally:
                if dialog:
                    dialog.accept()
        guard.start(15000)
        QTimer.singleShot(100, inspect)
        pane.enlarge.click()
        assert observed and not errors, errors
        assert path.read_bytes() == before
    finally:
        guard.stop()
        pane.shutdown()
        pane.close()


def test_wechat_body_result_opens_exact_page_while_chat_only_does_not_invent_pdf_focus(preview_app, tmp_path):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from localmind.chats import Attachment, ChatMessage, ChatStore
    from localmind.db import LocalMindDB
    from localmind.embeddings import EmbeddingService
    from localmind.indexer import IndexingService
    from localmind.models import SearchQuery
    from localmind.ocr import OcrService
    from localmind.reranker import RerankerService
    from localmind.searcher import SearchService
    from localmind.ui import MainWindow

    app = preview_app
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    path = pdf_fixture(tmp_path / "备用电池维护.pdf")
    db = LocalMindDB(tmp_path / "index.db")
    embedding = EmbeddingService(force_hash=True)
    IndexingService(db, embedding_service=embedding, ocr_service=OcrService(enabled=False)).index_path(path)
    ChatStore(db).ingest(platform="wechat", account="synthetic", conversation="设备资料组", source_key="one",
        messages=[ChatMessage("file", "轮岗教学时会用到这份资料。", sender="张同事",
                              attachments=[Attachment(path.name, str(path))])], embedding_service=embedding)
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("search_retrieval_mode", "keyword")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    service = SearchService(db, embedding_service=embedding, reranker_service=RerankerService(enabled=False),
                            retrieval_mode="keyword")
    window = MainWindow(db)
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    window.evidence_filter.setCurrentIndex(window.evidence_filter.findData("content"))
    window.show()
    try:
        query = SearchQuery("低温 备用电池", scope="chat_files", evidence="content")
        window.search_input.setText(query.text)
        hits = service.search(query)
        assert hits and hits[0].page_no == 2 and hits[0].preview_anchor
        window._search_finished(hits, service.last_diagnostics, {"total_ms": 15, "index_refresh_ms": 0})
        wait(app, lambda: window.file_preview._payload is not None and visible_focus(window.file_preview))
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            wait(app, lambda: visible_focus(window.file_preview))
            assert window.file_preview._page == 1 and window.file_preview._payload["path"] == str(path)
            for widget in (window.file_preview.focus_button, window.file_preview.enlarge,
                           window.file_preview.page_number, window.file_actions[0]):
                point = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert point.x() <= width and point.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"pdf-focus-window-{width}.png"))
        query = SearchQuery("轮岗教学", scope="chat_files", evidence="chat")
        window.evidence_filter.setCurrentIndex(window.evidence_filter.findData("chat"))
        window.search_input.setText(query.text)
        hits = service.search(query)
        assert hits and hits[0].path == str(path) and not hits[0].preview_anchor
        window._search_finished(hits, service.last_diagnostics, {"total_ms": 15, "index_refresh_ms": 0})
        wait(app, lambda: window.file_preview._payload is not None)
        assert not window.file_preview._focus_available() and not window.file_preview.canvas.focus_polygons
    finally:
        window.close()
        app.processEvents()
