import base64
import hashlib
import io
import os
import time
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import pymupdf
from PIL import Image
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

from localmind.preview_worker import MAX_BYTES, PreviewError, preview_file
from localmind.desktop.detail_text import chat_excerpt
from localmind.desktop.file_preview import FilePreview
from test_result_lifecycle import screen  # noqa: F401


def wait(app, predicate, seconds=12):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not predicate():
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert predicate()


@pytest.fixture
def preview_app():
    return QApplication.instance() or QApplication([])


def test_pdf_original_pages_are_bounded_nonblank_and_read_only(tmp_path):
    path = tmp_path / "示例报告.pdf"
    doc = pymupdf.open()
    for title, color in [("Synthetic page one", (0.2, 0.6, 0.4)), ("Synthetic page two", (0.2, 0.4, 0.8))]:
        page = doc.new_page()
        page.insert_text((40, 60), title, fontsize=24)
        page.draw_rect((40, 100, 400, 300), color=color, fill=color)
    doc.save(path)
    doc.close()
    before = path.read_bytes()
    first, second = preview_file(str(path)), preview_file(str(path), 1)
    assert first["pages"] == 2 and second["page"] == 1
    assert first["image"] != second["image"]
    with Image.open(io.BytesIO(base64.b64decode(first["image"]))) as image:
        assert max(image.size) <= 1600
        assert any(low != high for low, high in image.getextrema())
    assert before == path.read_bytes()
    assert preview_file(str(path), 99)["page"] == 1


def test_image_and_text_never_render_markup(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (320, 200), "#369374").save(image)
    assert preview_file(str(image))["kind"] == "image"
    path = tmp_path / "说明.md"
    content = '<img src="https://example.invalid/private">\n中文内容\n' * 100
    path.write_text(content, encoding="utf-8")
    data = preview_file(str(path))
    assert data["kind"] == "text" and '<img src="https://' in data["text"]
    assert data["pages"] > 1 and preview_file(str(path), 1)["page"] == 1


def test_office_previews_are_content_not_faked_page_renders(tmp_path):
    word = tmp_path / "word.docx"
    document = Document()
    document.add_paragraph("第一段")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "字段", "值"
    document.add_paragraph("最后一段")
    document.save(word)
    data = preview_file(str(word))
    assert data["text"].index("第一段") < data["text"].index("字段") < data["text"].index("最后一段")
    assert "非原始排版" in data["note"]
    excel = tmp_path / "sheet.xlsx"
    book = Workbook()
    book.active.title = "第一页"
    book.active.append(["列名", "合成数据"])
    sheet = book.create_sheet("目标表")
    sheet.append(["<b>字面文字</b>", "=1+1"])
    book.save(excel)
    data = preview_file(str(excel), sheet="目标表")
    assert data["page"] == 1 and data["pages"] == 2
    assert data["rows"][0] == ["<b>字面文字</b>", ""]
    slides = tmp_path / "slides.pptx"
    deck = Presentation()
    for title in ("第一页标题", "第二页标题"):
        slide = deck.slides.add_slide(deck.slide_layouts[0])
        slide.shapes.title.text = title
    deck.save(slides)
    data = preview_file(str(slides), 1)
    assert "第二页标题" in data["text"]
    assert data["pages"] == 2 and data["caption"] == "幻灯片文字"


@pytest.mark.parametrize("path", ["https://example.invalid/file.pdf", "\\\\server\\share\\file.pdf", "//server/share/a.txt", "relative.txt", "localmind-chat://1"])
def test_remote_virtual_and_relative_paths_are_rejected(path):
    with pytest.raises(PreviewError):
        preview_file(path)


def test_large_unsupported_corrupt_and_changed_cached_files(tmp_path):
    large = tmp_path / "large.txt"
    with large.open("wb") as stream:
        stream.truncate(MAX_BYTES + 1)
    with pytest.raises(PreviewError, match="50 MB"):
        preview_file(str(large))
    executable = tmp_path / "script.exe"
    executable.write_bytes(b"not executable")
    with pytest.raises(PreviewError, match="格式"):
        preview_file(str(executable))
    text = tmp_path / "file.txt"
    text.write_text("first", encoding="utf-8")
    digest = hashlib.sha256(text.read_bytes()).hexdigest()
    assert preview_file(str(text), expected_hash=digest)["text"] == "first"
    text.write_text("changed", encoding="utf-8")
    with pytest.raises(PreviewError, match="不一致"):
        preview_file(str(text), expected_hash=digest)
    archive = tmp_path / "large.docx"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("large.xml", b"0" * (25 * 1024 * 1024))
    with pytest.raises(PreviewError, match="内部资源"):
        preview_file(str(archive))


def test_real_process_latest_selection_clear_and_timeout(preview_app, tmp_path):
    app = preview_app
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("第一份", encoding="utf-8")
    second.write_text("第二份", encoding="utf-8")
    pane = FilePreview()
    pane.resize(420, 260)
    pane.show()
    try:
        pane.set_target(str(first))
        wait(app, lambda: pane._process is not None)
        old_process, old_generation = pane._process, pane._generation
        pane.set_target(str(second))
        pane._finished(old_process, old_generation, 0)
        wait(app, lambda: pane._payload is not None)
        assert pane.text.toPlainText() == "第二份"
        assert pane._payload["path"] == str(second)
        pane.clear()
        assert pane._target is None and not pane.text.toPlainText()
        pane.TIMEOUT_MS = 1
        pane.set_target(str(first))
        wait(app, lambda: "已停止" in pane.message.text())
        assert pane._process is None and not pane.enlarge.isEnabled()
        wait(app, lambda: not pane._retired)
    finally:
        pane.shutdown()
        pane.close()


def test_detail_summary_is_readable_expandable_and_technical_info_is_hidden(screen):
    app, window, _, first, *_ = screen
    window.selected_result().snippet = "重点句。" * 200 + "末尾内容"
    window.show_selected_result()
    assert "消息 ID:" not in window.context_preview.toPlainText()
    assert "采集时间:" not in window.context_preview.toPlainText()
    assert first["name"] not in window.preview.toPlainText()
    assert len(window.preview.toPlainText()) < 600
    assert window.evidence_expand.isVisible()
    window.evidence_expand.click()
    assert "末尾内容" in window.preview.toPlainText()
    window.preview_tabs.setCurrentIndex(2)
    app.processEvents()
    assert window.file_summary.isVisible() and not window.info_preview.isVisible()
    assert "排序分" not in window.file_summary.toPlainText()
    window.technical_toggle.click()
    assert window.info_preview.isVisible() and "排序分" in window.info_preview.toPlainText()
    assert window.preview.font().pixelSize() >= 16
    assert window.preview.document().firstBlock().blockFormat().lineHeight() >= 140


def test_compacted_chat_excerpt_keeps_body_and_does_not_interpret_markup():
    text = "平台: wechat 会话: 合成群 发送者: 我 消息时间: 2026 附件: a.txt 消息: [文件] a.txt 上下文（引用 / 相邻消息，非附件内容）: 张工: <b>合成线索</b>"
    value = chat_excerpt(text, "a.txt")
    assert "合成线索" in value and "<b>" in value
    assert "平台:" not in value and "不属于文件正文" in value


def test_window_preview_keeps_displayed_file_when_attachment_rebound(screen):
    from localmind.chats import ChatStore
    app, window, db, first, other, attachment = screen
    assert window.file_preview._target[0] == first["path"]
    ChatStore(db).bind(attachment, other["id"])
    window.search_index_dirty.emit()
    assert window.file_preview._target[0] == first["path"]
    window.result_table.clearSelection()
    assert window.file_preview._target is None


def test_remote_mapped_drive_is_rejected_before_file_access(monkeypatch):
    import localmind.preview_worker as worker
    monkeypatch.setattr(worker, "_is_remote_drive", lambda _: True)
    with pytest.raises(PreviewError, match="网络路径"):
        worker.preview_file(os.path.abspath("not-present.pdf"))


def test_paging_fit_width_collapse_and_large_window(preview_app, tmp_path):
    app = preview_app
    path = tmp_path / "pages.pdf"
    document = pymupdf.open()
    for text in ("Page one", "Page two"):
        page = document.new_page()
        page.insert_text((50, 80), text, fontsize=24)
    document.save(path)
    document.close()
    pane = FilePreview()
    pane.resize(400, 260)
    pane.show()
    try:
        pane.set_target(str(path))
        wait(app, lambda: pane._payload is not None)
        assert pane.canvas.width() >= 380
        assert pane.image_scroll.verticalScrollBar().maximum() > 0
        pane.next.click()
        wait(app, lambda: pane._payload is not None and pane._page == 1)
        assert pane.page_number.value() == 2 and not pane.next.isEnabled()
        pane.collapse.click()
        app.processEvents()
        assert not pane.stack.isVisible() and pane.maximumHeight() == 44
        pane.collapse.click()
        app.processEvents()
        assert pane.stack.isVisible() and pane.maximumHeight() > 1000
        observed = []
        def inspect_dialog():
            dialog = QApplication.activeModalWidget()
            if not isinstance(dialog, QDialog):
                return
            child = dialog.findChild(FilePreview)
            observed.append((child._payload["page"], child._target[0], child.canvas.width()))
            dialog.accept()
        QTimer.singleShot(100, inspect_dialog)
        QTimer.singleShot(3000, lambda: QApplication.activeModalWidget().reject()
                          if QApplication.activeModalWidget() else None)
        pane.enlarge.click()
        assert observed and observed[0][0:2] == (1, str(path))
        assert observed[0][2] > 900
    finally:
        pane.shutdown()
        pane.close()


@pytest.mark.parametrize("exists", [True, False])
def test_corrupt_or_missing_file_has_actionable_error(preview_app, tmp_path, exists):
    path = tmp_path / "broken.pdf"
    if exists:
        path.write_bytes(b"not a PDF")
    pane = FilePreview()
    try:
        pane.set_target(str(path))
        wait(preview_app, lambda: pane._target is not None and not pane._debounce.isActive() and pane._process is None)
        assert pane._payload is None and not pane.enlarge.isEnabled()
        assert ("损坏" if exists else "移动或删除") in pane.message.text()
    finally:
        pane.shutdown()


def test_pending_attachment_preview_requires_current_signature_and_hash(screen, tmp_path):
    from localmind.chats import ChatStore
    from localmind.wechat_files import AttachmentLocator
    app, window, db, _, _, attachment_id = screen
    path = tmp_path / "已定位但未索引.txt"
    path.write_text("已确认的本地附件", encoding="utf-8")
    locator = AttachmentLocator(db, cache_dir=tmp_path / "cache")
    locator.remember(locator.attachment(attachment_id), path, "manual")
    with db.conn:
        db.conn.execute("UPDATE chat_attachments SET file_id=NULL WHERE id=?", (attachment_id,))
    window.run_search()
    wait(app, lambda: not window.search_busy)
    pending = window.selected_result()
    assert pending.result_kind == "attachment_pending"
    attachment = next(row for row in ChatStore(db).attachments() if row["id"] == attachment_id)
    window._show_file_preview(pending, attachment)
    wait(app, lambda: window.file_preview._payload is not None)
    assert window.file_preview.text.toPlainText() == "已确认的本地附件"
    path.write_text("被替换的内容", encoding="utf-8")
    window.file_preview.reload()
    wait(app, lambda: "不一致" in window.file_preview.message.text())
    with db.conn:
        db.conn.execute("UPDATE chat_attachment_locations SET signature='stale' WHERE attachment_id=?", (attachment_id,))
    window._show_file_preview(pending, attachment)
    assert window.file_preview._target is None
    assert "尚未定位" in window.file_preview.message.text()
