import csv
import hashlib
import os
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.desktop.file_preview import FilePreview
from localmind.embeddings import EmbeddingService
from localmind.indexer import IndexingService
from localmind.models import SearchQuery
from localmind.ocr import OcrService
from localmind.preview_focus import find_anchor, table_window
from localmind.preview_worker import PreviewError, _read_text, preview_file
from localmind.query_processing import process_query
from localmind.searcher import make_preview_anchor
from test_file_preview import preview_app, wait  # noqa: F401
from test_weflow_v04 import index_file, searcher


ANCHOR = "低温环境下，备用电池每季度进行放电测试，并记录容量衰减。"


def text_fixture(path):
    path.write_text("常规章节与背景说明。\n" * 4000 + "😀" + ANCHOR + "\n" + "后续说明。\n" * 1000, encoding="utf-8")
    return path


def test_whitespace_alignment_retains_original_offsets_and_first_occurrence():
    value = "前言\n😀低温\n环境  电池\t维护。后文低温环境电池维护。"
    start, end = find_anchor(value, "低温环境 电池维护。")
    assert value[start:end] == "低温\n环境  电池\t维护。"
    assert find_anchor(value, "另一段") is None
    assert find_anchor(value, " \n") is None


def test_semantic_anchor_retains_distinct_context_instead_of_repeated_longest_line(tmp_path):
    boilerplate = "常规归档及背景说明。" * 10
    chunk = boilerplate + "\n" + ANCHOR
    anchor = make_preview_anchor(chunk, process_query("寒冷天气下蓄电设备如何检测"), source_type="docx")
    assert anchor == chunk
    path = tmp_path / "repeated.docx"
    document = Document()
    for _ in range(400):
        document.add_paragraph(boilerplate)
    document.add_paragraph(ANCHOR)
    document.save(path)
    assert ANCHOR not in preview_file(str(path))["text"]
    result = preview_file(str(path), anchor=anchor)
    assert result["focus_status"] == "matched" and ANCHOR in result["text"]
    assert len(make_preview_anchor("a" * 1000, process_query("测试"))) <= 512


def test_long_text_focus_is_current_read_only_bounded_and_paged(tmp_path):
    path = text_fixture(tmp_path / "long.txt")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ANCHOR not in preview_file(str(path))["text"]
    focused = preview_file(str(path), anchor=ANCHOR)
    assert focused["focus_status"] == "matched" and ANCHOR in focused["text"]
    start, end = focused["focus_spans"][0]
    assert focused["text"][start:end] == ANCHOR
    assert focused["pages"] <= 7 and len(focused["text"]) <= 3500
    second = preview_file(str(path), page=1, anchor=ANCHOR)
    assert second["page"] == 1 and second["text"] != focused["text"]
    assert second["focus_status"] == "matched" and not second["focus_spans"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    path.write_text("当前文件已修改，原来的段落不在这里。", encoding="utf-8")
    missing = preview_file(str(path), anchor=ANCHOR)
    assert missing["focus_status"] == "not_found" and "已修改" in missing["text"]
    assert not missing["focus_spans"]


def test_focus_does_not_scan_beyond_read_budget_or_accept_unbounded_anchor(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("A" * (2 * 1024 * 1024 + 100) + ANCHOR, encoding="utf-8")
    result = preview_file(str(path), anchor=ANCHOR)
    assert result["focus_status"] == "not_found" and "读取范围" in result["note"]
    assert len(result["text"]) <= 3500
    for bad in (None, [], "字" * 513):
        with pytest.raises(PreviewError, match="片段"):
            preview_file(str(path), anchor=bad)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "gb18030"])
def test_byte_boundary_is_not_mistaken_for_wrong_encoding(tmp_path, encoding):
    path = tmp_path / "encoding.txt"
    path.write_bytes(("中文资料" * 100).encode(encoding))
    value, count = _read_text(path, 17, path.stat().st_size)
    assert count == 17 and value and set(value) <= set("中文资料")


def test_docx_late_paragraph_and_late_table_row_can_be_reached(tmp_path):
    path = tmp_path / "long.docx"
    doc = Document()
    for _ in range(400):
        doc.add_paragraph("常规背景资料及说明。" * 10)
    doc.add_paragraph(ANCHOR)
    table = doc.add_table(rows=100, cols=3)
    for i, row in enumerate(table.rows):
        row.cells[0].text, row.cells[2].text = str(i + 1), "示例条目"
    table.rows[90].cells[2].text = "冷却装置每月清洗过滤网"
    doc.save(path)
    original = path.read_bytes()
    assert ANCHOR not in preview_file(str(path))["text"]
    assert ANCHOR in preview_file(str(path), anchor=ANCHOR)["text"]
    result = preview_file(str(path), anchor="91 | 冷却装置每月清洗过滤网")
    assert result["focus_status"] == "matched" and "冷却装置" in result["text"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["xlsx", "csv"])
def test_late_table_rows_columns_and_empty_cells_keep_coordinates(tmp_path, kind):
    rows = [[f"项目{i}", "", "普通说明", *([""] * 21), f"条目{i}"] for i in range(1, 181)]
    rows[149][24] = ANCHOR
    path = tmp_path / ("table." + kind)
    if kind == "xlsx":
        book = Workbook()
        book.active.title = "目标表"
        for row in rows:
            book.active.append(row)
        book.save(path)
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
    before = path.read_bytes()
    assert not any(ANCHOR in cell for row in preview_file(str(path))["rows"] for cell in row)
    result = preview_file(str(path), anchor=ANCHOR, sheet="目标表")
    assert result["focus_status"] == "matched"
    i, j = result["focus_cells"][0]
    assert result["row_start"] + i == 150 and result["column_start"] + j == 25
    assert ANCHOR in result["rows"][i][j]
    assert len(result["rows"]) <= 60 and all(len(row) <= 12 for row in result["rows"])
    assert path.read_bytes() == before
    db = LocalMindDB(tmp_path / "index.db")
    try:
        IndexingService(db, embedding_service=EmbeddingService(force_hash=True),
                        ocr_service=OcrService(enabled=False)).index_path(path)
        hit = searcher(db, mode="keyword").search("备用电池")[0]
        actual = preview_file(hit.path, anchor=hit.preview_anchor, sheet=hit.sheet_name)
        assert actual["focus_status"] == "matched"
        i, j = actual["focus_cells"][0]
        assert actual["row_start"] + i == 150 and actual["column_start"] + j == 25
        assert ANCHOR in actual["rows"][i][j]
    finally:
        db.close()


def test_table_ui_coordinates_highlights_and_no_stale_cells(preview_app, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFontDatabase
    from localmind.desktop.theme import APP_STYLESHEET

    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    path = tmp_path / "table.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([[str(i), "", ANCHOR if i == 150 else "普通说明"]
                                     for i in range(1, 181)])
    pane = FilePreview()
    pane.setStyleSheet(APP_STYLESHEET)
    pane.resize(380, 340)
    pane.show()
    try:
        pane.set_target(str(path), anchor=ANCHOR)
        wait(preview_app, lambda: pane._payload is not None)
        assert pane.table.verticalHeader().isVisible()
        i, j = pane._payload["focus_cells"][0]
        assert pane.table.verticalHeaderItem(i).text() == "150"
        assert pane.table.horizontalHeaderItem(j).text() == "3"
        item = pane.table.item(i, j)
        assert item.background().color().name() == "#fff2a8"
        assert not item.isSelected() and ANCHOR in item.toolTip()
        assert pane.table.viewport().rect().intersects(pane.table.visualItemRect(item))
        pixels = pane.table.viewport().grab().toImage()
        visible = pane.table.visualItemRect(item).intersected(pane.table.viewport().rect())
        assert sum(pixels.pixelColor(x, y).name() == "#fff2a8"
                   for y in range(visible.top(), visible.bottom(), 2)
                   for x in range(visible.left(), visible.right(), 2)) > 100
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            assert pane.grab().save(str(Path(output) / "preview-focus-table.png"))
        payload = dict(pane._payload, rows=[["新首行"], ["新次行", "第二列"]],
                       focus_cells=[], focus_status="not_found", row_start=1, column_start=1)
        pane._display(payload)
        assert pane.table.item(0, 1) is None
        assert pane.table.item(0, 0).background().style() == Qt.BrushStyle.NoBrush
    finally:
        pane.shutdown()
        pane.close()


@pytest.mark.parametrize("focused", [True, False])
def test_large_preview_preserves_focus_mode_and_refresh_target(preview_app, tmp_path, focused):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog

    path = text_fixture(tmp_path / "large.txt")
    pane = FilePreview()
    pane.show()
    observed = []
    failures = []
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(lambda: QApplication.activeModalWidget().reject()
                          if QApplication.activeModalWidget() else None)
    try:
        pane.set_target(str(path), anchor=ANCHOR)
        wait(preview_app, lambda: pane._payload is not None)
        if not focused:
            pane.focus_button.click()
            wait(preview_app, lambda: pane._payload is not None)
        pane.next.click()
        wait(preview_app, lambda: pane._payload is not None and pane._page == 1)
        def inspect():
            dialog = QApplication.activeModalWidget()
            try:
                assert isinstance(dialog, QDialog)
                child = dialog.findChild(FilePreview)
                assert child._focus_mode == focused and child.focus_button.isChecked() == focused
                assert child._target == pane._target and child._page == 1
                child.refresh.click()
                wait(preview_app, lambda: child._payload is not None)
                assert child._page == 1 and bool(child._payload["focus_status"]) == focused
                observed.append(True)
            except Exception as error:
                failures.append(error)
            finally:
                if dialog:
                    dialog.accept()
        guard.start(15000)
        QTimer.singleShot(100, inspect)
        pane.enlarge.click()
        assert observed and not failures, failures
    finally:
        guard.stop()
        pane.shutdown()
        pane.close()


def test_table_long_cell_excerpts_preserve_target_and_row_marker_alignment():
    rows = [["", "文" * 600 + ANCHOR], ["其他", ""]]
    result = table_window(rows, ANCHOR, numbered=True)
    assert ANCHOR in result["rows"][0][1] and result["rows"][0][1].startswith("…")
    assert len(result["rows"][0][1]) <= 301
    result = table_window(rows, "R2: 其他", numbered=True)
    assert result["focus_cells"] == [[1, 0]]


def test_anchor_is_snapshot_body_evidence_not_chat_or_a_later_database_read(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        path = text_fixture(tmp_path / "target.txt")
        target = index_file(db, path, path.read_text(encoding="utf-8"))
        ChatStore(db).ingest(platform="wechat", account="synthetic", source_key="source", conversation="合成组",
                             messages=[ChatMessage("one", "这是季度复盘会上要用的资料。", attachments=[Attachment(path.name, str(path))])])
        service = searcher(db, mode="keyword")
        body = service.search(SearchQuery("低温 备用电池", scope="chat_files", evidence="content"))[0]
        assert body.file_id == target["id"] and "备用电池" in body.preview_anchor
        assert preview_file(body.path, anchor=body.preview_anchor)["focus_status"] == "matched"
        chat = service.search(SearchQuery("季度复盘", scope="chat_files", evidence="chat"))[0]
        assert chat.preview_anchor == ""
        anchor = body.preview_anchor
        db.conn.execute("UPDATE chunks SET chunk_text='新索引片段' WHERE id=?", (body.chunk_id,))
        db.conn.commit()
        assert body.preview_anchor == anchor
    finally:
        db.close()


def test_real_worker_mode_switch_paging_and_same_file_new_anchor(preview_app, tmp_path):
    app = preview_app
    path = text_fixture(tmp_path / "long.txt")
    pane = FilePreview()
    pane.resize(340, 280)
    pane.show()
    try:
        pane.set_target(str(path), anchor=ANCHOR)
        wait(app, lambda: pane._payload is not None)
        assert pane._payload["focus_status"] == "matched" and pane.focus_button.isChecked()
        assert any(ANCHOR == s.cursor.selectedText() for s in pane.text.extraSelections())
        assert 0 <= pane.text.cursorRect().top() < pane.text.viewport().height()
        pane.next.click()
        wait(app, lambda: pane._payload is not None and pane._page == 1)
        assert not pane.text.extraSelections()
        pane.focus_button.click()
        wait(app, lambda: pane._payload is not None)
        assert not pane._focus_mode and ANCHOR not in pane.text.toPlainText()
        pane.focus_button.click()
        wait(app, lambda: pane._payload is not None)
        assert pane._page == 0 and ANCHOR in pane.text.toPlainText()
        pane.set_target(str(path), anchor="不在文件中的旧索引片段")
        wait(app, lambda: pane._payload is not None)
        assert pane._payload["focus_status"] == "not_found"
        assert "未找到" in pane.focus_notice.text()
        pane.clear()
        assert not pane._target and not pane.text.extraSelections() and pane.navigation.isHidden()
    finally:
        pane.shutdown()
        pane.close()


def test_ui_snapshot_target_and_two_widths(preview_app, tmp_path):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from localmind.ui import MainWindow

    app = preview_app
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db = LocalMindDB(tmp_path / "index.db")
    path = text_fixture(tmp_path / "备用电池维护说明.txt")
    index_file(db, path, path.read_text(encoding="utf-8"))
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("search_retrieval_mode", "keyword")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    window = MainWindow(db)
    service = searcher(db, mode="keyword")
    window.search_input.setText("低温 备用电池")
    hits = service.search("低温 备用电池")
    window.show()
    try:
        window._search_finished(hits, service.last_diagnostics, {"total_ms": 15, "index_refresh_ms": 0})
        wait(app, lambda: window.file_preview._payload is not None)
        assert ANCHOR in window.file_preview.text.toPlainText()
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            for widget in (window.file_preview.focus_button, window.file_preview.next,
                           window.file_preview.enlarge, window.file_preview.text, window.file_actions[0]):
                point = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert point.x() <= width and point.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"preview-focus-{width}.png"))
        window.result_table.clearSelection()
        assert window.file_preview._target is None and not window.file_preview.text.extraSelections()
    finally:
        window.close()
        app.processEvents()
