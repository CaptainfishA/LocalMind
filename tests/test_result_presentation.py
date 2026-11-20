import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QTextEdit

from localmind.chats import ChatStore
from localmind.searcher import SearchDiagnostics
from localmind.ui_formatting import display_time, plain_tooltip, result_source_label, source_label
from test_result_lifecycle import screen  # noqa: F401


@pytest.mark.parametrize("code,label", [
    ("chat", "聊天线索"), ("txt", "文本正文"), ("md", "文本正文"),
    ("xlsx", "表格内容"), ("pptx", "幻灯片"), ("pdf", "PDF 正文"),
    ("ocr_image", "图片 OCR"), ("ocr_pdf", "扫描 PDF"), ("ocr_future", "OCR 文本"),
    ("unrecognized", "其他来源"), (None, "其他来源"),
])
def test_source_codes_have_bounded_readable_labels(code, label):
    assert source_label(code) == label


def test_source_summary_uses_actual_contributors_instead_of_only_the_primary_row():
    result = SimpleNamespace(source_type="chat", supporting_evidence=())
    assert result_source_label(result) == "聊天线索"
    result.supporting_evidence = (SimpleNamespace(source_type="docx"),)
    assert result_source_label(result) == "文档正文"
    result.supporting_evidence += (SimpleNamespace(source_type="chat"),)
    assert result_source_label(result) == "正文与聊天"
    result.supporting_evidence = (SimpleNamespace(source_type="pdf"), SimpleNamespace(source_type="ocr_pdf"))
    assert result_source_label(result) == "文件正文"


def test_tooltip_escapes_embedded_tags_and_preserves_literal_text():
    app = QApplication.instance() or QApplication([])
    text = '<img src="https://invalid.example/image"> & <b>合同</b>\n第二行'
    tooltip = plain_tooltip(text)
    assert "<img" not in tooltip and "<b>" not in tooltip
    assert "&lt;img" in tooltip and "&amp;" in tooltip and "<br>" in tooltip
    assert tooltip.startswith("<qt>") and tooltip.endswith("</qt>")
    preview = QTextEdit()
    preview.setHtml(tooltip)
    assert preview.toPlainText() == text
    preview.close()
    app.processEvents()


def test_content_source_and_file_metadata_are_separate_but_keep_the_same_target(screen, monkeypatch):
    _, window, _, first, _, _ = screen
    assert window.result_table.horizontalHeaderItem(0).text() == "文件 / 消息"
    assert window.result_table.item(0, 0).text() == first["name"]
    assert window.result_table.item(0, 1).text() == "聊天线索"
    assert window.result_table.item(0, 3).text() == f"{window.selected_result().score:.3f}"
    assert "正确概率" in window.result_table.horizontalHeaderItem(3).toolTip()
    assert "路径：" not in window.preview.toPlainText()
    assert "关键词：" not in window.preview.toPlainText()
    assert first["path"] in window.info_preview.toPlainText()
    assert "排序分：" in window.info_preview.toPlainText()
    assert "模拟项目群" in window.context_preview.toPlainText()
    assert window.preview_tabs.tabText(2) == "文件信息" and window.preview_tabs.isTabEnabled(2)
    actions = []
    monkeypatch.setattr(window, "perform_file_action", lambda *args: actions.append(args))
    for tab in (0, 1, 2):
        window.preview_tabs.setCurrentIndex(tab)
        window.copy_selected_path()
    assert actions == [(first["path"], "copy")] * 3


def test_long_literal_names_and_metadata_fit_without_expanding_rows_or_rendering_markup(screen):
    app, window, *_ = screen
    original = window.selected_result()
    name = "<b>季度资料</b>" + "VeryLongFileName" * 25 + ".pdf"
    modified = "2026-10-01T01:30:45.123456Z"
    value = replace(original, name=name, ext="VeryLongExtension" * 8,
                    match_reason='<img src="https://invalid.example/image">普通文字',
                    modified_at=modified, file_state="changed")
    diagnostics = SearchDiagnostics(
        query_terms=[], keyword_candidates=1, semantic_candidates=0, total_candidates=1,
        indexed_embedding_model=None, active_embedding_model=None,
        semantic_status="synthetic", reranker_status="已关闭",
        retrieval_mode="keyword", scope="chat_files",
    )
    window._search_finished([value], diagnostics, {"total_ms": 1, "index_refresh_ms": 0})
    assert window.detail_title.text() == name and name in window.info_preview.toPlainText()
    assert name not in window.preview.toPlainText()
    assert "&lt;b&gt;" in window.result_table.item(0, 0).toolTip()
    assert "<img " not in window.result_table.item(0, 4).toolTip()
    assert value.match_reason in window.info_preview.toPlainText()
    assert window.result_table.item(0, 5).text() == display_time(modified)
    assert modified in window.result_table.item(0, 5).toolTip()
    assert modified in window.info_preview.toPlainText()
    assert window.detail_warning.isVisible() and "文件已变化" in window.detail_warning.text()
    assert window.detail_warning.textFormat() == Qt.TextFormat.PlainText
    for width, height in ((980, 650), (1240, 800)):
        window.resize(width, height)
        for tab in (0, 1, 2):
            window.preview_tabs.setCurrentIndex(tab)
            app.processEvents()
            assert window.size().width() == width and window.size().height() == height
            assert window.result_table.rowHeight(0) == 70
            for widget in (window.preview_tabs, window.detail_warning, window.search_button, window.file_actions[-1]):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"result-details-{width}-tab{tab}.png"))


def test_stale_warning_is_visible_across_tabs_and_rebind_cannot_display_new_target(screen):
    app, window, db, first, other, attachment = screen
    window.selected_result().file_state = "changed"
    window.search_index_dirty.emit()
    for tab in (0, 1, 2):
        window.preview_tabs.setCurrentIndex(tab)
        app.processEvents()
        assert window.detail_warning.isVisible()
        assert "文件已变化" in window.detail_warning.text()
        assert "上次搜索快照" in window.detail_warning.text()
    window.preview_tabs.setCurrentIndex(1)
    ChatStore(db).bind(attachment, other["id"])
    window.search_index_dirty.emit()
    assert not window.context_preview.toPlainText()
    assert not window.preview_tabs.isTabEnabled(1)
    assert window.preview_tabs.currentIndex() != 1
    assert "关联已变化" in window.detail_warning.text()
    assert first["path"] in window.info_preview.toPlainText()
    assert other["path"] not in window.info_preview.toPlainText()
    assert not window.index_result_button.isEnabled()


def test_deselection_clears_details_and_disables_file_actions_without_forgetting_results(screen, monkeypatch):
    _, window, _, first, *_ = screen
    results = window.results
    called = []
    monkeypatch.setattr(window, "perform_file_action", lambda *args: called.append(args))
    window.result_table.clearSelection()
    assert window.selected_result() is None
    assert not window.preview.toPlainText() and not window.info_preview.toPlainText()
    assert not window.context_preview.toPlainText() and not window.detail_warning.isVisible()
    assert not window.preview_tabs.isTabEnabled(2)
    assert all(not button.isEnabled() for button in window.file_actions)
    window.open_selected_file()
    assert not called and window.results is results
    window.result_table.selectRow(0)
    assert window.selected_result().path == first["path"]
    assert window.preview_tabs.isTabEnabled(2)
    assert first["path"] in window.info_preview.toPlainText()


def test_empty_results_and_clear_index_remove_file_metadata(screen, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from test_result_lifecycle import wait_for

    app, window, _, first, *_ = screen
    window.preview_tabs.setCurrentIndex(2)
    window.search_input.setText("MISSING-827462")
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert not window.results and not window.info_preview.toPlainText()
    assert window.preview_tabs.currentIndex() == 0 and not window.preview_tabs.isTabEnabled(2)
    assert first["path"] not in window.preview.toPlainText()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.clear_index()
    assert not window.preview.toPlainText() and not window.info_preview.toPlainText()
    assert not window.detail_warning.isVisible()


def test_external_update_preserves_metadata_tab_and_scroll_position(screen):
    app, window, db, _, other, _ = screen
    window.selected_result().match_reason = "合成匹配原因\n" * 100
    window.show_selected_result()
    window.preview_tabs.setCurrentIndex(2)
    window.technical_toggle.click()
    app.processEvents()
    window.info_preview.verticalScrollBar().setValue(100)
    position = window.info_preview.verticalScrollBar().value()
    assert position > 0
    with db.conn:
        db.conn.execute("UPDATE files SET size=size+1 WHERE id=?", (other["id"],))
    window._poll_index_updates()
    assert window.preview_tabs.currentIndex() == 2
    assert window.info_preview.verticalScrollBar().value() == position
    assert window.detail_warning.isVisible()
    app.processEvents()
    assert window.info_preview.verticalScrollBar().value() == position
