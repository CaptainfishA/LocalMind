import os
from datetime import datetime
from pathlib import Path
from threading import Event

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMessageBox

from localmind.chats import Attachment, ChatMessage
from localmind.searcher import SearchService
from test_result_lifecycle import screen, wait_for  # noqa: F401
from test_weflow_v04 import index_file, ingest


def start(screen):
    app, window, *_ = screen
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert window._displayed_request.query.limit == 20
    assert window.expand_results_button.isEnabled()
    return app, window


def test_expansion_reveals_more_files_with_original_scope_and_reuses_index(screen, monkeypatch):
    app, window, db, *_ = screen
    accepted = set()
    for group, count in (("甲项目群", 110), ("乙项目群", 20)):
        messages = []
        for i in range(count):
            path = db.db_path.parent / group / f"material-{i:03d}.txt"
            record = index_file(db, path, "EXPAND-8274 审核流程与登记事项")
            if group == "甲项目群":
                accepted.add(record["id"])
            messages.append(ChatMessage(str(i), "EXPAND-8274 审核材料",
                                        sender="模拟经办人", attachments=[Attachment(path.name, str(path))]))
        ingest(db, messages, source=group, conversation=group)
    window.search_index_dirty.emit()
    window.search_input.setText("EXPAND-8274")
    window.conversation_filter.setText("甲项目群")
    window.sender_filter.setText("模拟经办人")
    window.type_filter.setCurrentIndex(window.type_filter.findData("txt"))
    start(screen)
    assert len(window.results) == 20
    initial_ids = {r.file_id for r in window.results}
    service = window.search_worker.service
    rebuilds = []
    original_rebuild = service.vector_index.rebuild
    monkeypatch.setattr(service.vector_index, "rebuild", lambda *args: rebuilds.append(True) or original_rebuild(*args))
    window.result_table.selectRow(7)
    selected = window._result_identity(window.selected_result())
    requests = []
    window.search_requested.connect(requests.append)
    for limit in (50, 100):
        window.expand_results_button.click()
        assert window.search_busy and not window.expand_results_button.isEnabled()
        assert not window.enable_reranker.isEnabled()
        wait_for(app, lambda: not window.search_busy)
        assert len(window.results) == limit
        assert {r.file_id for r in window.results} <= accepted
        assert {r.file_id for r in window.results} - initial_ids
        assert window._result_identity(window.selected_result()) == selected
        assert window.search_worker.service is service and not rebuilds
        assert window.result_count_label.text() == f"{limit} 项 · 上限 {limit}"
        request = requests[-1]
        assert request["query"].limit == limit
        assert request["query"].conversation == "甲项目群"
        assert request["query"].sender == "模拟经办人"
        assert request["query"].file_type == "txt" and request["query"].scope == "chat_files"
        assert request["retrieval_mode"] == "keyword" and not request["enable_reranker"]
    assert not window.expand_results_button.isEnabled()
    assert "不表示所有" in window.expand_results_button.toolTip()
    assert "全部" not in window.result_count_label.text()
    with db.conn:
        db.conn.execute("UPDATE files SET size=size+1 WHERE id=?", (min(accepted),))
    window._poll_index_updates()
    window.refresh_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert len(window.results) == 100 and window._displayed_request.query.limit == 100
    assert window._result_identity(window.selected_result()) == selected
    assert not rebuilds and window.search_worker.service is service
    for width, height in ((980, 650), (1240, 800)):
        window.resize(width, height)
        app.processEvents()
        for widget in (window.result_count_label, window.expand_results_button, window.preview, window.file_actions[-1]):
            corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
            assert corner.x() <= width and corner.y() <= height
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(Path(output) / f"expanded-results-{width}.png"))
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert len(window.results) == 20 and window._displayed_request.query.limit == 20


@pytest.mark.parametrize("control", ["query", "type", "scope", "evidence", "conversation", "sender", "mode", "reranker"])
def test_changed_controls_block_expansion_until_restored(screen, control):
    _, window = start(screen)
    widget = {
        "query": window.search_input, "type": window.type_filter, "scope": window.scope_filter,
        "evidence": window.evidence_filter, "conversation": window.conversation_filter,
        "sender": window.sender_filter, "mode": window.retrieval_buttons["hybrid"],
        "reranker": window.enable_reranker,
    }[control]
    if control in {"query", "conversation", "sender"}:
        previous = widget.text()
        widget.setText("changed")
        restore = lambda: widget.setText(previous)
    elif control in {"type", "scope", "evidence"}:
        previous = widget.currentIndex()
        widget.setCurrentIndex((previous + 1) % widget.count())
        restore = lambda: widget.setCurrentIndex(previous)
    elif control == "mode":
        widget.click()
        restore = window.retrieval_buttons["keyword"].click
    else:
        previous = widget.isChecked()
        widget.setChecked(not previous)
        restore = lambda: widget.setChecked(previous)
    assert not window.expand_results_button.isEnabled()
    assert "已修改" in window.expand_results_button.toolTip()
    window.expand_search()
    assert not window.search_busy
    restore()
    assert window.expand_results_button.isEnabled()


def test_failed_expansion_keeps_accepted_limit_selection_and_can_retry(screen, monkeypatch):
    app, window = start(screen)
    previous = window.results
    selected = window._result_identity(window.selected_result())
    original = SearchService.search
    errors = []

    def fail(service, query):
        if query.limit == 50:
            raise RuntimeError("synthetic expansion failure")
        return original(service, query)

    monkeypatch.setattr(SearchService, "search", fail)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(True))
    window.expand_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert errors and window.results is previous
    assert window._displayed_request.query.limit == 20
    assert window._result_identity(window.selected_result()) == selected
    assert window.expand_results_button.isEnabled()
    monkeypatch.setattr(SearchService, "search", original)
    window.expand_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert window._displayed_request.query.limit == 50
    assert window._result_identity(window.selected_result()) == selected


def test_short_or_empty_result_set_is_not_claimed_exhaustive(screen):
    app, window = start(screen)
    assert len(window.results) < 20 and window.expand_results_button.isEnabled()
    window.search_input.setText("UNMATCHED-129712")
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert not window.results
    assert window.expand_results_button.isEnabled()
    window.expand_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert not window.results and window.result_count_label.text() == "0 项 · 上限 50"


def test_external_commit_blocks_expansion_until_explicit_fresh_search(screen):
    app, window = start(screen)
    db, other = screen[2], screen[4]
    from localmind.db import LocalMindDB

    writer = LocalMindDB(db.db_path)
    try:
        with writer.conn:
            writer.conn.execute("UPDATE files SET size=size+1 WHERE id=?", (other["id"],))
    finally:
        writer.close()
    window.expand_search()
    assert not window.search_busy and not window.expand_results_button.isEnabled()
    assert window.result_notice.isVisible()
    assert window._displayed_request.query.limit == 20
    window.refresh_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert window.expand_results_button.isEnabled()


def test_stale_inflight_expansion_is_not_applied(screen, monkeypatch):
    app, window = start(screen)
    previous = window.results
    entered, release = Event(), Event()
    original = SearchService.search

    def blocked(service, query):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("test worker timed out")
        return original(service, query)

    monkeypatch.setattr(SearchService, "search", blocked)
    window.expand_results_button.click()
    try:
        wait_for(app, entered.is_set)
        window.search_index_dirty.emit()
        assert not window.expand_results_button.isEnabled()
    finally:
        release.set()
    wait_for(app, lambda: not window.search_busy)
    assert window.results is previous and window._displayed_request.query.limit == 20
    assert not window.expand_results_button.isEnabled()


def test_relative_date_scope_is_frozen_for_expansion_not_new_search(screen, monkeypatch):
    import localmind.searcher as module
    from localmind.query_processing import process_query

    app, window, *_ = screen
    now = [datetime(2026, 10, 21)]
    monkeypatch.setattr(module, "process_query", lambda text: process_query(text, now=now[0]))
    window.search_input.setText("上个月验收合同")
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("files"))
    start(screen)
    first = window._displayed_request.query
    assert first.modified_after.startswith("2026-09")
    assert first.modified_before.startswith("2026-10")
    now[0] = datetime(2026, 11, 2)
    window.expand_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    extended = window._displayed_request.query
    assert (extended.modified_after, extended.modified_before) == (first.modified_after, first.modified_before)
    assert window.expand_results_button.isEnabled()
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert window._displayed_request.query.modified_after.startswith("2026-10")


def test_chat_task_and_clear_index_update_expansion_controls(screen, monkeypatch):
    app, window = start(screen)
    window.chat_panel.requested.disconnect(window.chat_panel.worker.process)
    assert window.chat_panel.launch("sessions")
    assert not window.expand_results_button.isEnabled()
    window.chat_panel.finished({"kind": "sessions", "seconds": 0, "error": "synthetic"})
    assert window.expand_results_button.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.clear_index()
    assert not window.expand_results_button.isEnabled()
    assert window._displayed_request is None and window.result_count_label.text() == "0 项"
