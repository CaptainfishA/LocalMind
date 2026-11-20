import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from localmind.models import SearchQuery
from localmind.searcher import SearchCancelled, SearchService
from localmind.ui import SearchWorker
from test_clause_semantics import QUERY, corpus  # noqa: F401
from test_result_lifecycle import screen, wait_for  # noqa: F401
from test_search_expansion import start
from test_search_snapshot import setup  # noqa: F401


def test_cancel_before_query_does_not_start_models_or_reconcile_links(setup, monkeypatch):
    reader, _, service, index, *_ = setup
    stop = Event()
    stop.set()
    service.should_cancel = stop.is_set
    calls = []
    monkeypatch.setattr(service, "_prepare_links", lambda: calls.append("links"))
    monkeypatch.setattr(service.embedding_service, "encode_query", lambda text: calls.append("embedding"))
    with pytest.raises(SearchCancelled):
        service.search("付款")
    assert not calls and not reader.conn.in_transaction and index.rebuilds == 1


@pytest.mark.parametrize("stage", ["embedding", "keyword", "vector", "companion", "rerank", "file_check"])
def test_stage_cancel_skips_following_work_releases_snapshot_and_reuses_valid_index(setup, monkeypatch, stage):
    import localmind.searcher as module

    reader, writer, service, index, *_ = setup
    service.retrieval_mode = "hybrid"
    original_results = service.search("付款")
    original_diagnostics = service.last_diagnostics
    original_candidates = service.last_candidate_file_ids
    stop, later = Event(), []
    service.should_cancel = stop.is_set
    originals = {
        "embedding": (service.embedding_service, "encode_query"),
        "keyword": (service, "_keyword_candidates"),
        "vector": (index, "search"),
        "companion": (service, "_extend_joint_candidates"),
        "file_check": (module, "annotate_results"),
    }
    if stage == "rerank":
        service.reranker_service = SimpleNamespace(
            enabled=True, recommended_top_k=30, model_name="synthetic", load_error=None,
            runtime=SimpleNamespace(name="cpu"),
            rerank=lambda query, passages: stop.set() or [.8] * len(passages),
        )
    else:
        owner, name = originals[stage]
        original = getattr(owner, name)

        def cancelling(*args, **kwargs):
            result = original(*args, **kwargs)
            stop.set()
            return result

        monkeypatch.setattr(owner, name, cancelling)
    if stage not in {"rerank", "file_check"}:
        monkeypatch.setattr(service, "_rank_candidates", lambda batch: later.append("ranking"))
    elif stage == "rerank":
        monkeypatch.setattr(module, "annotate_results", lambda *args: later.append("file_check"))
    with pytest.raises(SearchCancelled):
        service.search("付款")
    assert not later and not reader.conn.in_transaction
    assert writer.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    assert service.last_diagnostics is original_diagnostics
    assert service.last_candidate_file_ids is original_candidates
    assert index.rebuilds == 1
    monkeypatch.undo()
    service.reranker_service.enabled = False
    stop.clear()
    assert [r.file_id for r in service.search("付款")] == [r.file_id for r in original_results]
    assert index.rebuilds == 1


def test_cancel_rolls_back_temp_scope_but_does_not_commit_caller_transaction(setup, monkeypatch):
    reader, writer, service, _, fid, *_ = setup
    stop = Event()
    service.should_cancel = stop.is_set
    original_scope = service._set_scope
    with reader.conn:
        reader.conn.execute("UPDATE files SET name='uncommitted.txt' WHERE id=?", (fid,))

        def after_scope(query):
            original_scope(query)
            stop.set()

        monkeypatch.setattr(service, "_set_scope", after_scope)
        with pytest.raises(SearchCancelled):
            service.search(SearchQuery("付款", scope="files"))
        assert reader.conn.in_transaction
        assert writer.conn.execute("SELECT name FROM files WHERE id=?", (fid,)).fetchone()[0] != "uncommitted.txt"
        assert service._scope_key is None and service._vector_revision is None
        reader.conn.rollback()
    monkeypatch.undo()
    stop.clear()
    assert service.search(SearchQuery("付款", scope="files"))
    assert not reader.conn.in_transaction


def test_cancel_during_scope_install_reinstalls_scope_on_next_query(setup, monkeypatch):
    reader, _, service, index, *_ = setup
    stop = Event()
    service.should_cancel = stop.is_set
    original_scope = service._set_scope

    def after_scope(query):
        original_scope(query)
        stop.set()

    monkeypatch.setattr(service, "_set_scope", after_scope)
    with pytest.raises(SearchCancelled):
        service.search(SearchQuery("付款", scope="files", path_contains="not-present"))
    assert service._scope_key is None and not reader.conn.in_transaction
    monkeypatch.undo()
    stop.clear()
    assert service.search(SearchQuery("付款", scope="files"))
    assert index.rebuilds == 1


def test_cancel_after_fts_does_not_start_substring_lookup(setup, monkeypatch):
    reader, _, service, _, *_ = setup
    stop, calls = Event(), []
    service.should_cancel = stop.is_set
    service.retrieval_mode = "keyword"
    original = reader.search_fts

    def stop_after_fts(*args, **kwargs):
        rows = original(*args, **kwargs)
        stop.set()
        return rows

    monkeypatch.setattr(reader, "search_fts", stop_after_fts)
    monkeypatch.setattr(reader, "search_like_terms", lambda *args, **kwargs: calls.append(True) or [])
    with pytest.raises(SearchCancelled):
        service.search("payment")
    assert stop.is_set() and not calls and not reader.conn.in_transaction


def test_cancel_in_clause_recall_is_not_swallowed_as_semantic_fallback(corpus, monkeypatch):
    _, service, _, index, *_ = corpus
    stop = Event()
    service.should_cancel = stop.is_set
    service.enable_clause_semantics = True
    original = index.search
    calls = []

    def cancel_second(*args, **kwargs):
        calls.append(True)
        rows = original(*args, **kwargs)
        if len(calls) == 2:
            stop.set()
        return rows

    monkeypatch.setattr(index, "search", cancel_second)
    with pytest.raises(SearchCancelled):
        service.search(SearchQuery(QUERY, scope="chat_files"))
    assert len(calls) == 2
    assert not service.db.conn.in_transaction


def test_worker_preserves_cancellation_before_queued_request_starts(tmp_path):
    worker = SearchWorker(str(tmp_path / "not-created.db"), str(tmp_path / "cache"), "auto", "cpu")
    cancelled, failures, results = [], [], []
    worker.cancelled.connect(cancelled.append)
    worker.failed.connect(failures.append)
    worker.finished.connect(lambda *args: results.append(args))
    worker.cancel_event.set()
    worker.process({"query": SearchQuery("query"), "enable_reranker": False})
    assert len(cancelled) == 1 and not failures and not results
    assert worker.service is None and worker.db is None
    assert not (tmp_path / "not-created.db").exists()


@pytest.mark.parametrize("fail_after_release", [False, True])
def test_ui_waits_for_backend_to_end_keeps_previous_results_and_can_retry(screen, monkeypatch, fail_after_release):
    app, window = start(screen)
    previous, request = window.results, window._displayed_request
    selected = window._result_identity(window.selected_result())
    service = window.search_worker.service
    entered, release = Event(), Event()
    original_search = SearchService.search
    messages = []

    def blocked(current, query):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("test worker timed out")
        if fail_after_release:
            raise RuntimeError("synthetic error after cancellation")
        return original_search(current, query)

    monkeypatch.setattr(SearchService, "search", blocked)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: messages.append(args))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args))
    window.expand_search()
    try:
        wait_for(app, entered.is_set)
        assert window.cancel_search_button.isEnabled()
        if fail_after_release:
            window.cancel_search_button.click()
        else:
            window.activateWindow()
            app.processEvents()
            QTest.keyClick(window, Qt.Key.Key_Escape)
            wait_for(app, window.search_worker.cancel_event.is_set)
        assert window.search_worker.cancel_event.is_set()
        assert window.search_busy and not window.search_button.isEnabled()
        assert not window.cancel_search_button.isEnabled()
        assert not window.expand_results_button.isEnabled()
        assert "正在取消" in window.search_status.text()
        assert "已取消" not in window.search_status.text()
        assert window.results is previous and window._displayed_request is request
        assert not window.close() and window.search_thread.isRunning()
        window.run_search()
        assert window._pending_request.query.limit == 50
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            for widget in (window.search_button, window.cancel_search_button, window.preview, window.file_actions[-1]):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output and not fail_after_release:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"cancelling-search-{width}.png"))
        messages.clear()
    finally:
        release.set()
    wait_for(app, lambda: not window.search_busy)
    assert not messages
    assert "搜索已取消" in window.search_status.text()
    assert window.results is previous and window._displayed_request is request
    assert window._result_identity(window.selected_result()) == selected
    assert window.search_worker.service is service
    assert window.expand_results_button.isEnabled()
    monkeypatch.setattr(SearchService, "search", original_search)
    window.expand_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert not window.search_worker.cancel_event.is_set()
    assert window._displayed_request.query.limit == 50


def test_late_success_after_cancel_request_is_not_applied(screen):
    _, window, *_ = screen
    previous = window.results
    window.search_requested.disconnect(window.search_worker.process)
    window.run_search()
    window.cancel_search()
    window._search_finished([], SimpleNamespace(), {"total_ms": 7})
    assert window.results is previous
    assert not window.search_busy and "已取消" in window.search_status.text()


def test_cancelling_first_query_does_not_leave_a_phantom_result_snapshot(screen):
    _, window, *_ = screen
    window.results = []
    window._displayed_request = None
    window._results_db_revision = None
    window._has_search = False
    window.search_requested.disconnect(window.search_worker.process)
    window.run_search()
    window.cancel_search()
    window._search_cancelled(1)
    assert not window._has_search and not window.search_busy
    assert not window.result_notice.isVisible()
    assert not window.expand_results_button.isEnabled()


def test_cancellation_does_not_hide_index_update_notice(screen, monkeypatch):
    app, window = start(screen)
    entered, release = Event(), Event()
    original_search = SearchService.search

    def blocked(service, query):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("test worker timed out")
        return original_search(service, query)

    monkeypatch.setattr(SearchService, "search", blocked)
    window.run_search()
    try:
        wait_for(app, entered.is_set)
        window.cancel_search()
        window.search_index_dirty.emit()
    finally:
        release.set()
    wait_for(app, lambda: not window.search_busy)
    assert window.result_notice.isVisible() and window._results_stale
    assert not window.expand_results_button.isEnabled()
