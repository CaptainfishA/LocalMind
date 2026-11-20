import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import pytest
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from localmind.chats import ChatStore
from localmind.db import LocalMindDB
from localmind.index_notifications import IndexRevisionProbe
from localmind.searcher import SearchService
from test_result_lifecycle import screen, wait_for  # noqa: F401
from test_weflow_v04 import index_file


def test_idle_results_notice_another_connection_rebinding_without_a_new_search(screen):
    _, window, db, first, other, attachment = screen
    original = window.results
    writer = LocalMindDB(db.db_path)
    try:
        ChatStore(writer).bind(attachment, other["id"])
    finally:
        writer.close()
    QTest.qWait(2300)
    assert window.result_notice.isVisible()
    assert window.results is original
    assert window.selected_result().path == first["path"]
    assert not window.context_preview.toPlainText()
    assert not window.index_result_button.isEnabled()
    assert window.search_worker.service is None


def test_probe_is_read_only_and_missing_path_is_not_created(tmp_path):
    path = tmp_path / "missing.db"
    probe = IndexRevisionProbe(path)
    with pytest.raises(sqlite3.OperationalError):
        probe.read()
    assert not path.exists() and probe._conn is None
    db = LocalMindDB(path)
    try:
        before = db.index_revisions()
        assert probe.read() == before[0]
        assert probe._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            probe._conn.execute("DELETE FROM index_revisions")
        assert db.index_revisions() == before
        assert not probe._conn.in_transaction
    finally:
        probe.close()
        db.close()


def test_probe_sees_only_commits_and_does_not_pin_wal(tmp_path):
    db = LocalMindDB(tmp_path / "读 only #1.db")
    probe = IndexRevisionProbe(db.db_path)
    try:
        old = probe.read()
        db.conn.execute("UPDATE index_revisions SET revision=revision+1")
        assert probe.read() == old
        db.conn.rollback()
        assert probe.read() == old
        with db.conn:
            db.conn.execute("UPDATE index_revisions SET revision=revision+2")
        assert probe.read() == old + 2
        assert db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        assert not probe._conn.in_transaction
    finally:
        probe.close()
        db.close()


def test_probe_does_not_wait_for_exclusive_lock_and_can_recover(tmp_path):
    path = tmp_path / "locked.db"
    writer = sqlite3.connect(path)
    writer.execute("CREATE TABLE index_revisions(id INTEGER PRIMARY KEY,revision INTEGER)")
    writer.execute("INSERT INTO index_revisions VALUES(1,1)")
    writer.commit()
    probe = IndexRevisionProbe(path)
    try:
        assert probe.read() == 1
        writer.execute("BEGIN EXCLUSIVE")
        started = time.perf_counter()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            probe.read()
        assert time.perf_counter() - started < .5
        assert probe._conn is None
        writer.rollback()
        assert probe.read() == 1
        writer.execute("DELETE FROM index_revisions")
        writer.commit()
        with pytest.raises(sqlite3.DatabaseError, match="Missing"):
            probe.read()
        assert probe._conn is None
        writer.execute("INSERT INTO index_revisions VALUES(1,3)")
        writer.commit()
        assert probe.read() == 3
    finally:
        probe.close()
        writer.close()


def test_ordinary_settings_do_not_mark_results_stale_or_refresh_service(screen, monkeypatch):
    _, window, db, _, _, _ = screen
    writer = LocalMindDB(db.db_path)
    try:
        writer.set_setting("synthetic_ui_preference", "1")
    finally:
        writer.close()
    repaint = []
    monkeypatch.setattr(window, "show_selected_result", lambda: repaint.append(True))
    for _ in range(10):
        window._poll_index_updates()
    assert not window.result_notice.isVisible()
    assert not repaint and window.search_worker.service is None


def test_independent_process_commit_is_seen_without_restarting_window(screen):
    _, window, db, _, other, _ = screen
    results = window.results
    original_revision = window._results_db_revision
    code = (
        "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
        "c.execute('UPDATE files SET size=size+1 WHERE id=?',(int(sys.argv[2]),)); "
        "c.commit(); c.close()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(db.db_path), str(other["id"])],
        capture_output=True, text=True, timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert completed.returncode == 0, completed.stderr
    window._poll_index_updates()
    assert window.result_notice.isVisible()
    assert window._last_index_check_revision > original_revision
    assert window.results is results and window.search_worker.service is None


def test_multiple_external_updates_revalidate_sources_even_after_notice_is_visible(screen, monkeypatch):
    _, window, db, first, other, attachment = screen
    writer = LocalMindDB(db.db_path)
    revision = window._index_revision
    requested = []
    window.search_requested.connect(requested.append)
    try:
        with writer.conn:
            writer.conn.execute("UPDATE files SET size=size+1 WHERE id=?", (other["id"],))
        window._poll_index_updates()
        assert window.result_notice.isVisible()
        assert window.context_preview.toPlainText()
        assert window.index_result_button.isEnabled()
        ChatStore(writer).bind(attachment, other["id"])
        window._poll_index_updates()
        assert not window.context_preview.toPlainText()
        assert not window.index_result_button.isEnabled()
        assert window.selected_result().path == first["path"]
        calls = []
        monkeypatch.setattr(window, "show_selected_result", lambda: calls.append(True))
        for _ in range(10):
            window._poll_index_updates()
        assert not calls and not requested and window._index_revision == revision
    finally:
        writer.close()


@pytest.mark.parametrize("state", ["hidden", "minimized", "no_search", "transaction"])
def test_inactive_or_caller_owned_state_skips_database_probe(screen, monkeypatch, state):
    _, window, db, _, _, _ = screen
    calls = []
    monkeypatch.setattr(window.index_revision_probe, "read", lambda: calls.append(True) or 0)
    if state == "hidden":
        window.hide()
    elif state == "minimized":
        window.showMinimized()
    elif state == "no_search":
        window._has_search = False
    else:
        db.conn.execute("BEGIN")
    try:
        window._poll_index_updates()
        assert not calls
    finally:
        if state == "transaction":
            db.conn.rollback()


@pytest.mark.parametrize("event", ["show", "activate", "restore"])
def test_window_return_checks_immediately_without_waiting_for_periodic_tick(screen, event):
    app, window, db, _, other, attachment = screen
    window.index_watch_timer.stop()
    if event == "show":
        window.hide()
    elif event == "restore":
        window.showMinimized()
        app.processEvents()
    writer = LocalMindDB(db.db_path)
    try:
        ChatStore(writer).bind(attachment, other["id"])
    finally:
        writer.close()
    if event == "show":
        window.show()
    elif event == "restore":
        window.showNormal()
    else:
        QApplication.sendEvent(window, QEvent(QEvent.Type.ActivationChange))
    app.processEvents()
    assert window.result_notice.isVisible()
    assert not window.index_result_button.isEnabled()


def test_unavailable_revision_is_not_reported_as_no_change_and_recovery_clears_warning(screen, monkeypatch):
    _, window, _, _, _, _ = screen
    original = window.index_revision_probe.read
    results = window.results

    def locked():
        raise sqlite3.OperationalError("synthetic locked database")

    monkeypatch.setattr(window.index_revision_probe, "read", locked)
    window._poll_index_updates()
    assert window.result_notice.isVisible()
    assert "暂时无法核对" in window.result_notice_label.text()
    assert not window._results_stale and window.results is results
    monkeypatch.setattr(window.index_revision_probe, "read", original)
    window._poll_index_updates()
    assert not window.result_notice.isVisible()
    assert not window._index_watch_error and window.results is results


def test_external_update_while_searching_does_not_discard_a_newer_valid_snapshot(screen, monkeypatch):
    app, window, db, first, other, _ = screen
    entered, release = Event(), Event()
    original = SearchService.search
    old_results = window.results

    def blocked(service, query):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("test wait timed out")
        return original(service, query)

    monkeypatch.setattr(SearchService, "search", blocked)
    window.run_search()
    try:
        wait_for(app, entered.is_set)
        writer = LocalMindDB(db.db_path)
        try:
            with writer.conn:
                writer.conn.execute("UPDATE files SET size=size+1 WHERE id=?", (other["id"],))
        finally:
            writer.close()
        window._poll_index_updates()
        assert window.result_notice.isVisible() and not window.refresh_results_button.isEnabled()
    finally:
        release.set()
    wait_for(app, lambda: not window.search_busy)
    assert window.results is not old_results
    assert window.selected_result().file_id == first["id"]
    assert not window.result_notice.isVisible()
    assert window._results_db_revision == db.index_revisions()[0]
    assert "本次结果未采用" not in window.search_status.text()


def test_empty_results_can_be_refreshed_after_external_insert_and_clear_resets_watcher(screen, monkeypatch):
    app, window, db, _, _, _ = screen
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("files"))
    window.search_input.setText("OUTSIDE-7915")
    window.run_search()
    wait_for(app, lambda: not window.search_busy)
    assert not window.results
    service = window.search_worker.service
    writer = LocalMindDB(db.db_path)
    try:
        target = index_file(writer, db.db_path.parent / "external.txt", "OUTSIDE-7915")
    finally:
        writer.close()
    window._poll_index_updates()
    assert window.result_notice.isVisible() and not window.results
    window.refresh_results_button.click()
    wait_for(app, lambda: not window.search_busy)
    assert window.selected_result().file_id == target["id"]
    assert window.search_worker.service is service
    assert not window.result_notice.isVisible()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.clear_index()
    window._poll_index_updates()
    assert window._results_db_revision is None and not window.result_notice.isVisible()


def test_notification_layout_and_connection_shutdown(screen, monkeypatch):
    app, window, db, _, other, _ = screen
    with db.conn:
        db.conn.execute("UPDATE files SET size=size+1 WHERE id=?", (other["id"],))
    window._poll_index_updates()
    assert window.index_revision_probe._conn is not None
    assert window.result_notice.isVisible()
    for width, height in ((980, 650), (1240, 800)):
        window.resize(width, height)
        app.processEvents()
        for widget in (window.result_notice_label, window.refresh_results_button, window.preview, window.file_actions[-1]):
            corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
            assert corner.x() <= width and corner.y() <= height
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(Path(output) / f"external-notice-{width}.png"))
    window.search_busy = True
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    assert not window.close()
    assert window.index_watch_timer.isActive() and not window._closing
    window.search_busy = False
    assert window.close()
    assert window._closing and not window.index_watch_timer.isActive()
    assert window.index_revision_probe._conn is None
    window._poll_index_updates()
