import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.models import SearchQuery
from localmind.source_history_ui import SourceHistoryDialog, _display_time
from localmind import searcher as search_module
from test_weflow_v04 import db, index_file, ingest, searcher


def sources(db, file, count=60, *, source="archive", conversation="项目归档群"):
    return ingest(db, [
        ChatMessage(str(i), f"交付记录 {i} " + ("很长的来源。" * 1000 if i == 0 else ""),
                    "陈工" if i % 2 else "李工",
                    "2026-09-12T04:00:00Z" if i < count - 1 else None,
                    [Attachment(file["name"], path=file["path"])])
        for i in range(count)
    ], source=source, conversation=conversation)


def test_file_history_pages_are_complete_stable_readonly_and_metadata_only(db, tmp_path):
    file = index_file(db, tmp_path / "archive.txt")
    other = index_file(db, tmp_path / "other" / "archive.txt")
    sources(db, file)
    sources(db, other, 2, source="other")
    store = ChatStore(db)
    changes = db.conn.total_changes
    db.conn.execute("PRAGMA query_only=ON")
    cursor, seen, sizes = None, [], []
    while True:
        rows, cursor = store.file_sources(file["id"], cursor=cursor, limit=25)
        sizes.append(len(rows))
        seen.extend(row["attachment_id"] for row in rows)
        assert all("body" not in row and "attachment_context" not in row for row in rows)
        if cursor is None:
            assert rows[-1]["sent_at"] is None
            break
    assert sizes == [25, 25, 10]
    expected = {r[0] for r in db.conn.execute("SELECT id FROM chat_attachments WHERE file_id=?", (file["id"],))}
    assert set(seen) == expected and len(seen) == len(expected)
    assert len(store.attachment_context(seen[-2], file_id=file["id"])) < 5000
    assert db.conn.total_changes == changes


def test_source_cursor_does_not_repeat_when_a_newer_record_arrives(db, tmp_path):
    file = index_file(db, tmp_path / "cursor.txt")
    sources(db, file, 7)
    store = ChatStore(db)
    first, cursor = store.file_sources(file["id"], limit=3)
    ingest(db, [ChatMessage("new", "新来源", sent_at="2026-10-20T00:00:00Z",
                           attachments=[Attachment(file["name"], path=file["path"])])], source="new")
    second, _ = store.file_sources(file["id"], cursor=cursor, limit=100)
    assert len(first) == 3 and len(second) == 4
    assert not {r["attachment_id"] for r in first} & {r["attachment_id"] for r in second}
    assert all(r["sent_at"] != "2026-10-20T00:00:00Z" for r in second)
    assert store.file_sources(file["id"])[0][0]["sent_at"] == "2026-10-20T00:00:00Z"


def test_history_preserves_effective_query_and_does_not_leak_date_scope(db, tmp_path, monkeypatch):
    file = index_file(db, tmp_path / "time.txt", "正文独有材料")
    sources(db, file, 5)
    ingest(db, [ChatMessage("new", "新交付", "李工", "2026-10-01T00:00:00Z",
                           [Attachment(file["name"], path=file["path"])])], source="october", conversation="项目归档群")
    original = search_module.process_query
    monkeypatch.setattr(search_module, "process_query", lambda text: original(text, now=datetime(2026, 10, 20)))
    query = SearchQuery("上个月正文独有材料", scope="chat_files", evidence="content",
                        conversation="归档", sender="李工")
    result = searcher(db, mode="keyword").search(query)[0]
    query.conversation = "UI later changed"
    assert result.source_query.text == ""
    rows, _ = ChatStore(db).file_sources(file["id"], result.source_query)
    assert len(rows) == 2 and all(row["sender"] == "李工" for row in rows)
    assert all(row["sent_at"].startswith("2026-09") for row in rows)
    outsider = db.conn.execute("SELECT id FROM chat_attachments ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert not ChatStore(db).attachment_context(outsider, file_id=file["id"], query=result.source_query)
    # A local file modification date must not become a message date restriction.
    rows, _ = ChatStore(db).file_sources(file["id"], SearchQuery("", scope="files", modified_after="2099-01-01"))
    assert len(rows) == 6


def test_source_search_is_literal_and_context_rechecks_current_link(db, tmp_path):
    file = index_file(db, tmp_path / "literal.txt")
    sources(db, file, 2, conversation="项目_100%归档")
    sources(db, file, 2, source="plain", conversation="项目归档")
    store = ChatStore(db)
    rows, _ = store.file_sources(file["id"], text="100%")
    assert len(rows) == 2
    assert not store.file_sources(file["id"], text="' OR 1=1 --")[0]
    assert len(store.file_sources(file["id"], text="陈工")[0]) == 2
    attachment = rows[0]["attachment_id"]
    text = store.attachment_context(attachment, file_id=file["id"])
    assert text
    db.conn.execute("UPDATE chat_attachments SET file_id=NULL WHERE id=?", (attachment,))
    db.conn.commit()
    assert not store.attachment_context(attachment, file_id=file["id"])


@pytest.mark.parametrize("limit", [0, -1, 101, True])
def test_source_page_bounds(db, limit):
    with pytest.raises(ValueError):
        ChatStore(db).file_sources(1, limit=limit)


def test_source_time_is_local_and_bad_values_remain_inspectable():
    assert _display_time("2026-10-01T00:00:00+08:00") == _display_time("2026-09-30T16:00:00Z")
    assert _display_time("2026-10-01T12:30:00") == "2026-10-01 12:30"
    assert _display_time(None) == "未知"
    assert _display_time("unrecognized") == "unrecognized"
    assert _display_time("0001-01-01T00:00:00Z")


def test_history_dialog_pages_filters_copy_and_preserves_original_result(db, tmp_path):
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    from localmind.ui import APP_STYLESHEET
    file = index_file(db, tmp_path / "交付材料.txt")
    sources(db, file)
    result = searcher(db, mode="keyword").search(SearchQuery("付款节点", scope="chat_files", evidence="content"))[0]
    original = asdict(result)
    changes = db.conn.total_changes
    dialog = SourceHistoryDialog(ChatStore(db), result)
    dialog.setStyleSheet(APP_STYLESHEET)
    dialog.show()
    try:
        app.processEvents()
        assert len(dialog.rows) == 25 and dialog.next.isEnabled() and not dialog.previous.isEnabled()
        first_ids = [r["attachment_id"] for r in dialog.rows]
        dialog.next.click()
        assert dialog.previous.isEnabled()
        assert set(first_ids).isdisjoint(r["attachment_id"] for r in dialog.rows)
        dialog.previous.click()
        assert [r["attachment_id"] for r in dialog.rows] == first_ids
        dialog.search.setText("陈工")
        assert not dialog.copy_button.isEnabled() and not dialog.next.isEnabled()
        dialog.reset_pages()
        assert all(r["sender"] == "陈工" for r in dialog.rows)
        dialog.table.selectRow(3)
        dialog.copy_button.click()
        assert QApplication.clipboard().text() == dialog.preview.toPlainText()
        assert "陈工" in dialog.preview.toPlainText()
        for width, height in ((640, 480), (900, 660)):
            dialog.resize(width, height)
            app.processEvents()
            for widget in (dialog.table, dialog.preview, dialog.copy_button, dialog.search):
                pos = widget.mapTo(dialog, widget.rect().topLeft())
                assert pos.x() >= 0 and pos.y() >= 0
                assert pos.x() + widget.width() <= dialog.width()
                assert pos.y() + widget.height() <= dialog.height()
            if output := os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR"):
                Path(output).mkdir(parents=True, exist_ok=True)
                dialog.grab().save(str(Path(output) / f"source-history-{width}.png"))
        dialog.search.setText("不存在")
        dialog.reset_pages()
        assert not dialog.rows and not dialog.preview.toPlainText()
        assert not dialog.copy_button.isEnabled()
        assert db.conn.total_changes == changes and asdict(result) == original
        dialog.search.setText("待处理筛选")
        dialog.reject()
        assert not dialog.timer.isActive()
    finally:
        dialog.close()
        app.processEvents()


def test_history_dialog_handles_deleted_links_and_errors_without_stale_copy(db, tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    file = index_file(db, tmp_path / "stale.txt")
    sources(db, file, 1)
    result = searcher(db, mode="keyword").search(SearchQuery("付款节点", scope="chat_files", evidence="content"))[0]
    store = ChatStore(db)
    dialog = SourceHistoryDialog(store, result)
    try:
        QApplication.clipboard().setText("keep unrelated clipboard")
        db.conn.execute("DELETE FROM chat_attachments WHERE id=?", (dialog.rows[0]["attachment_id"],))
        db.conn.commit()
        dialog.copy_button.click()
        assert not dialog.copy_button.isEnabled()
        assert QApplication.clipboard().text() == "keep unrelated clipboard"
        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("private details not displayed")
        monkeypatch.setattr(store, "file_sources", fail)
        dialog.load_page()
        assert dialog.status.text() == "来源读取失败"
        assert not dialog.rows and not dialog.preview.toPlainText()
        assert not dialog.copy_button.isEnabled()
    finally:
        dialog.close()
        app.processEvents()
