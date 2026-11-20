from dataclasses import replace
from datetime import UTC, datetime, timedelta
import os
import sqlite3

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from localmind.chat_timeline import TimelineError, discussions, verify_message, window
from localmind.chats import Attachment, ChatMessage, ChatStore, chat_source_reference, init_chat_schema
from localmind.db import LocalMindDB
from localmind.models import SearchQuery, SearchResult
from localmind.searcher import SearchDiagnostics
from localmind.desktop.chat_timeline import ChatTimeline
from test_evidence_sources import evidence_corpus, evidence_screen
from test_result_lifecycle import wait_for
from test_weflow_alpha import alpha_message
from localmind.weflow_protocol import decode_messages


@pytest.fixture
def corpus(tmp_path):
    db = LocalMindDB(tmp_path / "timeline.db")
    yield db
    db.close()


def add(db, messages, key="source", account="test"):
    return ChatStore(db).ingest(platform="wechat", account=account, source_key=key,
                               conversation="测试讨论", messages=messages)


def message(i, *, reply=None, body=None, **kwargs):
    return ChatMessage(str(i), body or f"合成消息 {i}", sender="联系人",
        sent_at=(datetime(2026, 10, 20, tzinfo=UTC) + timedelta(minutes=i)).isoformat(), reply_to=reply, **kwargs)


def result(db, external_id, key="source", query=None):
    row = db.conn.execute("""SELECT m.*,m.id AS message_id,s.platform,s.account,s.source_key,s.conversation
        FROM chat_messages m JOIN chat_sources s ON s.id=m.source_id WHERE external_id=? AND source_key=?""",
        (str(external_id), key)).fetchone()
    ref = chat_source_reference(dict(row))
    return SearchResult(file_id=0, chunk_id=None, name=row["body"], path="", ext="", modified_at=row["sent_at"] or "",
        score=.8, keyword_score=.8, semantic_score=0, fused_score=.8, reranker_score=None,
        snippet=row["body"], source_type="chat", result_kind="message", chat_message_id=row["id"],
        chat_source=ref, source_query=query or SearchQuery("合成", scope="messages"))


def test_strong_reply_groups_ignore_interleaving_and_deduplicate_identity(corpus):
    add(corpus, [message(1), message(2), message(3, reply="1"), message(4, reply="3")])
    hits = [result(corpus, i) for i in (1, 2, 3, 4)]
    groups, more = discussions(corpus.db_path, hits[0], hits + [replace(hits[0], chunk_id=99)])
    assert not more and len(groups) == 2
    assert len(groups[0].refs) == 3 and len(groups[0].hit_ids) == 3
    page = window(corpus.db_path, hits[0], groups[0])
    assert [r["external_id"] for r in page.rows] == ["1", "3", "4"]
    assert page.rows[1]["gap_before"]
    nearby = window(corpus.db_path, hits[0], groups[0], nearby=True)
    assert [r["external_id"] for r in nearby.rows] == ["1", "2", "3", "4"]
    assert not nearby.rows[1]["hit"] and all(not r["gap_before"] for r in nearby.rows)


def test_same_text_and_ids_in_different_accounts_remain_separate(corpus):
    for key, account in (("a", "a"), ("b", "b"), ("c", "a")):
        add(corpus, [message(1, body="完全相同"), message(2, reply="1")], key, account)
    hits = [result(corpus, 2, key) for key in ("a", "b", "c")]
    groups, _ = discussions(corpus.db_path, hits[0], hits)
    assert len(groups) == 3
    for group in groups:
        rows = window(corpus.db_path, hits[0], group).rows
        assert len(rows) == 2 and all(r["account"] == group.account for r in rows)
    bad = replace(groups[0], refs=groups[0].refs + groups[1].refs)
    with pytest.raises(TimelineError):
        window(corpus.db_path, hits[0], bad)


def test_missing_quotes_and_cycles_are_not_evidence_for_merging(corpus):
    add(corpus, [message(1, reply="missing"), message(2, reply="missing"),
                 message(3, reply="4"), message(4, reply="3")])
    hits = [result(corpus, i) for i in range(1, 5)]
    groups, _ = discussions(corpus.db_path, hits[0], hits)
    assert len(groups) == 4


def test_nearby_uses_keyset_pages_without_overlap_even_for_identical_times(corpus):
    messages = [message(i) for i in range(161)]
    for msg in messages:
        msg.sent_at = messages[0].sent_at
    add(corpus, messages)
    hit = result(corpus, 80)
    group = discussions(corpus.db_path, hit)[0][0]
    page = window(corpus.db_path, hit, group, nearby=True)
    assert [r["external_id"] for r in page.rows] == [str(i) for i in range(60, 101)]
    assert page.previous and page.following
    before = window(corpus.db_path, hit, group, nearby=True, cursor=page.before_cursor, page=-1)
    after = window(corpus.db_path, hit, group, nearby=True, cursor=page.after_cursor, page=1)
    assert [r["external_id"] for r in before.rows] == [str(i) for i in range(20, 60)]
    assert [r["external_id"] for r in after.rows] == [str(i) for i in range(101, 141)]
    final = window(corpus.db_path, hit, group, nearby=True, cursor=after.after_cursor, page=2)
    assert len(final.rows) == 20 and not final.following
    with pytest.raises(TimelineError):
        window(corpus.db_path, hit, group, nearby=True, page=500)


def test_sender_filter_constrains_hits_but_context_retains_other_speakers(corpus):
    add(corpus, [message(0), message(1), replace(message(2, reply="1"), sender="另一人"), message(3)])
    query = SearchQuery("合成", scope="messages", sender="联系人",
                        modified_after="2026-10-20T00:01:00Z", modified_before="2026-10-20T00:03:00Z")
    hit = result(corpus, 1, query=query)
    group = discussions(corpus.db_path, hit)[0][0]
    page = window(corpus.db_path, hit, group, nearby=True)
    assert [r["external_id"] for r in page.rows] == ["1", "2"]
    assert page.rows[1]["sender"] == "另一人" and not page.rows[1]["hit"]


def test_unknown_time_is_not_replaced_by_observed_time(corpus):
    add(corpus, [ChatMessage("x", "无时间", sender="我"), message(1)])
    hit = result(corpus, "x")
    group = discussions(corpus.db_path, hit)[0][0]
    page = window(corpus.db_path, hit, group, nearby=True)
    assert len(page.rows) == 1 and page.rows[0]["from_me"] is None
    assert "时间未知" in page.notice and page.rows[0]["sent_epoch"] is None


@pytest.mark.parametrize("column,value", [("external_id", "replacement"), ("reply_to", "2")])
def test_stale_message_or_relationship_is_rejected(corpus, column, value):
    add(corpus, [message(1), message(2)])
    hit = result(corpus, 1)
    group = discussions(corpus.db_path, hit)[0][0]
    with corpus.conn:
        corpus.conn.execute(f"UPDATE chat_messages SET {column}=? WHERE id=?", (value, hit.chat_message_id))
    with pytest.raises(TimelineError):
        window(corpus.db_path, hit, group)


@pytest.mark.parametrize("column", ["account", "source_key", "platform"])
def test_source_identity_change_fails_closed(corpus, column):
    add(corpus, [message(1)])
    hit = result(corpus, 1)
    group = discussions(corpus.db_path, hit)[0][0]
    with corpus.conn:
        corpus.conn.execute(f"UPDATE chat_sources SET {column}='new'")
    with pytest.raises(TimelineError):
        window(corpus.db_path, hit, group)


def test_expansion_is_bounded_and_reports_truncation(corpus):
    add(corpus, [message(0)] + [message(i, reply="0") for i in range(1, 321)])
    hit = result(corpus, 0)
    group = discussions(corpus.db_path, hit)[0][0]
    first = window(corpus.db_path, hit, group)
    assert len(first.rows) == 40 and first.limited and first.following
    last = window(corpus.db_path, hit, group, page=7)
    assert len(last.rows) == 20 and not last.following
    assert "上限" in first.notice


def test_display_metadata_updates_preserve_existing_chunk_vectors(corpus):
    add(corpus, [message(1)])
    original = list(corpus.conn.execute("SELECT chunk_id,fingerprint FROM chat_units"))
    add(corpus, [message(1, sender_id="wxid", from_me=True, message_type="text", display_body="合成消息 1")])
    row = corpus.conn.execute("SELECT * FROM chat_messages").fetchone()
    assert row["from_me"] == 1 and row["sender_id"] == "wxid" and row["display_body"] == "合成消息 1"
    assert list(corpus.conn.execute("SELECT chunk_id,fingerprint FROM chat_units")) == original
    with corpus.conn:
        corpus.conn.execute("UPDATE chat_messages SET sent_at='2026-10-21T00:00:00Z'")
    assert corpus.conn.execute("SELECT sent_epoch FROM chat_messages").fetchone()[0] == datetime(2026, 10, 21, tzinfo=UTC).timestamp()


@pytest.mark.parametrize("fields,direction", [({}, None), ({"from_me": True}, True), ({"from_me": False}, False),
    ({"from_me": "true"}, None), ({"isSend": 1}, True), ({"isSend": 0}, False), ({"isSend": "1"}, None)])
def test_protocol_retains_reliable_direction_and_separate_quote(fields, direction):
    msg = decode_messages([alpha_message(1, **fields, quote={"content": "之前的消息", "sender": "引用作者"})])[0]
    assert msg.from_me is direction and msg.sender_id == "test-member"
    assert msg.display_body == "测试文档说明1" and "引用快照" in msg.body
    assert msg.quote_body == "之前的消息" and msg.quote_sender == "引用作者"


def test_read_only_view_never_changes_revision_and_cancels(corpus):
    add(corpus, [message(1)])
    hit = result(corpus, 1)
    revision = corpus.index_revisions()
    group = discussions(corpus.db_path, hit)[0][0]
    window(corpus.db_path, hit, group)
    assert corpus.index_revisions() == revision
    with pytest.raises(TimelineError):
        discussions(corpus.db_path, hit, cancelled=lambda: True)
    with corpus.conn:
        corpus.conn.execute("UPDATE chat_messages SET body='updated'")
    with pytest.raises(TimelineError):
        verify_message(corpus.db_path, group, hit.chat_message_id, expected_body=hit.snippet)


def test_file_sources_have_separate_discussions_and_rebinding_is_rejected(evidence_corpus):
    db, _, results, _ = evidence_corpus
    hit = results[0]
    groups, _ = discussions(db.db_path, hit)
    assert len(groups) == 3
    assert {g.key[2] for g in groups} == {"first", "second", "newer"}
    selected = groups[1]
    assert window(db.db_path, hit, selected).rows[0]["attachments"][0]["file_id"] == hit.file_id
    with db.conn:
        db.conn.execute("UPDATE chat_attachments SET file_id=NULL WHERE id=?", (selected.refs[0].attachment_id,))
    with pytest.raises(TimelineError):
        window(db.db_path, hit, selected)


def test_timeline_widget_is_plain_text_cancels_stale_responses_and_has_full_copy(corpus):
    add(corpus, [message(1, body='<img src="https://invalid.test/x">' + "正文" * 1000, from_me=True), message(2)])
    app = QApplication.instance() or QApplication([])
    timeline = ChatTimeline(corpus.db_path)
    try:
        first, second = result(corpus, 1), result(corpus, 2)
        timeline.set_result(first)
        stale = timeline.generation
        timeline.set_result(second)
        timeline._finished(stale, None, "不应出现")
        wait_for(app, lambda: timeline.payload is not None)
        assert timeline.model.rows[0]["external_id"] == "2"
        timeline.set_result(first)
        wait_for(app, lambda: timeline.payload is not None)
        doc, _ = timeline.view.itemDelegate().layout(timeline.model.rows[0], 400)
        assert '<img src=' in doc.toPlainText() and "正文" in doc.toPlainText()
        assert len(doc.toPlainText()) < 500
        timeline.copy_message()
        assert first.snippet in QApplication.clipboard().text()
        timeline.clear()
        assert not timeline.model.rows and not timeline.busy and not timeline.selector.count()
    finally:
        assert timeline.shutdown()
        timeline.deleteLater()
        app.processEvents()


def test_timeline_layout_and_file_target_stay_consistent(evidence_screen, tmp_path):
    app, main, db, hit, path = evidence_screen
    wait_for(app, lambda: main.chat_timeline.payload is not None)
    assert main.chat_timeline.selector.count() == 3
    assert not main.file_preview.isVisible()
    main.chat_timeline.selector.setCurrentIndex(1)
    wait_for(app, lambda: main.chat_timeline.payload is not None)
    assert main.selected_result() is hit and main.file_preview._target[0] == str(path)
    for width, height in ((980, 650), (1240, 800)):
        main.resize(width, height)
        app.processEvents()
        for widget in (main.chat_timeline, main.chat_timeline.view, main.chat_timeline.selector, main.file_actions[0]):
            corner = widget.mapTo(main, QPoint(widget.width(), widget.height()))
            assert corner.x() <= main.width() and corner.y() <= main.height()
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            from pathlib import Path
            Path(output).mkdir(parents=True, exist_ok=True)
            assert main.grab().save(str(Path(output) / f"chat-timeline-{width}.png"))
        assert main.chat_timeline.view.height() >= 80
        assert "历史可能未完整同步" in main.chat_timeline.mode.toolTip()
        assert main.height() == height and main.width() == width
        corner = main.chat_timeline.view.mapTo(main.chat_timeline, main.chat_timeline.view.rect().bottomRight())
        assert corner.y() < main.chat_timeline.height()


@pytest.mark.parametrize("field,value", [("body", "new content"), ("sender", "new sender"), ("from_me", 1),
    ("sent_at", "2026-10-22T00:00:00Z"), ("quote_sender", "new author"), ("display_body", "new display")])
def test_bubble_file_click_emits_exact_attachment_and_rejects_stale_body(evidence_corpus, field, value):
    db, _, results, _ = evidence_corpus
    app = QApplication.instance() or QApplication([])
    timeline = ChatTimeline(db.db_path)
    timeline.resize(640, 700)
    opened = []
    timeline.attachmentRequested.connect(lambda msg, att: opened.append((msg["id"], att["id"])))
    try:
        timeline.show()
        timeline.set_result(results[0])
        wait_for(app, lambda: timeline.payload is not None)
        index = timeline.model.index(0)
        row = timeline.model.rows[0]
        rect = timeline.view.itemDelegate().file_rects(timeline.view.visualRect(index), row)[0]
        QTest.mouseClick(timeline.view.viewport(), Qt.MouseButton.LeftButton, pos=rect.center().toPoint())
        assert opened == [(row["id"], row["attachments"][0]["id"])]
        with db.conn:
            db.conn.execute(f"UPDATE chat_messages SET {field}=? WHERE id=?", (value, row["id"]))
        timeline._attachment(row, row["attachments"][0])
        assert len(opened) == 1
    finally:
        assert timeline.shutdown()
        timeline.close()
        timeline.deleteLater()
        app.processEvents()


def test_grouped_and_flat_views_preserve_all_original_result_objects(evidence_screen):
    app, main, db, _, _ = evidence_screen
    add(db, [message(1), message(2, reply="1"), message(3)])
    hits = [result(db, i) for i in (1, 2, 3)]
    original = [(id(r), r.score, r.chat_source.identity) for r in hits]
    main._search_finished(hits, SearchDiagnostics([], 3, 0, 3, None, None, "", "", scope="messages"),
                         {"total_ms": 1, "index_refresh_ms": 0})
    wait_for(app, lambda: main.chat_timeline.payload is not None)
    assert main.discussion_mode.isVisible()
    assert sum(not main.result_table.isRowHidden(i) for i in range(3)) == 2
    main._update_result_controls()
    assert "2 段讨论" in main.result_count_label.text()
    main.discussion_mode.setCurrentIndex(1)
    assert all(not main.result_table.isRowHidden(i) for i in range(3))
    main._update_result_controls()
    assert "3 条结果" in main.result_count_label.text()
    assert [(id(r), r.score, r.chat_source.identity) for r in main.results] == original


def test_host_attachment_action_rechecks_path_even_when_file_id_is_reused(evidence_screen, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    app, main, db, _, _ = evidence_screen
    wait_for(app, lambda: main.chat_timeline.payload is not None)
    row = main.chat_timeline.model.rows[0]
    att = row["attachments"][0]
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[1]))
    with db.conn:
        db.conn.execute("UPDATE files SET path='another-file.txt' WHERE id=?", (att["file_id"],))
    main._open_timeline_attachment(row, att)
    assert notices == ["附件已变化"]


def test_revision_refresh_keeps_reader_position_and_multispeaker_layout(corpus):
    add(corpus, [message(0, body="准备发布本周的质量评审资料。", from_me=False)] + [
        message(i, reply="0", body="先核对验收范围和材料编号，再确认负责人。\n" * 3,
                from_me=i % 2 == 0, quote_body="准备发布本周的质量评审资料。" if i == 2 else "",
                quote_sender="项目负责人", display_body="材料核对完成，明天安排评审。" if i == 2 else None)
        for i in range(1, 12)])
    app = QApplication.instance() or QApplication([])
    timeline = ChatTimeline(corpus.db_path)
    timeline.resize(720, 680)
    hit = result(corpus, 0)
    try:
        timeline.show()
        timeline.set_result(hit, revision=1)
        wait_for(app, lambda: timeline.payload is not None)
        timeline.view.setCurrentIndex(timeline.model.index(4))
        timeline.view.scrollTo(timeline.model.index(4))
        app.processEvents()
        scroll = timeline.view.verticalScrollBar().value()
        selected = timeline.view.currentIndex().data(Qt.ItemDataRole.UserRole)["id"]
        timeline.set_result(hit, revision=2)
        wait_for(app, lambda: timeline.payload is not None)
        assert timeline.view.currentIndex().data(Qt.ItemDataRole.UserRole)["id"] == selected
        assert timeline.view.verticalScrollBar().value() == scroll
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            from pathlib import Path
            Path(output).mkdir(parents=True, exist_ok=True)
            timeline.view.scrollTo(timeline.model.index(2))
            app.processEvents()
            assert timeline.grab().save(str(Path(output) / "chat-bubbles.png"))
    finally:
        assert timeline.shutdown()
        timeline.close()
        timeline.deleteLater()
        app.processEvents()
