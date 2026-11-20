from dataclasses import replace
import os
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from localmind.chat_timeline import discussions, window
from localmind.chats import ChatMessage, ChatStore, ForwardItem, forward_items
from localmind.desktop.chat_timeline import ChatTimeline
from localmind.desktop.chat_transcript import AlphaContextDialog, TranscriptDialog, forward_rows
from localmind.weflow_protocol import decode_messages
from test_chat_timeline import add, corpus, result
from test_weflow_forward import item, record, xml
from test_weflow_alpha import alpha_message
from test_alpha_chat_context import ANCHOR, Client
from test_result_lifecycle import wait_for as wait_with_app


def wait_for(predicate):
    wait_with_app(QApplication.instance(), predicate)


def archive(db):
    nested = item(17, recordxml=ET.tostring(record([item(1, "原始聊天子消息", name="内层作者"), item(8, "内层附件.pdf")]), encoding="unicode"))
    msg = decode_messages([alpha_message(1, text=xml([nested, item(8, "外层附件.pdf")]))])[0]
    add(db, [msg])
    hit = result(db, msg.external_id)
    group = discussions(db.db_path, hit)[0][0]
    return hit, group, window(db.db_path, hit, group).rows[0]


def test_forward_json_roundtrip_nested_identity_and_attachment_ordinal(corpus):
    hit, group, outer = archive(corpus)
    assert [f.path for f in outer["forward_items"]] == [(1,), (1, 1), (1, 2), (2,)]
    rows = forward_rows(outer)
    assert rows[1]["sender"] == "内层作者" and rows[1]["from_me"] is None
    assert rows[2]["attachments"][0]["name"] == "内层附件.pdf"
    assert rows[3]["attachments"][0]["id"] != rows[2]["attachments"][0]["id"]
    assert all(r["sent_at"] is None for r in rows)
    assert forward_items(outer["forward_json"]) == outer["forward_items"]


def test_forward_ui_nested_back_copy_and_exact_file_action(corpus):
    _, _, outer = archive(corpus)
    actions = []
    dialog = TranscriptDialog("合成转发", forward_rows(outer), attachment=lambda *args: actions.append(args))
    dialog.show()
    try:
        assert len(dialog.model.rows) == 2
        dialog.inspect()
        assert dialog.branch == (1,) and len(dialog.model.rows) == 2
        dialog.copy_message()
        assert QApplication.clipboard().text() == "内层作者\n原始聊天子消息"
        row = dialog.model.rows[1]
        dialog.open_attachment(row, row["attachments"][0])
        assert actions[0][1]["name"] == "内层附件.pdf"
        dialog.go_back()
        assert dialog.branch == ()
        dialog.guard = lambda: False
        dialog.copy_message()
        assert not dialog.model.rows
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.mark.parametrize("width,height", [(500, 500), (760, 680)])
def test_transcript_responsive_bubbles_and_snapshot(corpus, width, height):
    from localmind.desktop.theme import APP_STYLESHEET
    _, _, outer = archive(corpus)
    rows = list(forward_rows(outer))
    rows[0]["body"] += "\n" + "很长的中文聊天内容与文件名称" * 20
    dialog = TranscriptDialog("合成转发记录", rows, notice="转发原记录 · 只读")
    dialog.setStyleSheet(APP_STYLESHEET)
    dialog.resize(width, height)
    dialog.show()
    app = QApplication.instance()
    app.processEvents()
    try:
        dialog.view.doItemsLayout()
        assert dialog.width() == width and dialog.view.width() < width
        assert dialog.view.height() > 200
        assert dialog.view.horizontalScrollBar().maximum() == 0
        output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            assert dialog.grab().save(str(Path(output) / f"chat-transcript-{width}.png"))
    finally:
        dialog.close()
        dialog.deleteLater()


def test_partial_nested_xml_keeps_unique_paths_and_valid_records():
    invalid = item(17, recordxml="<recordinfo><datalist count='2'/></recordinfo>")
    issues = []
    msg = decode_messages([alpha_message(1, text=xml([invalid, item(8, "保留.pdf")]))], issues=issues)[0]
    msg.validate()
    assert [r.path for r in msg.forward_items] == [(1,), (2,)]
    assert msg.forward_items[0].kind == "unknown" and issues


def test_invalid_forward_parent_is_rejected():
    msg = ChatMessage("x", "body", forwarded=True, forward_items=(ForwardItem((1, 1), body="orphan"),))
    with pytest.raises(ValueError, match="身份"):
        msg.validate()


@pytest.mark.parametrize("value", ['{}', '[null]', '[{"path":null}]', 'not-json'])
def test_malformed_forward_archive_is_controlled(value):
    with pytest.raises(ValueError):
        forward_items(value)


def test_remote_dialog_async_completion_and_credential_release():
    client = Client()
    dialog = AlphaContextDialog(client, ANCHOR, options={"kind": "around"}, guard=lambda: True)
    dialog.show()
    try:
        wait_for(lambda: not dialog.thread.isRunning())
        assert len(dialog.model.rows) == 3 and client._token == ""
        assert all(not row["attachments"] for row in dialog.model.rows)
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_remote_dialog_cancellation_never_displays_late_rows():
    client = Client()
    original = client._request_json

    def slow(*args, **kwargs):
        time.sleep(.04)
        return original(*args, **kwargs)

    client._request_json = slow
    dialog = AlphaContextDialog(client, ANCHOR, options={"kind": "around"}, guard=lambda: True)
    dialog.show()
    dialog.reject()
    wait_for(lambda: not dialog.thread.isRunning())
    wait_for(lambda: not dialog.isVisible())
    assert not dialog.model.rows and client._token == ""
    dialog.deleteLater()


def test_remote_dialog_stale_anchor_discards_read_result():
    client = Client()
    dialog = AlphaContextDialog(client, ANCHOR, options={"kind": "around"}, guard=lambda: False)
    dialog.show()
    try:
        wait_for(lambda: not dialog.thread.isRunning())
        assert not dialog.model.rows and "来源已更新" in dialog.notice.text()
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_topic_ui_ignores_late_response_and_supports_split(corpus):
    from test_chat_topics import data, Encoder
    from localmind.chat_topics import TopicOrganizer
    hits, groups = data(corpus, ["合同 AB123 的验收付款节点需要财务审核"] * 2)
    timeline = ChatTimeline(corpus.db_path)
    requests = []
    timeline.topicsRequested.connect(requests.append)
    try:
        timeline.set_result(hits[0], hits)
        wait_for(lambda: timeline.payload is not None)
        timeline.organize_topics()
        request = requests[-1]
        merged, notice = TopicOrganizer().organize(corpus.db_path, hits[0], groups, Encoder())
        timeline.topics_finished(request, merged, notice)
        wait_for(lambda: timeline.payload is not None)
        assert len(timeline.groups) == 1
        timeline.split_topic()
        wait_for(lambda: timeline.payload is not None)
        assert len(timeline.groups) == 2 and len(timeline.topic_excluded) == 2
        timeline.clear()
        timeline.topics_finished(request, merged, notice)
        assert not timeline.groups and not timeline.model.rows
    finally:
        assert timeline.shutdown()
        timeline.deleteLater()
