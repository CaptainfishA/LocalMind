from __future__ import annotations

import hashlib
import html
import json
import sqlite3
import xml.etree.ElementTree as ET

import pytest

from localmind.chat_limits import MAX_FORWARD_ITEMS, MAX_MESSAGE_CHARS
from localmind.chats import Attachment, ChatMessage, ChatStore, digest, init_chat_schema, parse_export
from localmind.db import LocalMindDB
from localmind.models import SearchQuery
from localmind.weflow import decode_messages, sync_session
from localmind.weflow_protocol import MessageContentError
from test_weflow_alpha import alpha_message
from test_weflow_batch import START, END, batch_client, run
from test_weflow_v04 import CountingEmbedding, index_file, ingest, searcher


def item(kind, text="", *, name="模拟原发送者", time="2026-10-01 09:30", **fields):
    node = ET.Element("dataitem", datatype=str(kind))
    values = {"sourcename": name, "sourcetime": time, "datatitle" if kind == 8 else "datadesc": text, **fields}
    for key, value in values.items():
        ET.SubElement(node, key).text = str(value)
    return node


def record(items):
    root = ET.Element("recordinfo")
    ET.SubElement(root, "title").text = "合成转发样本"
    children = ET.SubElement(root, "datalist", count=str(len(items)))
    children.extend(items)
    return root


def xml(items, encoding="element"):
    root = ET.Element("msg")
    app = ET.SubElement(root, "appmsg")
    ET.SubElement(app, "type").text = "19"
    ET.SubElement(app, "title").text = "合成转发样本"
    container = ET.SubElement(app, "recorditem")
    content = record(items)
    if encoding == "element":
        container.append(content)
    else:
        container.text = ET.tostring(content, encoding="unicode")
    raw = ET.tostring(root, encoding="unicode")
    if encoding == "cdata":
        raw = raw.replace(html.escape(container.text), "<![CDATA[" + container.text + "]]>")
    return html.escape(raw) if encoding == "escaped" else raw


def message(items, **extra):
    return decode_messages([alpha_message(1, text=xml(items), **extra)])[0]


@pytest.fixture
def db(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    yield db
    db.close()


@pytest.mark.parametrize("encoding", ["element", "text", "cdata", "escaped"])
def test_forward_xml_encodings_and_no_source_path_or_keys(encoding):
    raw = xml([item(1, "交付给南京团队的竣工材料"), item(8, "验收&清单.txt", datasize=31,
              fullmd5="A" * 32, datasourcepath="C:/private/original.txt", cdndatakey="never-store-this",
              dataurl="https://example.invalid/download")], encoding)
    msg = decode_messages([alpha_message(1, text="[聊天记录]", rawContent=raw)])[0]
    assert msg.forwarded and len(msg.attachments) == 1
    assert msg.attachments[0].name == "验收&清单.txt"
    att = msg.attachments[0]
    assert att.size == 31 and att.md5 == "a" * 32 and not att.path
    assert "南京团队" in att.context and "第 2 项" in att.origin
    assert "private" not in str(msg) and "never-store" not in str(msg) and "example.invalid" not in str(msg)
    msg.validate()


def test_nested_entries_have_distinct_origins_without_fabricated_messages():
    nested = item(17, recordxml=ET.tostring(record([item(1, "蓝桥投标说明"), item(8, "预算.xlsx")]), encoding="unicode"))
    msg = message([nested, item(8, "会议.docx")])
    assert len(msg.attachments) == 2
    assert "第 1.2 项" in msg.attachments[0].origin and "第 2 项" in msg.attachments[1].origin
    assert "蓝桥投标" in msg.attachments[0].context and "蓝桥投标" not in msg.attachments[1].context
    assert msg.external_id == decode_messages([alpha_message(1)])[0].external_id


def test_different_file_contexts_and_ambiguous_middle_text_are_not_spread():
    msg = message([item(1, "广州采购用途"), item(8, "采购.txt"), item(1, "无法判断属于哪个文件"),
                   item(8, "考勤.txt"), item(1, "苏州人事用途")])
    first, second = msg.attachments
    assert "广州采购" in first.context and "苏州人事" not in first.context
    assert "苏州人事" in second.context and "广州采购" not in second.context
    assert all("无法判断" not in a.context for a in msg.attachments)
    assert "无法判断" in msg.body


def test_inner_unsupported_and_malformed_preserve_files_and_report_partial():
    bad_nested = item(17, recordxml='<!DOCTYPE r [<!ENTITY leak "private-secret">]><r>&leak;</r>')
    raw = xml([item(8, "正常.txt"), item(999, "private-media", secret="do-not-store"), bad_nested, item(8, "保留.txt")])
    issues = []
    msg = decode_messages([alpha_message(1, text=raw)], issues=issues)[0]
    assert [a.name for a in msg.attachments] == ["正常.txt", "保留.txt"]
    assert msg.forwarded and "子项待解析" in msg.body
    assert "private-secret" not in msg.body and "do-not-store" not in msg.body
    assert issues[0]["code"] == "forward_partial" and issues[0]["partial"] is True
    assert "private" not in json.dumps(issues)
    with pytest.raises(MessageContentError, match="部分子项"):
        decode_messages([alpha_message(1, text=raw)])


@pytest.mark.parametrize("kind", ["count", "depth", "body", "missing-file"])
def test_incomplete_or_excessive_forward_is_not_silent_success(kind):
    rows = [item(8, "正常.txt")]
    code = "forward_limit"
    if kind == "count":
        rows *= MAX_FORWARD_ITEMS + 1
    elif kind == "depth":
        for _ in range(4):
            rows = [item(17, recordxml=ET.tostring(record(rows), encoding="unicode"))]
    elif kind == "body":
        rows = [item(1, "中" * (MAX_MESSAGE_CHARS // 2)), *rows]
        code = "body_limit"  # Body plus independent attachment context is bounded.
    else:
        rows = [item(8, "")]
        code = "forward_partial"
    issues = []
    result = decode_messages([alpha_message(1, text=xml(rows))], issues=issues)
    assert issues[0]["code"] == code
    assert bool(result) == (kind == "missing-file")


def test_container_count_mismatch_and_unknown_text_not_promoted_to_file():
    issues = []
    invalid = xml([item(8, "合同.pdf")]).replace('count="1"', 'count="2"')
    assert not decode_messages([alpha_message(1, text=invalid)], issues=issues)
    assert issues[0]["code"] == "forward_unsupported"
    assert not message([item(1, "只是提到了合同.pdf，未发送文件")]).attachments


def test_search_context_content_provenance_and_no_reembedding_after_reopen(db, tmp_path):
    embed = CountingEmbedding()
    first = index_file(db, tmp_path / "交接.txt", "付款节点在质量验收合格之后", embed=embed)
    second = index_file(db, tmp_path / "安排.txt", "年假天数与出勤登记", embed=embed)
    rows = [item(1, "发给南宁评审组的竣工报告"),
            item(8, "交接.txt", fullmd5=hashlib.md5((tmp_path / "交接.txt").read_bytes()).hexdigest(), datasize=first["size"]),
            item(8, "安排.txt", fullmd5=hashlib.md5((tmp_path / "安排.txt").read_bytes()).hexdigest(), datasize=second["size"]),
            item(1, "下个月部门考勤登记")]
    msg = message(rows)
    ingest(db, [msg], embedding=embed)
    assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 1
    assert len(ChatStore(db).attachments()) == 2
    service = searcher(db, embed, "keyword")
    for query in ("南宁评审组", "付款节点"):
        found = service.search(SearchQuery(query, scope="chat_files"))
        assert found and found[0].path == first["path"]
        assert all(result.path == first["path"] for result in found)
        assert "合并转发内第 2 项" in found[0].chat_context and "模拟原发送者" in found[0].chat_context
    assert service.search(SearchQuery("考勤登记", scope="chat_files"))[0].path == second["path"]
    assert service.search(SearchQuery("南宁评审组", scope="messages"))[0].chat_message_id
    count = embed.encoded
    reopened = LocalMindDB(db.db_path)
    try:
        assert ingest(reopened, [msg], embedding=embed)["updated_units"] == 0
        assert embed.encoded == count
        assert not reopened.conn.execute("PRAGMA foreign_key_check").fetchall()
        changed = message([item(8, "安排.txt")])
        ingest(reopened, [changed], embedding=embed)
        assert len(ChatStore(reopened).attachments()) == 1
        assert not searcher(reopened, embed, "keyword").search(SearchQuery("南宁评审组", scope="chat_files"))
        assert not reopened.conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        reopened.close()


def test_partial_sync_retry_keeps_supported_attachment_and_recovers(db):
    client = batch_client(1)
    client.opener.rows["chat-0"] = [alpha_message(1, conversation_id="chat-0", text=xml([item(8, "已读.txt"), item(17)]))]
    embed = CountingEmbedding()
    first = run(db, client, embedding_service=embed)
    assert first["state"] == "partial" and first["failed_messages"] == 1
    assert len(ChatStore(db).attachments()) == 1
    assert db.conn.execute("SELECT code FROM chat_sync_issues").fetchone()[0] == "forward_partial"
    second = run(db, client, embedding_service=embed)
    assert second["state"] == "partial" and second["updated_units"] == 0
    client.opener.rows["chat-0"][0]["text"] = xml([item(8, "已读.txt"), item(8, "修复.txt")])
    fixed = run(db, client, embedding_service=embed)
    assert fixed["state"] == "completed" and fixed["failed_messages"] == 0
    assert len(ChatStore(db).attachments()) == 2
    assert not db.conn.execute("SELECT 1 FROM chat_sync_issues").fetchone()
    assert not db.conn.execute("PRAGMA foreign_key_check").fetchall()


def test_outer_neighbors_and_replies_do_not_mix_inner_file_contexts(db):
    msg = message([item(8, "内部.txt"), item(1, "内部采购用途")])
    other = ChatMessage("outer", "外层独立描述", sent_at=msg.sent_at, attachments=[Attachment("外层.txt")])
    reply = ChatMessage("reply", "外部回复内容" * 100, reply_to=msg.external_id)
    ingest(db, [msg, other, reply])
    query = "SELECT a.name,c.chunk_text FROM chat_units u JOIN chat_attachments a ON a.id=u.attachment_id JOIN chunks c ON c.id=u.chunk_id"
    for row in db.conn.execute(query):
        if row["name"] == "内部.txt":
            assert "外层独立描述" not in row["chunk_text"] and "外部回复内容" not in row["chunk_text"]
        else:
            assert "内部采购用途" not in row["chunk_text"]


def test_old_schema_migration_preserves_rows_and_can_repeat():
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript("""CREATE TABLE files(id INTEGER PRIMARY KEY,name TEXT,file_hash TEXT);
        CREATE TABLE chat_messages(id INTEGER PRIMARY KEY,source_id INTEGER,external_id TEXT,sender TEXT,sent_at TEXT,body TEXT,observed_at TEXT,reply_to TEXT);
        CREATE TABLE chat_attachments(id INTEGER PRIMARY KEY,message_id INTEGER,ordinal INTEGER,name TEXT,path_hint TEXT,sha256 TEXT,size INTEGER,kind TEXT,file_id INTEGER,link_state TEXT,linked_hash TEXT,md5 TEXT);
        INSERT INTO chat_messages(id,body) VALUES(1,'old-message');
        INSERT INTO chat_attachments(id,name) VALUES(1,'old-file');""")
        init_chat_schema(conn)
        init_chat_schema(conn)
        assert conn.execute("SELECT body,forwarded FROM chat_messages").fetchone() == ("old-message", 0)
        assert conn.execute("SELECT name,context,origin FROM chat_attachments").fetchone() == ("old-file", "", "")
    finally:
        conn.close()


def test_generated_import_identity_does_not_change_with_new_columns(tmp_path):
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"messages": [{"text": "说明.txt"}]}), encoding="utf-8")
    previous_fields = {"name": "说明.txt", "path": "", "sha256": "", "size": None, "kind": "mention", "md5": ""}
    expected = digest(json.dumps(["", None, "说明.txt", [previous_fields]], ensure_ascii=False, sort_keys=True)) + ":1"
    assert parse_export(path)[1][0].external_id == expected


def test_forward_in_quote_is_not_promoted_to_outer_message():
    root = ET.fromstring(xml([item(8, "原引用文件.txt")]))
    outer = ET.Element("msg")
    app = ET.SubElement(outer, "appmsg")
    ET.SubElement(app, "type").text = "57"
    ET.SubElement(app, "title").text = "关于之前的聊天记录"
    reference = ET.SubElement(app, "refermsg")
    ET.SubElement(reference, "svrid").text = "previous-forward"
    reference.append(root)
    msg = decode_messages([alpha_message(1, text=ET.tostring(outer, encoding="unicode"))])[0]
    assert not msg.forwarded and not msg.attachments
    assert msg.body == "关于之前的聊天记录" and msg.reply_to == "previous-forward"


def test_long_forward_tail_is_kept_and_cancel_does_not_write(db):
    body = "项目背景说明。" * 1800 + "桂林并网评审专用"
    msg = message([item(1, body), item(8, "长记录.txt")])
    embed = CountingEmbedding()
    with pytest.raises(RuntimeError, match="取消"):
        ChatStore(db).ingest(platform="wechat", account="mock", conversation="mock", source_key="cancel",
                             messages=[msg], embedding_service=embed, should_cancel=lambda: True)
    assert not ChatStore(db).sources()
    ingest(db, [msg], embedding=embed)
    results = searcher(db, embed, "keyword").search(SearchQuery("桂林并网评审", scope="chat_files"))
    assert results and results[0].attachment_id
    assert "桂林并网评审" in results[0].snippet


def test_maximum_forward_files_persist_as_one_message_without_duplicates(db):
    msg = message([item(8, f"材料-{number}.txt") for number in range(MAX_FORWARD_ITEMS)])
    result = ingest(db, [msg])
    assert result["updated_units"] > MAX_FORWARD_ITEMS
    assert len(ChatStore(db).attachments()) == MAX_FORWARD_ITEMS
    assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 1
    assert ingest(db, [msg])["updated_units"] == 0
    assert not db.conn.execute("PRAGMA foreign_key_check").fetchall()


def test_oversized_inner_field_does_not_discard_other_forwarded_files():
    issues = []
    messages = decode_messages([alpha_message(1, text=xml([item(8, "长" * 241), item(8, "正常.txt")]))], issues=issues)
    assert [a.name for a in messages[0].attachments] == ["正常.txt"]
    assert issues[0]["code"] == "forward_partial"
