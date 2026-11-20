import copy
import json
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QFontDatabase

from localmind.db import LocalMindDB
from localmind.evaluation.review import (
    ReviewStore, atomic_json, build, extract_corpus, inspect_source, make_questions, validate_question,
)
from localmind.evaluation.review_ui import ReviewWindow


@pytest.fixture(autouse=True)
def review_fonts(qt_application):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).exists():
            QFontDatabase.addApplicationFont(path)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.db"
    db = LocalMindDB(path)
    db.conn.execute("INSERT INTO settings(key,value) VALUES('private_token','not-to-copy')")
    db.conn.execute("""INSERT INTO chat_sources(id,source_key,platform,account,conversation,kind,updated_at)
        VALUES(1,'source','wechat','account','测试项目群','weflow','2026-01-01')""")
    for i in range(1, 36):
        db.conn.execute("""INSERT INTO files(id,path,name,ext,size,created_at,modified_at,scan_status,parse_status,ocr_status)
            VALUES(?,?,?,'.txt',0,'2026-01-01','2026-01-01','indexed','parsed','skipped')""",
            (i, str(tmp_path / f"file{i}.txt"), f"项目{i}离线记录方案.txt"))
        db.conn.execute("""INSERT INTO chunks(id,file_id,chunk_text,source_type,created_at)
            VALUES(?,?,?,'text','2026-01-01')""", (i, i, f"ABC{i} ABC{i} ABC{i} 项目{i}的离线存储方案包含备用电源和数据恢复流程。"))
        db.conn.execute("""INSERT INTO chat_messages(id,source_id,external_id,sender,sent_at,body,observed_at)
            VALUES(?,1,?,'测试成员','2026-01-01','现场验收时重点检查停电期间的数据缓存','2026-01-01')""", (i, str(i)))
        db.conn.execute("""INSERT INTO chat_attachments(id,message_id,ordinal,name,path_hint,sha256,kind,file_id,context)
            VALUES(?,?,0,'方案.txt','','','explicit',?,?)""", (i, i, i, f"TASK{i} TASK{i} TASK{i} 请使用这个方案检查离线缓存"))
    db.conn.commit()
    db.close()
    return path


@pytest.fixture
def dataset(source, tmp_path):
    root = tmp_path / "pilot"
    build(source, root)
    return root


def confirmed(store):
    state = copy.deepcopy(store.state)
    q = state["questions"][0]
    q.update(status="confirmed", notes="正文明确记载备用电源与离线缓存", judgments={q["suggested_file"]: 3})
    return state


def test_build_readonly_private_and_deterministic(source, tmp_path):
    before = source.read_bytes()
    assert inspect_source(source)["linked_files"] == 35
    root = tmp_path / "pilot"
    summary = build(source, root)
    assert summary["questions"] == 30 and summary["distinct_suggested_targets"] == 30
    store = ReviewStore(root)
    assert all(q["status"] == "draft" and not q["judgments"] for q in store.state["questions"])
    assert "not-to-copy" not in (root / "corpus.json").read_text(encoding="utf-8")
    assert source.read_bytes() == before
    assert len(store.documents) == 35
    with pytest.raises(ValueError, match="已存在"):
        build(source, root)
    assert extract_corpus(source)[0] == extract_corpus(source)[0]


def test_confirm_resume_export(dataset):
    store = ReviewStore(dataset)
    with pytest.raises(ValueError, match="尚无"):
        store.export()
    store.save(confirmed(store))
    loaded = ReviewStore(dataset)
    assert loaded.state["questions"][0]["status"] == "confirmed"
    payload = json.loads(loaded.export().read_text(encoding="utf-8"))
    assert len(payload["questions"]) == 1
    assert payload["strict_positive_grade"] == 2
    assert payload["excluded_questions"] == 29
    assert not payload["independent_test_set"]


@pytest.mark.parametrize("change", [
    {"text": ""}, {"notes": ""}, {"judgments": {}}, {"outcome": "uncertain"},
    {"judgments": {"foreign": 3}}, {"judgments": {"f0001": True}},
])
def test_confirmation_validation(dataset, change):
    store = ReviewStore(dataset)
    q = confirmed(store)["questions"][0]
    q.update(change)
    with pytest.raises(ValueError):
        validate_question(q, store.ids, confirming=True)


def test_no_complete_answer_keeps_partial_and_requires_all(dataset):
    store = ReviewStore(dataset)
    q = confirmed(store)["questions"][0]
    q.update(outcome="no_complete_answer", judgments={q["suggested_file"]: 1}, all_reviewed=True)
    with pytest.raises(ValueError, match="全部"):
        validate_question(q, store.ids, confirming=True)
    for fid in store.ids:
        q["judgments"].setdefault(fid, 0)
    validate_question(q, store.ids, confirming=True)
    q["all_reviewed"] = False
    with pytest.raises(ValueError):
        validate_question(q, store.ids, confirming=True)


def test_concurrent_edit_protection(dataset):
    first, second = ReviewStore(dataset), ReviewStore(dataset)
    first.save(confirmed(first))
    with pytest.raises(ValueError, match="另一个窗口"):
        second.save(confirmed(second))
    with pytest.raises(ValueError, match="外部修改"):
        second.export()


def test_duplicate_drafts_do_not_pad_count():
    docs = [{"file_id": f"f{i}", "name": "same.txt", "body": [], "sources": []} for i in range(3)]
    with pytest.raises(ValueError, match="不重复"):
        make_questions(docs, 3)


def test_corpus_tamper_protection(dataset):
    store = ReviewStore(dataset)
    corpus = copy.deepcopy(store.corpus)
    corpus["documents"][0]["name"] = "changed"
    atomic_json(dataset / "corpus.json", corpus)
    with pytest.raises(ValueError, match="变化"):
        ReviewStore(dataset)
    with pytest.raises(ValueError, match="变化"):
        store.export()
    with pytest.raises(ValueError, match="变化"):
        store.save(store.state)


def test_failed_save_retains_previous_state(dataset, monkeypatch):
    store = ReviewStore(dataset)
    old = (dataset / "review.json").read_bytes()
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr("localmind.evaluation.review.atomic_json", fail)
    with pytest.raises(OSError):
        store.save(confirmed(store))
    assert (dataset / "review.json").read_bytes() == old
    assert store.state["revision"] == 0


def test_empty_source_has_no_partial_output(tmp_path):
    path = tmp_path / "empty.db"
    LocalMindDB(path).close()
    with pytest.raises(ValueError):
        build(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_chat_units_retained_and_used(source):
    db = LocalMindDB(source)
    db.conn.execute("""INSERT INTO chunks(id,file_id,chunk_text,source_type,created_at)
        VALUES(100,1,'平台: wechat\n会话: 测试项目群\n上下文（引用 / 相邻消息，非附件内容）:\n现场讨论数据库迁移与回滚验收','chat','2026-01-01')""")
    db.conn.execute("""INSERT INTO chat_units(chunk_id,owner_file_id,message_id,attachment_id,unit_key,fingerprint)
        VALUES(100,1,1,1,'unit','fingerprint')""")
    db.conn.commit()
    db.close()
    docs, _ = extract_corpus(source)
    doc = next(d for d in docs if d['source_file_id'] == 1)
    assert '数据库迁移' in doc['chat_chunks'][0]['text']


def test_ui_review_navigation_resume(dataset, qt_application):
    window = ReviewWindow(dataset)
    window.show()
    qt_application.processEvents()
    assert window.grade.currentData() is None
    window.query.setPlainText("找一下停电后还能继续缓存数据的方案")
    window.notes.setPlainText("必须具有备用电源与离线缓存两个条件")
    window.grade.setCurrentIndex(window.grade.findData(3))
    window.confirm()
    assert window.index == 1
    assert window.state["questions"][0]["status"] == "confirmed"
    window.navigate(0)
    window.query.setPlainText("修改后的真实查询")
    assert window.state["questions"][0]["status"] == "draft"
    window.close()
    assert ReviewStore(dataset).state["questions"][0]["text"] == "修改后的真实查询"
    reopened = ReviewWindow(dataset)
    assert reopened.query.toPlainText() == "修改后的真实查询"
    reopened.close()


def test_ui_lock_and_synthetic_layout(dataset, qt_application):
    window = ReviewWindow(dataset)
    with pytest.raises(ValueError, match="另一个窗口"):
        ReviewWindow(dataset)
    directory = os.environ.get("LOCALMIND_REVIEW_SCREENSHOTS")
    for width, height in ((1320, 860), (980, 700)):
        window.resize(width, height)
        window.show()
        qt_application.processEvents()
        assert window.next.isVisible() and window.query.isVisible()
        assert window.receipt.geometry().bottom() < window.height()
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(Path(directory) / f"review-{width}.png"))
    window.close()


def test_ui_noanswer_manual_confirmation(dataset, qt_application, monkeypatch):
    window = ReviewWindow(dataset)
    window.notes.setPlainText("检查全部候选，无一满足全部要求")
    window.outcome.setCurrentIndex(window.outcome.findData("no_complete_answer"))
    window.grade.setCurrentIndex(window.grade.findData(1))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.mark_remaining()
    window.all_reviewed.setChecked(True)
    window.confirm()
    assert window.state["questions"][0]["status"] == "confirmed"
    assert 1 in window.state["questions"][0]["judgments"].values()
    window.close()
