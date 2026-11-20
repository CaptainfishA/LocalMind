from dataclasses import replace
import os
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.clause_semantics import file_clause_evidence
from localmind.db import LocalMindDB
from localmind.models import SearchQuery, TextChunk
from localmind.query_processing import process_query, single_source_clauses
from localmind.reranker import RerankerService
from localmind.scanner import build_file_record
from localmind.searcher import SearchCancelled, SearchService
from test_search_snapshot import CountedIndex, FixedQuery
from test_file_preview import preview_app, wait  # noqa: F401
from test_clause_semantics import item


QUERY = "断电以后机房怎样持续供电；轮岗新人培训演示时会用的材料"
BODY_A = "电力中断时应启用后备电源保障计算设备正常运行。"
BODY_B = "本资料留作新入职员工讲解示范的教学参考。"


@pytest.mark.parametrize("source,text", [
    ("content", QUERY), ("chat", QUERY),
    ("content", "正文说明断电以后机房怎样持续供电，内容是轮岗新人培训演示时会用的材料"),
    ("chat", "聊天里说断电以后机房怎样持续供电，聊天里说轮岗新人培训演示时会用的材料"),
    *[("content", "断电以后机房怎样持续供电，" + conjunction + "轮岗新人培训演示时会用的材料")
      for conjunction in ("而且", "并且", "同时", "还要", "还需", "以及", "另外")],
])
def test_only_explicit_conjunctions_are_decomposed(source, text):
    query = process_query(text)
    clauses = single_source_clauses(query, source)
    assert len(clauses) == 2 and all(part in query.original for part in clauses)


@pytest.mark.parametrize("source,text", [
    ("content", "外面的客人要进办公楼，需要办哪些手续"),
    ("chat", "外面的客人要进办公楼，需要办哪些手续"),
    ("content", "正文说明断电以后机房怎样持续供电；聊天里说轮岗新人培训演示时会用的材料"),
    ("chat", "正文说明断电以后机房怎样持续供电；聊天里说轮岗新人培训演示时会用的材料"),
    ("content", "关于‘断电以后；机房怎样持续供电’的资料，轮岗新人培训演示时会用的材料"),
    ("content", "断电以后机房怎样持续供电；或者轮岗新人培训演示时会用的材料"),
    ("chat", "不要断电以后机房怎样持续供电；轮岗新人培训演示时会用的材料"),
    ("content", QUERY + "；明年公司的采购申请办理流程"),
    ("content", "轮岗新人培训演示时会用的材料"),
])
def test_ambiguous_or_conflicting_sources_keep_whole_query(source, text):
    assert single_source_clauses(process_query(text), source) == ()


def test_blend_keeps_whole_query_contributor_and_never_uses_stale_chat():
    candidates = [item(1, 10), item(2, 10), item(3, 10), item(4, 10, "chat", stale=True), item(5, 11)]
    scores = {1: (.9, .1), 2: (.2, .8), 3: (.4, .4), 4: (1., 1.), 5: (.1, .1)}
    whole = {1: .3, 2: .4, 3: .7, 4: 1., 5: .99}
    plain = file_clause_evidence(candidates, scores, ("条件甲", "条件乙"))
    result = file_clause_evidence(candidates, scores, ("条件甲", "条件乙"), whole_scores=whole)
    assert result[10][0] == pytest.approx((2 * .9 * .8 / 1.7 + .7) / 2)
    assert plain[10][0] == pytest.approx(2 * .9 * .8 / 1.7)
    assert [part["chunk_id"] for part in result[10][1]] == [1, 2, 3]
    assert result[10][1][0]["query_clause"] == "条件甲"
    assert result[10][1][1]["query_clause"] == "条件乙"
    assert "query_clause" not in candidates[0]
    assert whole[4] == 1.


def test_whole_query_score_cannot_supply_a_missing_required_source():
    candidates = [item(1, 10, "chat")]
    result = file_clause_evidence(candidates, {1: (.9, .9)},
        ("聊天里说需要新人培训资料", "正文说明机房供电保障流程"), whole_scores={1: 1.})
    assert result[10] == (0., ())


class ConditionEmbedding(FixedQuery):
    def __init__(self, db):
        self.db = db
        self.calls = []
        self.documents = []
        self.on_encode = lambda: None

    def encode_query(self, text):
        assert not self.db.conn.in_transaction
        self.calls.append((text,))
        self.on_encode()
        return np.array([1., 0., 0., 0.], dtype="float32")

    def encode_queries(self, texts):
        assert not self.db.conn.in_transaction
        self.calls.append(tuple(texts))
        self.on_encode()
        return np.array([[1., 0., 0., 0.],
                         *[[0., 1., 0., 0.] if "供电" in text else [0., 0., 1., 0.]
                           for text in texts[1:]]], dtype="float32")

    def encode_documents(self, texts):
        self.documents.extend(texts)
        raise AssertionError("No document inference needed for these short chunks")


@pytest.fixture
def source_corpus(tmp_path):
    db = LocalMindDB(tmp_path / "source.db")
    files, bodies, chats = {}, {}, {}
    try:
        for name, texts, vectors in (
            ("target", (BODY_A, BODY_B), ((.25, .96, .1, 0), (.2, .1, .97, 0))),
            ("decoy", ("普通家具采购明细", "会务物料留档目录"), ((.85, .51, .05, 0), (.8, .59, .05, 0))),
        ):
            path = tmp_path / f"{name}.txt"
            path.write_text("\n".join(texts), encoding="utf-8")
            record = build_file_record(path, include_hash=True)
            files[name], bodies[name] = db.replace_file_index(
                record, [TextChunk(text, "txt") for text in texts], parse_status="done", ocr_status="not_required",
                embeddings=np.array(vectors, dtype="float32"), model_name=FixedQuery.active_model_name)
            chats[name] = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                key = f"{name}-{i}"
                ChatStore(db).ingest(platform="wechat", account="synthetic", conversation=f"资料组{i}",
                    source_key=key, messages=[ChatMessage("one", text, sender=f"同事{i}",
                        sent_at=f"2026-10-{20 + i}T10:00:00+08:00",
                        attachments=[Attachment(path.name, str(path))])])
                cid = db.conn.execute("""SELECT u.chunk_id FROM chat_units u
                    JOIN chat_messages m ON m.id=u.message_id JOIN chat_sources s ON s.id=m.source_id
                    WHERE s.source_key=? AND u.attachment_id IS NOT NULL""", (key,)).fetchone()[0]
                db.insert_vectors([cid], np.array([vector], dtype="float32"), model_name=FixedQuery.active_model_name)
                chats[name].append(cid)
        embedding, index = ConditionEmbedding(db), CountedIndex()
        service = SearchService(db, embedding_service=embedding, vector_index=index,
                                reranker_service=RerankerService(enabled=False))
        yield db, service, embedding, index, files, bodies, chats
    finally:
        db.close()


@pytest.mark.parametrize("scope,evidence", [("files", "all"), ("chat_files", "content"), ("chat_files", "chat")])
def test_same_target_conditions_use_only_selected_source_without_reindexing(source_corpus, scope, evidence):
    db, service, embedding, index, files, bodies, chats = source_corpus
    vectors = [tuple(row) for row in db.conn.execute("SELECT * FROM vectors ORDER BY chunk_id")]
    query = SearchQuery(QUERY, scope=scope, evidence=evidence, infer_dates=False)
    service.enable_clause_semantics = False
    assert service.search(query)[0].file_id == files["decoy"]
    service.enable_clause_semantics = True
    embedding.calls.clear()
    target = service.search(query)[0]
    assert target.file_id == files["target"]
    expected = chats if evidence == "chat" else bodies
    assert {part.chunk_id for part in target.supporting_evidence} == set(expected["target"])
    assert {part.source_type for part in target.supporting_evidence} == {"chat" if evidence == "chat" else "txt"}
    assert len(embedding.calls) == 1 and len(embedding.calls[0]) == 3 and not embedding.documents
    assert service.last_diagnostics.clause_count == 2
    assert ("聊天 / 聊天" if evidence == "chat" else "正文 / 正文") in service.last_diagnostics.clause_status
    assert index.rebuilds == 1
    assert [tuple(row) for row in db.conn.execute("SELECT * FROM vectors ORDER BY chunk_id")] == vectors
    assert bool(target.preview_anchor) == (evidence != "chat")


def test_filtered_chat_evidence_never_borrows_another_conversation(source_corpus):
    _, service, _, _, _, _, chats = source_corpus
    hits = service.search(SearchQuery(QUERY, scope="chat_files", evidence="chat", conversation="资料组0"))
    assert hits
    allowed = {ids[0] for ids in chats.values()}
    for hit in hits:
        assert hit.supporting_evidence
        assert {part.chunk_id for part in hit.supporting_evidence} <= allowed
        assert all(part.chat_source.conversation == "资料组0" for part in hit.supporting_evidence)


@pytest.mark.parametrize("evidence", ["content", "chat"])
def test_source_scoped_candidates_obey_sender_date_and_file_filters(source_corpus, evidence):
    _, service, _, _, files, bodies, chats = source_corpus
    query = SearchQuery(QUERY, scope="chat_files", evidence=evidence, sender="同事0",
                        modified_after="2026-10-20", modified_before="2026-10-21", path_contains="target", file_type="txt")
    hits = service.search(query)
    assert [hit.file_id for hit in hits] == [files["target"]]
    expected = {chats["target"][0]} if evidence == "chat" else set(bodies["target"])
    assert {part.chunk_id for part in hits[0].supporting_evidence} <= expected
    assert hits[0].chat_source.sender == "同事0"
    assert service.search(replace(query, file_type="pdf")) == []


def test_keyword_only_candidate_receives_actual_whole_query_score(source_corpus, monkeypatch):
    _, service, _, index, files, bodies, _ = source_corpus
    first, second = bodies["target"]
    recall = service._semantic_candidates

    def omit_first(*args):
        scores, ranks, model, status = recall(*args)
        scores.pop(first, None)
        ranks.pop(first, None)
        return scores, ranks, model, status

    monkeypatch.setattr(service, "_semantic_candidates", omit_first)
    monkeypatch.setattr(service, "_keyword_candidates", lambda *_: {first: 1})
    hits = service.search(SearchQuery(QUERY, scope="chat_files", evidence="content"))
    target = next(hit for hit in hits if hit.file_id == files["target"])
    scores = index.score_ids(np.eye(4, dtype="float32")[:3], [first, second])
    a, b = scores[first][1], scores[second][2]
    assert target.semantic_score == pytest.approx((2 * a * b / (a + b) + scores[first][0]) / 2)
    assert index.rebuilds == 1


@pytest.mark.parametrize("evidence", ["content", "chat"])
def test_clause_score_failure_keeps_whole_query_results(source_corpus, monkeypatch, evidence):
    _, service, _, index, *_ = source_corpus
    query = SearchQuery(QUERY, scope="chat_files", evidence=evidence)
    service.enable_clause_semantics = False
    baseline = [(hit.file_id, hit.score) for hit in service.search(query)]
    original = index.score_ids

    def fail_clauses(vectors, ids):
        if np.asarray(vectors).ndim == 2 and len(vectors) == 2:
            raise RuntimeError("synthetic coverage failure")
        return original(vectors, ids)

    monkeypatch.setattr(index, "score_ids", fail_clauses)
    service.enable_clause_semantics = True
    assert [(hit.file_id, hit.score) for hit in service.search(query)] == baseline
    assert service.last_diagnostics.clause_count == 0
    assert "保留整句排序" in service.last_diagnostics.clause_status


@pytest.mark.parametrize("scope,evidence", [("files", "all"), ("chat_files", "content"), ("chat_files", "chat")])
def test_cancel_does_not_publish_partial_source_results(source_corpus, scope, evidence):
    _, service, embedding, _, *_ = source_corpus
    previous = service.last_diagnostics
    stop = Event()
    embedding.on_encode = stop.set
    service.should_cancel = stop.is_set
    with pytest.raises(SearchCancelled):
        service.search(SearchQuery(QUERY, scope=scope, evidence=evidence))
    assert service.last_diagnostics is previous


@pytest.mark.parametrize("source", ["content", "chat"])
def test_source_conflict_preserves_filtered_whole_query(source_corpus, source):
    _, service, embedding, _, *_ = source_corpus
    query = SearchQuery("正文说明断电以后机房怎样持续供电；聊天里说轮岗新人培训演示时会用的材料",
                        scope="chat_files", evidence=source)
    service.enable_clause_semantics = False
    baseline = [(hit.file_id, hit.score) for hit in service.search(query)]
    service.enable_clause_semantics = True
    embedding.calls.clear()
    assert [(hit.file_id, hit.score) for hit in service.search(query)] == baseline
    assert service.last_diagnostics.clause_count == 0
    assert all(len(call) == 1 for call in embedding.calls)


@pytest.mark.parametrize("evidence", ["content", "chat"])
def test_single_source_explanation_and_preview_fit_both_window_sizes(preview_app, source_corpus, evidence):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from localmind.ui import MainWindow

    db, service, _, _, files, *_ = source_corpus
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(db.db_path.parent / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    hits = service.search(SearchQuery(QUERY, scope="chat_files", evidence=evidence, infer_dates=False))
    window = MainWindow(db)
    window.search_input.setText(QUERY)
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    window.evidence_filter.setCurrentIndex(window.evidence_filter.findData(evidence))
    window.show()
    try:
        window._search_finished(hits, service.last_diagnostics, {"total_ms": 35, "index_refresh_ms": 0})
        assert window.selected_result().file_id == files["target"]
        assert BODY_A in window.preview.toPlainText() and BODY_B in window.preview.toPlainText()
        label = "聊天 / 聊天" if evidence == "chat" else "正文 / 正文"
        assert label in window.search_status.toolTip()
        assert "对应描述：" not in window.preview.toPlainText()
        window.evidence_expand.click()
        assert "对应描述：" in window.preview.toPlainText()
        window.evidence_expand.click()
        wait(preview_app, lambda: window.file_preview._payload is not None)
        assert window.file_preview._target[0] == hits[0].path
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            preview_app.processEvents()
            for widget in (window.preview, window.file_preview, window.file_actions[0]):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            assert not window.preview_tabs.tabBar().geometry().intersects(window.evidence_expand.geometry())
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"single-{evidence}-{width}.png"))
    finally:
        window.close()
        preview_app.processEvents()
