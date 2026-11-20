from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.clause_semantics import evidence_key, evidence_parts, file_clause_evidence
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService, HashEmbeddingBackend, SentenceTransformerBackend
from localmind.models import SearchQuery
from localmind.query_processing import process_query, semantic_clauses
from localmind.reranker import RerankerService
from localmind.searcher import SearchService, _PreparedSemantic
from localmind.vector_index import LocalVectorIndex
from test_search_snapshot import CountedIndex, FixedQuery, document


QUERY = "杭州轮岗交接时要用的那份；讲的是服务器房间停电后怎样保持供电"


class ClauseEmbedding(FixedQuery):
    def __init__(self, db):
        self.db = db
        self.calls = []

    def encode_query(self, text):
        self.calls.append((1, self.db.conn.in_transaction))
        return np.array([1., 0., 0., 0.], dtype="float32")

    def encode_queries(self, texts):
        self.calls.append((len(texts), self.db.conn.in_transaction))
        return np.array([[1., 0., 0., 0.],
                         *[[0., 1., 0., 0.] if "杭州" in text else [0., 0., 1., 0.]
                           for text in texts[1:]]], dtype="float32")


def attach(db, path, vector, *, source, conversation="轮岗组"):
    ChatStore(db).ingest(
        platform="wechat", account="test", conversation=conversation, source_key=source,
        messages=[ChatMessage("one", "杭州轮岗交接安排", sender="模拟用户",
                              attachments=[Attachment(path.name, str(path))])],
    )
    aid = next(row["id"] for row in ChatStore(db).attachments() if row["local_path"] == str(path))
    cid = db.conn.execute("SELECT chunk_id FROM chat_units WHERE attachment_id=?", (aid,)).fetchone()[0]
    db.insert_vectors([cid], np.array([vector], dtype="float32"), model_name=FixedQuery.active_model_name)
    return aid, cid


@pytest.fixture
def corpus(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    target = tmp_path / "target.txt"
    decoy = tmp_path / "decoy.txt"
    fid, cid, _ = document(db, target, "机房备用电池的维护计划", (.2, 0, .98, 0))
    other, _, _ = document(db, decoy, "办公家具采购", (.1, 0, .05, .99))
    aid, chat = attach(db, target, (.65, .76, 0, 0), source="target")
    attach(db, decoy, (.9, .43, 0, 0), source="decoy", conversation="别的轮岗组")
    embed = ClauseEmbedding(db)
    index = CountedIndex()
    service = SearchService(db, embedding_service=embed, vector_index=index,
                            reranker_service=RerankerService(enabled=False), enable_clause_semantics=False)
    try:
        yield db, service, embed, index, fid, other, aid, cid, chat
    finally:
        db.close()


@pytest.mark.parametrize("text", [
    QUERY,
    "我想找服务器房间停电后的供电说明，杭州轮岗交班时用的材料",
    "给客户学习班演示时使用的文件。讲的是报销申请怎样办理",
])
def test_two_explicit_descriptions_are_preserved_without_guessing_roles(text):
    clauses = semantic_clauses(process_query(text))
    assert len(clauses) == 2
    assert all(part in process_query(text).original for part in clauses)


@pytest.mark.parametrize("text", [
    "杭州轮岗交接要用的备用电池维护资料",
    "合同，发票",
    "不是杭州轮岗组的那份，而是机房维护的那份",
    "帮我找轮岗交接的材料，或者机房维护的资料",
    "帮我找杭州轮岗交接的材料，或服务器房间备用供电的资料",
    "不要旧版的轮岗资料，讲的是机房停电方案",
    "关于新同事的交接安排；关于新同事的交接安排",
    "第一部分介绍交接安排，第二部分介绍机房维护，第三部分介绍采购流程",
    "杭州轮岗用的文件 target.pdf，讲的是服务器机房的停电处理",
    "杭州轮岗交接时用的资料，2026",
    "轮岗资料" * 100 + "；服务器房间的供电资料",
])
def test_ambiguous_short_literal_or_nonconjunctive_descriptions_do_not_split(text):
    assert not semantic_clauses(process_query(text))


def test_batched_query_encoding_uses_query_instruction_once_per_item():
    class Model:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return np.eye(len(texts), 4, dtype="float32")

    model = Model()
    service = EmbeddingService(force_hash=True)
    service._backend = SentenceTransformerBackend(model, "synthetic")
    result = service.encode_queries(["整句", "描述一", "描述二"])
    assert result.shape == (3, 4)
    assert model.calls[0][0] == [SentenceTransformerBackend.query_instruction + text
                                 for text in ["整句", "描述一", "描述二"]]
    assert len(model.calls) == 1
    assert service.encode_query("单句").shape == (4,)
    assert len(model.calls) == 2


def test_vector_scoring_is_exact_scoped_and_does_not_rebuild():
    index = CountedIndex()
    index.rebuild([7, 3, 9], np.array([[1., 0.], [0., 1.], [.6, .8]], dtype="float32"))
    result = index.score_ids(np.eye(2, dtype="float32"), [9, 7, 9, 100])
    assert set(result) == {7, 9}
    assert result[7] == pytest.approx((1, 0))
    assert result[9] == pytest.approx((.6, .8))
    assert index.rebuilds == 1
    with pytest.raises(ValueError, match="Invalid"):
        index.score_ids(np.array([[np.nan, 0]]), [7])
    with pytest.raises(ValueError, match="Invalid"):
        index.score_ids(np.zeros((2, 3)), [7])


def item(cid, fid, source="txt", *, stale=False, target=None):
    return {"chunk_id": cid, "row": {
        "file_id": fid, "name": "same.txt", "path": f"/synthetic/{fid}/same.txt", "source_type": source,
        "chat_refs": [{"target_id": fid if target is None else target,
                       "linked_hash": "old" if stale else "new", "target_hash": "new"}],
    }}


def test_coverage_never_combines_different_targets_or_stale_chat():
    items = [item(1, 10), item(2, 10, "chat", stale=True), item(3, 11, "chat"), item(4, 10, "chat", target=11)]
    scores = {1: (.1, 1.), 2: (1., .2), 3: (1., 0.), 4: (1., 1.)}
    result = file_clause_evidence(items, scores, ("描述一", "描述二"))
    assert result[10][0] == pytest.approx(2 * .1 / 1.1)
    assert [part["chunk_id"] for part in result[10][1]] == [1]
    assert result[11] == (0, ())
    assert result[10][1][0]["query_clause"] == "描述一\n描述二"


def test_evidence_merges_lexical_and_semantic_contributors_without_mutation():
    a, b, c = item(1, 10), item(2, 10, "chat"), item(3, 10)
    value = {"joint_evidence": (a, b), "semantic_evidence": ({**a, "query_clause": "描述一"}, {**c, "query_clause": "描述二"})}
    merged = evidence_parts(value)
    assert [part["chunk_id"] for part in merged] == [1, 2, 3]
    assert merged[0]["query_clause"] == "描述一"
    assert "query_clause" not in a
    assert evidence_key(value) == (1, 2, 3)


def test_file_semantic_coverage_corrects_context_only_leader_and_explains_both_sources(corpus):
    db, service, embed, index, fid, other, aid, cid, chat = corpus
    query = SearchQuery(QUERY, scope="chat_files")
    before = service.search(query)
    assert before[0].file_id == other
    service.enable_clause_semantics = True
    embed.calls.clear()
    result = service.search(query)
    assert result[0].file_id == fid
    assert result[0].semantic_score > result[1].semantic_score
    assert service.last_diagnostics.clause_count == 2
    assert embed.calls == [(3, False)]
    assert index.rebuilds == 1
    assert {part.source_type for part in result[0].supporting_evidence} == {"txt", "chat"}
    assert {part.chunk_id for part in result[0].supporting_evidence} == {cid, chat}
    assert all(part.query_clause for part in result[0].supporting_evidence)
    assert result[0].attachment_id == aid
    assert "分句语义覆盖" in result[0].match_reason


@pytest.mark.parametrize("mode,evidence,scope", [
    ("semantic", "all", "chat_files"), ("keyword", "all", "chat_files"),
    ("hybrid", "all", "messages"),
])
def test_nonhybrid_modes_and_message_scope_are_unchanged(corpus, mode, evidence, scope):
    _, service, embed, _, _, _, _, _, _ = corpus
    service.retrieval_mode = mode
    query = SearchQuery(QUERY, scope=scope, evidence=evidence)
    before = [(r.file_id, r.score) for r in service.search(query)]
    service.enable_clause_semantics = True
    embed.calls.clear()
    after = [(r.file_id, r.score) for r in service.search(query)]
    assert before == after
    assert service.last_diagnostics.clause_count == 0
    assert all(count == 1 for count, _ in embed.calls)


def test_clause_recall_and_evidence_obey_original_conversation_filter(corpus):
    db, service, _, _, fid, other, aid, _, _ = corpus
    service.enable_clause_semantics = True
    result = service.search(SearchQuery(QUERY, scope="chat_files", conversation="别的轮岗组"))
    assert [r.file_id for r in result] == [other]
    assert all(part.chunk_id not in {
        row[0] for row in db.conn.execute("SELECT chunk_id FROM chat_units WHERE attachment_id=?", (aid,))
    } for part in result[0].supporting_evidence)


def test_clause_recall_failure_preserves_original_whole_sentence_results(corpus, monkeypatch):
    _, service, _, index, _, _, _, _, _ = corpus
    query = SearchQuery(QUERY, scope="chat_files")
    before = [(r.file_id, r.score) for r in service.search(query)]
    original = index.search
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("synthetic clause recall failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(index, "search", fail_second)
    service.enable_clause_semantics = True
    assert [(r.file_id, r.score) for r in service.search(query)] == before
    assert service.last_diagnostics.clause_count == 0
    assert "保留整句召回" in service.last_diagnostics.clause_status


def test_clause_pool_is_bounded_and_respects_vector_scope():
    rng = np.random.default_rng(7)
    index = LocalVectorIndex()
    index.rebuild(list(range(1000)), rng.standard_normal((1000, 4)).astype("float32"))
    service = SearchService.__new__(SearchService)
    service.vector_index = index
    service.indexed_embedding_model = "synthetic"
    service._allowed_ids = set(range(100, 900))
    service._target_mode = "files"
    service._target_groups = {"files": {cid: str(cid) for cid in range(1000)}}
    prepared = _PreparedSemantic(np.array([1., 0, 0, 0]), "synthetic", "ready", ("a", "b"),
                                 np.array([[0., 1, 0, 0], [0., 0, 1, 0]]))
    scores, ranks, _, _ = service._semantic_candidates("query", 40, prepared)
    assert 40 < len(scores) <= 120
    assert set(scores) == set(ranks)
    assert set(scores) <= service._allowed_ids


def test_reranker_receives_all_contributors_once_per_file(corpus):
    _, service, _, _, fid, _, _, _, _ = corpus
    passages = []

    def rerank(_text, rows):
        passages.extend(rows)
        return [.9 if "target.txt" in row else .4 for row in rows]

    service.enable_clause_semantics = True
    service.reranker_service = SimpleNamespace(
        enabled=True, recommended_top_k=30, model_name="synthetic", load_error=None,
        runtime=SimpleNamespace(name="cpu"), rerank=rerank,
    )
    assert service.search(SearchQuery(QUERY, scope="chat_files"))[0].file_id == fid
    assert len(passages) == 2
    assert all("文件正文：" in row and "该文件的聊天线索：" in row for row in passages)


def test_hash_backend_does_not_claim_compositional_semantics(tmp_path, monkeypatch):
    from test_weflow_v04 import index_file, ingest

    db = LocalMindDB(tmp_path / "hash.db")
    try:
        file = index_file(db, tmp_path / "hash.txt", "机房备用电池")
        ingest(db, [ChatMessage("one", "杭州轮岗交接", attachments=[Attachment(file["name"], file["path"])])],
               embedding=EmbeddingService(force_hash=True))
        embed = EmbeddingService(force_hash=True)
        calls = []
        original = embed.encode_queries
        monkeypatch.setattr(embed, "encode_queries", lambda texts: calls.append(len(texts)) or original(texts))
        service = SearchService(db, embedding_service=embed, reranker_service=RerankerService(enabled=False))
        service.search(SearchQuery(QUERY, scope="chat_files"))
        assert calls == [1] and service.last_diagnostics.clause_count == 0
        assert service.indexed_embedding_model == HashEmbeddingBackend.name
    finally:
        db.close()


def test_clause_recall_can_find_a_file_outside_the_original_top20(tmp_path):
    db = LocalMindDB(tmp_path / "crowded.db")
    try:
        for number in range(170):
            path = tmp_path / f"decoy-{number:03d}.txt"
            document(db, path, "办公家具采购清单", (.8, .1, .01, .5))
            attach(db, path, (.9, .43, .01, 0), source=str(number))
        target = tmp_path / "target.txt"
        fid, _, _ = document(db, target, "机房备用电池维护", (.1, 0, .99, 0))
        attach(db, target, (.1, .99, 0, 0), source="target")
        index = CountedIndex()
        service = SearchService(
            db, embedding_service=ClauseEmbedding(db), vector_index=index,
            reranker_service=RerankerService(enabled=False), enable_clause_semantics=False,
        )
        query = SearchQuery(QUERY, scope="chat_files", limit=20)
        assert fid not in [result.file_id for result in service.search(query)]
        service.enable_clause_semantics = True
        assert service.search(query)[0].file_id == fid
        assert service.last_diagnostics.semantic_candidates <= 480
        assert index.rebuilds == 1
    finally:
        db.close()


def test_semantic_score_failure_keeps_original_scores(corpus, monkeypatch):
    _, service, _, index, _, _, _, _, _ = corpus
    query = SearchQuery(QUERY, scope="chat_files")
    before = [(r.file_id, r.score) for r in service.search(query)]
    original = index.score_ids

    def fail_for_candidates(vectors, ids):
        if len(ids):
            raise RuntimeError("synthetic scoring failure")
        return original(vectors, ids)

    monkeypatch.setattr(index, "score_ids", fail_for_candidates)
    service.enable_clause_semantics = True
    assert [(r.file_id, r.score) for r in service.search(query)] == before
    assert service.last_diagnostics.clause_count == 0
    assert "保留整句排序" in service.last_diagnostics.clause_status


def test_clause_evidence_ui_fits_and_retains_the_actual_source(corpus):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication
    from localmind.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db, service, _, _, fid, _, aid, _, _ = corpus
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(db.db_path.parent / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    service.enable_clause_semantics = True
    query = "聊天里说杭州轮岗交接要用的材料；内容是服务器房间停电后的供电保障"
    results = service.search(SearchQuery(query, scope="chat_files"))
    window = MainWindow(db)
    window.search_input.setText(query)
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    window.show()
    try:
        window._search_finished(results, service.last_diagnostics, {"total_ms": 27, "index_refresh_ms": 0})
        assert window.selected_result().file_id == fid
        assert window.selected_result().attachment_id == aid
        assert "分句语义" in window.search_status.text()
        assert "调和平均" in window.search_status.toolTip()
        assert "聊天 / 正文" in window.search_status.toolTip()
        assert "对应描述：" not in window.preview.toPlainText()
        assert "杭州轮岗交接安排" in window.preview.toPlainText()
        assert "机房备用电池的维护计划" in window.preview.toPlainText()
        window.evidence_expand.click()
        assert "对应描述：" in window.preview.toPlainText()
        assert "聊天里说杭州轮岗交接" in window.preview.toPlainText()
        window.evidence_expand.click()
        assert "对应描述：" not in window.preview.toPlainText()
        assert window.evidence_expand.toolTip() == "展开匹配详情"
        window.preview_tabs.setCurrentIndex(1)
        assert not window.evidence_expand.isVisible()
        window.preview_tabs.setCurrentIndex(0)
        assert window.evidence_expand.isVisible()
        assert "文件正文" in window.preview.toPlainText() and "聊天线索" in window.preview.toPlainText()
        assert window.preview_tabs.isTabEnabled(1)
        assert "杭州轮岗交接安排" in window.context_preview.toPlainText()
        from test_file_preview import wait
        wait(app, lambda: window.file_preview._payload is not None)
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            assert not window.preview_tabs.tabBar().geometry().intersects(window.evidence_expand.geometry())
            for widget in (window.preview, window.search_button, window.search_status, window.file_actions[-1]):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"clause-evidence-{width}.png"))
    finally:
        window.close()
        app.processEvents()
