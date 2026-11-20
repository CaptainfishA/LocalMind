from unittest.mock import Mock

import pytest

from localmind.db import LocalMindDB
from localmind.models import TextChunk
from localmind.query_processing import process_query
from localmind.searcher import lexical_evidence
from test_weflow_v04 import index_file, searcher


@pytest.mark.parametrize("query", [
    "服务器房间制冷机器的滤网多久洗一次",
    "小陈发给财务那份外出参展住宿报销的材料",
])
def test_sentences_are_descriptive(query):
    assert process_query(query).descriptive


@pytest.mark.parametrize("query", [
    "预算", "帮我找编号 PC-201 的采购合同", "帮我找那个标注SKU120的产品资料",
    "帮我找付款计划.xlsx这份文件", "票据编号123456789的报销记录", r"D:\文档\采购说明.txt",
])
def test_literal_queries_keep_exact_matching(query):
    assert not process_query(query).descriptive


def test_path_matches_are_not_body_evidence():
    query = process_query("年度预算")
    body = {"name": "资料.txt", "path": "D:/年度预算/资料.txt", "chunk_text": "设备维护计划"}
    path_score, _, reasons = lexical_evidence(body, query)
    assert path_score <= 0.08
    assert "路径关键词" in reasons and "正文关键词" not in reasons
    named = dict(body, name="年度预算.txt")
    assert "正文关键词" not in lexical_evidence(named, query)[2]
    assert "文件名关键词" in lexical_evidence(named, query)[2]
    assert lexical_evidence(dict(body, chunk_text="年度预算"), query)[0] > path_score


def test_short_term_candidates_rank_before_limit_and_limit_long_document(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        generic = index_file(db, tmp_path / "general.txt", "设备维护")
        db.insert_chunks(generic["id"], [TextChunk(chunk_text="设备维护", source_type="txt") for _ in range(120)])
        target = index_file(db, tmp_path / "target.txt", "低温电池设备维护")
        target_chunk = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (target["id"],)).fetchone()[0]
        rows = db.search_like_terms(["设备", "低温", "电池"], limit=5)
        assert rows[0]["chunk_id"] == target_chunk
        assert len(rows) == 5
        assert len(db.search_like_terms(["设备"], limit=100)) == 5
        db.set_search_scope({target_chunk})
        assert [r["chunk_id"] for r in db.search_like_terms(["设备"], restricted=True)] == [target_chunk]
    finally:
        db.close()


def test_substring_candidates_do_not_treat_input_as_wildcards(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        index_file(db, tmp_path / "normal.txt", "预算测算")
        target = index_file(db, tmp_path / "literal.txt", "code_A 与 50%")
        cid = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (target["id"],)).fetchone()[0]
        assert [r["chunk_id"] for r in db.search_like_terms(["code_"])] == [cid]
        assert [r["chunk_id"] for r in db.search_like_terms(["%"])] == [cid]
    finally:
        db.close()


def test_semantic_order_has_no_lexical_filename_or_length_bonus(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        first = index_file(db, tmp_path / "one.txt", "机房精密空调每个月清洗滤网")
        second = index_file(db, tmp_path / "服务器制冷清理.txt", "服务器房间制冷机器的滤网多久洗一次")
        a = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (first["id"],)).fetchone()[0]
        b = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (second["id"],)).fetchone()[0]
        extra = db.insert_chunks(second["id"], [TextChunk(chunk_text="其他无关段落", source_type="txt") for _ in range(10)])
        service = searcher(db, mode="semantic")
        scores = {a: .91, b: .90, **{cid: .10 for cid in extra}}
        service._semantic_candidates = lambda *args: (scores, {cid: i for i, cid in enumerate(scores, 1)}, "mock", "mock")
        result = service.search("服务器房间制冷机器的滤网多久洗一次")
        assert [r.file_id for r in result] == [first["id"], second["id"]]
        assert result[0].score == pytest.approx(.91) and result[1].score == pytest.approx(.90)
        assert "多处命中" not in result[1].match_reason
        assert service.last_diagnostics.retrieval_mode == "semantic"
    finally:
        db.close()


def test_descriptive_hybrid_and_literal_profile_differ(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        service = searcher(db)
        strong_semantic = (0.15, 0.85, 0.6, 0, 0)
        generic_literal = (0.55, 0.60, 0.9, 0, 0)
        assert service._base_score(*strong_semantic, descriptive=True) > service._base_score(*generic_literal, descriptive=True)
        assert service._base_score(*strong_semantic) < service._base_score(*generic_literal)
        service.retrieval_mode = "keyword"
        service.embedding_service.encode_query = Mock(side_effect=AssertionError("embedding must not run"))
        service.search("编号 PC-201")
        service.embedding_service.encode_query.assert_not_called()
    finally:
        db.close()


def test_hash_fallback_does_not_get_semantic_priority(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        index_file(db, tmp_path / "sample.txt", "服务器机房的空调滤网每个月清洗")
        service = searcher(db)
        service.search("服务器房间制冷机器的滤网多久洗一次")
        assert service.last_diagnostics.active_embedding_model == "localmind-hash-embedding"
        assert service.last_diagnostics.ranking_profile == "balanced"
    finally:
        db.close()


def test_mode_changes_reuse_worker_vector_index_and_validate_mode(tmp_path, monkeypatch):
    from localmind.ui import SearchWorker
    from localmind.models import SearchQuery
    from localmind.embeddings import EmbeddingService
    db = LocalMindDB(tmp_path / "index.db")
    index_file(db, tmp_path / "doc.txt", "检索验证")
    db.close()
    worker = SearchWorker(str(tmp_path / "index.db"), str(tmp_path / "cache"), "cpu")
    worker.embedding_service = EmbeddingService(force_hash=True)
    reports, errors = [], []
    worker.finished.connect(lambda results, diagnostics, timing: reports.append((diagnostics, timing)))
    worker.failed.connect(errors.append)
    try:
        worker.process({"query": SearchQuery("检索验证"), "enable_reranker": False, "retrieval_mode": "keyword"})
        service = worker.service
        rebuild = Mock(side_effect=AssertionError("mode switch must not rebuild"))
        monkeypatch.setattr(service.vector_index, "rebuild", rebuild)
        for mode in ("semantic", "hybrid", "keyword"):
            worker.process({"query": SearchQuery("检索验证"), "enable_reranker": False, "retrieval_mode": mode})
            assert reports[-1][0].retrieval_mode == mode
            assert reports[-1][1]["service_reused"] and worker.service is service
        assert not errors and rebuild.call_count == 0
        worker.process({"query": SearchQuery("test"), "enable_reranker": False, "retrieval_mode": "unknown"})
        assert errors == ["无效的检索模式"]
    finally:
        worker.close()
