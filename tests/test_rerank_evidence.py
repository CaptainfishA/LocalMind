"""Verify experimental input preservation, not production ranking accuracy."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from localmind.clause_semantics import evidence_parts
from localmind.query_processing import process_query
from localmind.searcher import make_evidence_snippet
from test_pending_clause_semantics import pending_corpus  # noqa: F401
from test_single_source_clauses import BODY_A, BODY_B, QUERY
from localmind.models import SearchQuery
from benchmarks.rerank_evidence_variants import FullEvidenceSearch


def capturing_reranker(received):
    return SimpleNamespace(enabled=True, recommended_top_k=30, model_name="synthetic", load_error=None,
        runtime=SimpleNamespace(name="cpu"),
        rerank=lambda query, passages: received.extend(passages) or [.8] * len(passages))


def test_pending_semantic_tails_reach_reranker_without_borrowing_other_attachments(pending_corpus):
    db, _, embedding, index, parts = pending_corpus
    service = FullEvidenceSearch(db, embedding_service=embedding, vector_index=index)
    received = []
    service.reranker_service = capturing_reranker(received)
    hits = service.search(SearchQuery(QUERY, scope="chat_files", evidence="chat", infer_dates=False))
    target = next(hit for hit in hits if hit.attachment_id == parts["target"][0]["attachment_id"])
    matching = [text for text in received if BODY_A in text]
    assert matching and all(BODY_B in text for text in matching)
    assert all("普通家具采购明细" not in text for text in matching)
    assert any("普通家具采购明细" in text and BODY_B not in text for text in received)
    assert all(part.chat_source.attachment_id == target.attachment_id for part in target.supporting_evidence)
    assert target.reranker_score == .8 and not target.preview_anchor


@pytest.mark.parametrize("kind", ["semantic_evidence", "detail_evidence"])
@pytest.mark.parametrize("source", ["txt", "chat", "ocr_pdf"])
def test_reranker_preserves_actual_selected_semantic_or_refined_text(kind, source):
    text = "资料在交接前需由相关人员逐项核对，办理签收并确认存放位置。\n" * 8 + "末尾关键语义证据。"
    row = dict(name="资料.txt", path="synthetic.txt", source_type=source, chunk_text=text)
    item = dict(chunk_id=1, row=row)
    part = dict(chunk_id=1, row=row)
    if kind == "detail_evidence":
        # Keep the full parent too: paragraph refinement must not erase context.
        item["row"] = {**row, "chunk_text": "完整父块的独立背景。"}
        item[kind] = {**part, "refined": True}
    else:
        item[kind] = (part,)
    before = deepcopy(item)
    received = []
    service = object.__new__(FullEvidenceSearch)
    service.cancel_check = None
    service.reranker_service = capturing_reranker(received)
    assert service._rerank(process_query("查询表述没有字面重合"), [item]) == {1: .8}
    assert text.replace("\n", " ") in received[0]
    if kind == "detail_evidence":
        assert "完整父块的独立背景。" in received[0]
    assert item == before
    assert len(evidence_parts(item)) == (2 if kind == "detail_evidence" else 1)


def test_pure_lexical_joint_evidence_keeps_existing_excerpt_and_labels():
    text = "无关背景。" * 60 + "关键词描述" + "后续背景。" * 60
    query = process_query("关键词")
    row = dict(name="资料.txt", path="synthetic.txt", source_type="txt", chunk_text=text)
    item = dict(chunk_id=1, row=row, joint_evidence=(dict(chunk_id=1, row=row),))
    received = []
    service = object.__new__(FullEvidenceSearch)
    service.cancel_check = None
    service.reranker_service = capturing_reranker(received)
    service._rerank(query, [item])
    assert received == [f"文件名：资料.txt\n文件正文：{make_evidence_snippet(text, query.terms)}"]
