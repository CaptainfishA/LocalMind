from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.clause_semantics import evidence_key, evidence_parts
from localmind.db import LocalMindDB
from localmind.desktop.detail_text import evidence_text
from localmind.models import SearchQuery
from localmind.paragraph_refinement import ParagraphRefiner, body_fragments
from localmind.preview_worker import preview_file
from localmind.reranker import RerankerService
from localmind.scanner import build_file_record
from localmind.searcher import SearchCancelled, SearchService
from test_search_snapshot import CountedIndex, FixedQuery, document


QUERY = "外面的客人要进办公楼，需要办哪些手续"
TARGET = "来访人员须出示身份证件，前台核验预约记录后发放临时通行证。"
BACKGROUND = "常规背景信息与归档说明。" * 10
DECOY = "住宿报销需要发票、酒店费用明细以及部门负责人审批通过的申请记录。"


class DetailEmbedding(FixedQuery):
    device = "cpu"

    def __init__(self, db):
        self.db = db
        self.calls = []
        self.on_encode = lambda: None

    def encode_queries(self, texts):
        return np.array([[1., 0.]] * len(texts), dtype="float32")

    def encode_documents(self, texts):
        assert not self.db.conn.in_transaction
        self.calls.append(tuple(texts))
        self.on_encode()
        return np.array([[.98, .2] if text == TARGET else [.1, .99] for text in texts], dtype="float32")


@pytest.fixture
def detail_corpus(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    path = tmp_path / "target.txt"
    fid, cid, record = document(db, path, BACKGROUND + "\n" + TARGET, (.25, .97))
    other, _, _ = document(db, tmp_path / "decoy.txt", DECOY, (.6, .8))
    for file_id, file_path in ((fid, path), (other, tmp_path / "decoy.txt")):
        actual = build_file_record(file_path)
        with db.conn:
            db.conn.execute("UPDATE files SET modified_at=?, file_hash=NULL WHERE id=?", (actual.modified_at, file_id))
    for key, name in (("target", "访客组"), ("decoy", "财务组")):
        file = tmp_path / f"{key}.txt"
        ChatStore(db).ingest(platform="wechat", account="synthetic", conversation=name, source_key=key,
            messages=[ChatMessage("one", "已归档，请核对资料。", attachments=[Attachment(file.name, str(file))])])
    embedding = DetailEmbedding(db)
    index = CountedIndex()
    service = SearchService(db, embedding_service=embedding, vector_index=index,
                            reranker_service=RerankerService(enabled=False))
    writer = LocalMindDB(db.db_path)
    query = SearchQuery(QUERY, scope="chat_files", evidence="content", infer_dates=False)
    try:
        yield db, writer, service, embedding, index, fid, cid, record, other, query
    finally:
        writer.close()
        db.close()


def test_literal_fragments_keep_offsets_deduplicate_and_bound_input():
    text = "\r\n  " + TARGET + "  \r\n" + BACKGROUND + "\n" + TARGET
    parts = body_fragments(text)
    assert len(parts) == 2 and parts[0].text == TARGET
    assert all(text[p.start:p.end] == p.text and 24 <= len(p.text) <= 384 for p in parts)
    assert body_fragments(TARGET) == ()
    assert body_fragments("X" * 4097) == ()
    assert TARGET in [part.text for part in body_fragments(BACKGROUND * 5 + TARGET)]


def test_refined_scores_evidence_preview_and_repeated_query_share_one_snapshot(detail_corpus):
    db, _, service, embedding, index, fid, cid, record, other, query = detail_corpus
    before = [tuple(row) for row in db.conn.execute("SELECT * FROM vectors ORDER BY chunk_id")]
    service.enable_paragraph_refinement = False
    assert service.search(query)[0].file_id == other
    service.enable_paragraph_refinement = True
    hits = service.search(query)
    assert hits[0].file_id == fid and hits[0].chunk_id == cid
    assert hits[0].semantic_score > .9 and TARGET in hits[0].snippet
    assert hits[0].preview_anchor == TARGET
    assert hits[0].supporting_evidence[0].refined
    assert hits[0].supporting_evidence[0].chunk_id == cid
    collapsed, clipped = evidence_text(hits[0])
    expanded, _ = evidence_text(hits[0], expanded=True)
    assert TARGET in collapsed and BACKGROUND not in collapsed and clipped
    assert "匹配片段" in collapsed and "上下文" in expanded and BACKGROUND in "".join(expanded.split())
    assert preview_file(record.path, anchor=hits[0].preview_anchor)["focus_status"] == "matched"
    first_calls = list(embedding.calls)
    repeated = service.search(query)
    assert [hit.file_id for hit in repeated] == [hit.file_id for hit in hits]
    assert embedding.calls == first_calls and service.last_diagnostics.refinement_cache_hits > 0
    assert service.last_diagnostics.refinement_encoded == 0 and index.rebuilds == 1
    assert before == [tuple(row) for row in db.conn.execute("SELECT * FROM vectors ORDER BY chunk_id")]


def test_expanded_refined_context_keeps_long_original_without_summary_ellipsis(detail_corpus):
    db, _, service, _, _, fid, cid, *_rest, query = detail_corpus
    parent = BACKGROUND * 4 + "\n" + TARGET
    with db.conn:
        db.conn.execute("UPDATE chunks SET chunk_text=? WHERE id=?", (parent, cid))
    hit = next(hit for hit in service.search(query) if hit.file_id == fid)
    assert hit.supporting_evidence[0].snippet == TARGET
    assert any(part.snippet == parent for part in hit.supporting_evidence if not part.refined)
    expanded, _ = evidence_text(hit, expanded=True)
    assert BACKGROUND * 4 + TARGET in "".join(expanded.split())


@pytest.mark.parametrize("mode", ["keyword", "semantic"])
def test_other_modes_keep_order_and_do_not_encode_documents(detail_corpus, mode):
    _, _, service, embedding, _, _, _, _, _, query = detail_corpus
    service.retrieval_mode = mode
    service.enable_paragraph_refinement = False
    before = service.search(query)
    service.enable_paragraph_refinement = True
    after = service.search(query)
    assert [(r.file_id, r.score) for r in after] == [(r.file_id, r.score) for r in before]
    assert not embedding.calls


@pytest.mark.parametrize("query", [SearchQuery("来访人员"), SearchQuery(QUERY, scope="chat_files", evidence="chat"),
    SearchQuery(QUERY, scope="messages"), SearchQuery(QUERY + " FK-864"),
    SearchQuery("聊天里说归档后交给行政；文件内容是外面的客人进办公楼需要办理手续")])
def test_literal_identifier_chat_and_clause_paths_do_not_add_inference(detail_corpus, query):
    _, _, service, embedding, *_ = detail_corpus
    service.search(query)
    assert not embedding.calls


def test_filters_restrict_fragment_encoding_to_eligible_files(detail_corpus):
    _, _, service, embedding, _, fid, _, _, _, query = detail_corpus
    hits = service.search(replace(query, conversation="访客组", file_type="txt", path_contains="target"))
    assert [hit.file_id for hit in hits] == [fid]
    assert embedding.calls and all(DECOY not in text for batch in embedding.calls for text in batch)


def test_outer_transaction_skips_refinement_and_remains_owned_by_caller(detail_corpus):
    db, _, service, embedding, *_rest, query = detail_corpus
    db.conn.execute("BEGIN")
    try:
        assert service.search(query)
        assert db.conn.in_transaction and not embedding.calls
    finally:
        db.conn.rollback()


@pytest.mark.parametrize("fault", ["exception", "nan", "shape", "zero", "model_change"])
def test_bad_embeddings_fall_back_without_publishing_cache_or_partial_ranking(detail_corpus, monkeypatch, fault):
    _, _, service, embedding, _, _, _, _, _, query = detail_corpus
    service.enable_paragraph_refinement = False
    before = service.search(query)
    service.enable_paragraph_refinement = True
    def bad(texts):
        if fault == "exception":
            raise ValueError("private model error detail")
        if fault == "model_change":
            embedding._backend = object()
            return np.ones((len(texts), 2), dtype="float32")
        return np.full((len(texts), 3 if fault == "shape" else 2), np.nan if fault == "nan" else 0)
    monkeypatch.setattr(embedding, "encode_documents", bad)
    after = service.search(query)
    assert [(r.file_id, r.score) for r in after] == [(r.file_id, r.score) for r in before]
    assert "保留基础排序" in service.last_diagnostics.refinement_status
    assert "private" not in service.last_diagnostics.refinement_status and not service._paragraph_refiner.cache


@pytest.mark.parametrize("stage", ["encoding", "ranking", "file_check"])
def test_cancel_never_publishes_new_cache_or_diagnostics(detail_corpus, monkeypatch, stage):
    import localmind.searcher as module
    _, _, service, embedding, _, _, _, _, _, query = detail_corpus
    previous = service.last_diagnostics
    candidates = service.last_candidate_file_ids
    stop = Event()
    service.should_cancel = stop.is_set
    if stage == "encoding":
        embedding.on_encode = stop.set
    else:
        owner, method = (service, "_rank_candidates") if stage == "ranking" else (module, "annotate_results")
        original = getattr(owner, method)
        def cancel_after(*args):
            result = original(*args)
            stop.set()
            return result
        monkeypatch.setattr(owner, method, cancel_after)
    with pytest.raises(SearchCancelled):
        service.search(query)
    assert service.last_diagnostics is previous and service.last_candidate_file_ids is candidates
    assert not service._paragraph_refiner.cache


def test_writer_during_detail_inference_does_not_change_displayed_evidence_or_cache(detail_corpus):
    db, writer, service, embedding, _, fid, cid, _, _, query = detail_corpus
    def update():
        with writer.conn:
            writer.conn.execute("UPDATE chunks SET chunk_text='新的索引正文' WHERE id=?", (cid,))
    embedding.on_encode = update
    hits = service.search(query)
    target = next(hit for hit in hits if hit.file_id == fid)
    assert TARGET in target.snippet and target.preview_anchor == TARGET
    assert not service.last_diagnostics.index_current and not service._paragraph_refiner.cache
    embedding.on_encode = lambda: None
    hits = service.search(query)
    assert all(TARGET not in hit.snippet for hit in hits)
    assert service.last_diagnostics.index_current


def test_index_revision_clears_old_text_cache_without_faiss_rebuild(detail_corpus):
    _, writer, service, _, index, _, cid, _, _, query = detail_corpus
    service.search(query)
    assert TARGET in service._paragraph_refiner.cache
    rebuilds = index.rebuilds
    with writer.conn:
        writer.conn.execute("UPDATE chunks SET chunk_text='新资料' WHERE id=?", (cid,))
    service.search(query)
    assert TARGET not in service._paragraph_refiner.cache and index.rebuilds == rebuilds


def test_cleared_index_has_no_cached_text_or_results(detail_corpus):
    _, writer, service, _, index, *_rest, query = detail_corpus
    service.search(query)
    assert service._paragraph_refiner.cache
    writer.clear_index()
    assert service.search(query) == []
    assert not service._paragraph_refiner.cache and not len(index.ids)


def test_backend_replacement_invalidates_same_model_cache(detail_corpus):
    _, _, service, embedding, _, *_rest, query = detail_corpus
    embedding._backend = object()
    service.search(query)
    calls = len(embedding.calls)
    embedding._backend = object()
    service.search(query)
    assert len(embedding.calls) > calls and service.last_diagnostics.refinement_cache_hits == 0


def test_worker_notification_releases_paragraph_text_without_rebuilding_faiss(detail_corpus, tmp_path):
    from localmind.ui import SearchWorker
    db, _, service, _, index, *_rest, query = detail_corpus
    service.search(query)
    worker = SearchWorker(str(db.db_path), str(tmp_path / "cache"), "cpu", "cpu")
    worker.service = service
    rebuilds = index.rebuilds
    worker.mark_index_dirty()
    assert worker.index_dirty and not service._paragraph_refiner.cache and index.rebuilds == rebuilds


def test_refined_rerank_input_and_identity_include_actual_passage(detail_corpus):
    from types import SimpleNamespace
    _, _, service, _, _, fid, *_rest, query = detail_corpus
    received = []
    service.reranker_service = SimpleNamespace(enabled=True, recommended_top_k=30, model_name="synthetic", load_error=None,
        runtime=SimpleNamespace(name="cpu"), rerank=lambda query, passages: received.extend(passages) or [.8] * len(passages))
    hit = next(hit for hit in service.search(query) if hit.file_id == fid)
    assert received and any(TARGET in text and BACKGROUND in text for text in received)
    assert hit.supporting_evidence[0].refined
    value = dict(chunk_id=1, row={"chunk_text": BACKGROUND}, detail_evidence=dict(chunk_id=1, row={"chunk_text": TARGET}))
    assert evidence_parts(value)[0]["row"]["chunk_text"] == TARGET
    other = {**value, "detail_evidence": {"chunk_id": 1, "row": {"chunk_text": DECOY}}}
    assert evidence_key(value) != evidence_key(other)


def items(count=30):
    return [dict(chunk_id=i, preliminary=1 - i / 100, row=dict(file_id=i, source_type="txt",
        chunk_text=f"第{i:03d}条，资料交接需要逐项核对名称、签收人员、接收时间和归档编号。\n"
                   f"第{i:03d}项，移交完成后应保留审核人员签字、确认日期与后续核验的记录。")) for i in range(count)]


@pytest.mark.parametrize("device,limit,batch_size", [("cpu", 8, 4), ("cuda", 24, 8)])
def test_budget_selection_batches_cache_and_lru_are_bounded(device, limit, batch_size):
    refiner, calls = ParagraphRefiner(), []
    def encode(texts):
        calls.append(tuple(texts))
        return np.ones((len(texts), 2))
    plan = refiner.prepare(items(), np.ones(2), identity=("model", 1), encoder=encode, device=device)
    assert plan.selected == limit and plan.encoded == limit and all(len(batch) <= batch_size for batch in calls)
    assert not refiner.cache
    refiner.commit(plan)
    repeated = refiner.prepare(items(), np.ones(2), identity=("model", 1), encoder=encode, device=device)
    assert repeated.used == plan.used and repeated.encoded == 0 and repeated.cache_hits == limit
    refiner.MAX_CACHE = 3
    refiner.commit(repeated)
    assert len(refiner.cache) == 3
    changed = refiner.prepare(items(1), np.ones(2), identity=("model", 2), encoder=encode)
    assert changed.encoded == 2 and not refiner.cache


def test_deadline_keeps_partial_vectors_staged_and_returns_no_partial_scores(monkeypatch):
    import localmind.paragraph_refinement as module
    clock = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    refiner = ParagraphRefiner()
    def encode(texts):
        clock[0] = .3
        return np.ones((len(texts), 2))
    plan = refiner.prepare(items(), np.ones(2), identity=("model", 1), encoder=encode)
    assert not plan.complete and plan.encoded == 4 and not plan.matches and not refiner.cache
    refiner.commit(plan)
    assert len(refiner.cache) == 4


def test_zero_time_budget_keeps_search_baseline(detail_corpus):
    _, _, service, embedding, _, _, _, _, _, query = detail_corpus
    service.enable_paragraph_refinement = False
    before = service.search(query)
    service.enable_paragraph_refinement = True
    service._paragraph_refiner.BUDGET_SECONDS = 0
    after = service.search(query)
    assert [(r.file_id, r.score) for r in after] == [(r.file_id, r.score) for r in before]
    assert not embedding.calls and "达到预算" in service.last_diagnostics.refinement_status
