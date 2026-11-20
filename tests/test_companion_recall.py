import sqlite3
from types import SimpleNamespace

import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.models import SearchQuery, TextChunk
from localmind.query_processing import process_query
from localmind.companion_recall import CompanionRecall, CompanionRequest, recall_companions
from localmind import companion_recall as recall_module
from localmind import searcher as search_module
from test_weflow_v04 import db, index_file, ingest, searcher


def crowded_corpus(db, tmp_path):
    for i in range(220):
        index_file(db, tmp_path / f"{i:03d}" / "doc.txt", "供应商采购")
    decoy = index_file(db, tmp_path / "220" / "doc.txt", "仓库库存")
    target = index_file(db, tmp_path / "221" / "doc.txt", "供应商采购")
    for file in (decoy, target):
        ingest(db, [ChatMessage(str(i), "苏州审计组现场核验", attachments=[
            Attachment(file["name"], path=file["path"])]) for i in range(8)],
            source=file["path"])
    return target, decoy


def test_crowded_keyword_candidates_recover_missing_content(db, tmp_path, monkeypatch, record_testsuite_property):
    target, decoy = crowded_corpus(db, tmp_path)
    service = searcher(db, mode="keyword")
    original = service._extend_joint_candidates
    monkeypatch.setattr(service, "_extend_joint_candidates", lambda *args, **kwargs: CompanionRecall())
    before = service.search(SearchQuery("苏州审计组现场核验供应商采购", scope="all", limit=20))
    assert before[0].file_id != target["id"]
    record_testsuite_property("companion_baseline_target_rank", next((i for i, r in enumerate(before, 1) if r.file_id == target["id"]), None))
    monkeypatch.setattr(service, "_extend_joint_candidates", original)
    results = service.search(SearchQuery("苏州审计组现场核验供应商采购", scope="all", limit=20))
    record_testsuite_property("companion_target_rank", next((i for i, r in enumerate(results, 1) if r.file_id == target["id"]), None))
    raw = service._keyword_candidates(process_query("苏州审计组现场核验供应商采购"), 160)
    body_id = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (target["id"],)).fetchone()[0]
    assert body_id not in raw
    assert target["id"] in service.last_candidate_file_ids["keyword"]
    assert results[0].file_id == target["id"]
    assert results[0].supporting_evidence
    assert "供应商" in results[0].supporting_evidence[0].snippet
    assert service.last_diagnostics.companion_candidates == 1
    assert service.last_diagnostics.companion_checked_files == 2
    assert decoy["id"] in service.last_candidate_file_ids["keyword"]
    assert all(not r.supporting_evidence for r in results if r.file_id != target["id"])


def test_content_lookup_is_literal_scoped_and_ranked_before_limit(db, tmp_path):
    file = index_file(db, tmp_path / "target.txt", "codeX 500")
    ids = db.insert_chunks(file["id"], [
        TextChunk(chunk_text="code_", source_type="txt"),
        TextChunk(chunk_text="code_ 50%", source_type="txt"),
        TextChunk(chunk_text="50%", source_type="txt"),
    ])
    other = index_file(db, tmp_path / "other.txt", "code_ 50%")
    request = [CompanionRequest(file["id"], "content", ("code_", "50%"))]
    before = db.conn.total_changes
    report = recall_companions(db, request)
    assert list(report.chunk_targets) == [ids[1], ids[0]]
    assert set(report.chunk_targets.values()) == {file["id"]}
    assert other["id"] not in report.chunk_targets.values()
    db.set_search_scope({ids[2]})
    db.conn.execute("PRAGMA query_only=ON")
    changes = db.conn.total_changes
    report = recall_companions(db, request, restricted=True)
    assert list(report.chunk_targets) == [ids[2]] and report.checked_files == 1
    assert db.conn.total_changes == changes and changes >= before


def test_missing_chat_uses_allowed_source_and_keeps_service_index(db, tmp_path, monkeypatch):
    file = index_file(db, tmp_path / "target.txt", "采购订单")
    for source, sender, date in (("old", "Alice", "2026-09-01"), ("new", "Bob", "2026-10-01")):
        ingest(db, [ChatMessage(source, "苏州审计", sender=sender, sent_at=date, attachments=[
            Attachment(file["name"], path=file["path"])])], source=source, conversation=source)
    body_id = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (file["id"],)).fetchone()[0]
    service = searcher(db, mode="keyword")
    index = service.vector_index._faiss_index
    monkeypatch.setattr(service, "_keyword_candidates", lambda *args: {body_id: 1})
    query = SearchQuery("采购订单 苏州审计", scope="chat_files", conversation="old", sender="Alice",
                        file_type="txt", path_contains="target", modified_before="2026-10-01")
    result = service.search(query)[0]
    assert result.supporting_evidence and result.file_id == file["id"]
    assert "Alice" in result.chat_context and "Bob" not in result.chat_context
    assert service.last_diagnostics.companion_candidates == 1
    assert service.last_diagnostics.companion_checked_files == 1
    assert service.vector_index._faiss_index is index
    assert service.last_diagnostics.semantic_candidates == 0
    assert service.last_diagnostics.timings_ms["companion"] >= 0
    again = service.search(query)[0]
    assert again.score == result.score and again.chat_message_id == result.chat_message_id


def test_chat_lookup_rejects_stale_link_and_cannot_return_other_file(db, tmp_path):
    first = index_file(db, tmp_path / "first.txt", "body")
    second = index_file(db, tmp_path / "second.txt", "body")
    for file in (first, second):
        ingest(db, [ChatMessage("one", "unique clue", attachments=[
            Attachment(file["name"], path=file["path"])])], source=file["name"])
    request = [CompanionRequest(first["id"], "chat", ("unique",))]
    report = recall_companions(db, request)
    assert len(report.chunk_targets) == 1 and set(report.chunk_targets.values()) == {first["id"]}
    db.conn.execute("UPDATE files SET file_hash='changed' WHERE id=?", (first["id"],))
    assert not recall_companions(db, request).chunk_targets


def test_file_and_zero_time_budgets_return_explicit_partial_state(db, tmp_path):
    files = [index_file(db, tmp_path / f"{i}.txt", "needle") for i in range(2)]
    requests = [CompanionRequest(f["id"], "content", ("needle",)) for f in files]
    report = recall_companions(db, requests, max_files=1)
    assert report.requested_files == 2 and report.checked_files == 1 and report.file_limited
    assert set(report.chunk_targets.values()) == {files[0]["id"]}
    report = recall_companions(db, requests, budget_ms=0)
    assert report.time_limited and report.checked_files == 0 and not report.chunk_targets
    assert not recall_companions(db, []).time_limited


def test_sql_interruption_keeps_completed_files_and_restores_connection(db, tmp_path, monkeypatch):
    first = index_file(db, tmp_path / "small.txt", "needle")
    large = index_file(db, tmp_path / "large.txt", "needle")
    db.insert_chunks(large["id"], [TextChunk(chunk_text=f"needle {i}", source_type="txt") for i in range(2000)])
    clock = iter(i / 1000 for i in range(1000))
    monkeypatch.setattr(recall_module, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    requests = [CompanionRequest(f["id"], "content", ("needle",)) for f in (first, large)]
    db.conn.execute("PRAGMA query_only=ON")
    before = db.conn.total_changes
    report = recall_companions(db, requests, budget_ms=3)
    assert report.time_limited and report.checked_files == 1
    assert set(report.chunk_targets.values()) == {first["id"]}
    assert db.conn.total_changes == before and not db.conn.in_transaction
    assert db.conn.execute("""
        WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM nums WHERE n<10000)
        SELECT sum(n) FROM nums""").fetchone()[0] == 50005000


def test_sql_errors_are_not_silently_reported_as_budget_timeouts(db, tmp_path):
    file = index_file(db, tmp_path / "file.txt", "needle")
    request = [CompanionRequest(file["id"], "content", ("needle",))]
    with pytest.raises(sqlite3.OperationalError, match="search_scope"):
        recall_companions(db, request, restricted=True)
    assert recall_companions(db, request).checked_files == 1


@pytest.mark.parametrize("options", [
    {"max_files": 0}, {"max_files": 65}, {"per_file": 0}, {"per_file": 5},
    {"budget_ms": -1}, {"budget_ms": float("inf")}, {"budget_ms": float("nan")},
])
def test_invalid_budgets_are_rejected(db, options):
    with pytest.raises(ValueError):
        recall_companions(db, [], **options)


@pytest.mark.parametrize("mode,scope,evidence", [
    ("semantic", "chat_files", "all"), ("keyword", "messages", "all"),
    ("keyword", "files", "all"), ("keyword", "chat_files", "content"),
    ("keyword", "chat_files", "chat"),
])
def test_other_search_modes_never_call_companion_lookup(db, tmp_path, monkeypatch, mode, scope, evidence):
    file = index_file(db, tmp_path / "file.txt", "采购订单")
    ingest(db, [ChatMessage("one", "苏州审计", attachments=[Attachment(file["name"], path=file["path"])])])
    monkeypatch.setattr(search_module, "recall_companions", lambda *args, **kwargs: pytest.fail("Unexpected lookup"))
    service = searcher(db, mode=mode)
    service.search(SearchQuery("采购订单 苏州审计", scope=scope, evidence=evidence))
    assert service.last_diagnostics.companion_candidates == 0
    assert service.last_diagnostics.companion_status == ""


def test_extra_fragment_does_not_get_invented_vector_or_rrf_score(db, tmp_path):
    file = index_file(db, tmp_path / "file.txt", "采购订单")
    cid = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (file["id"],)).fetchone()[0]
    row = dict(db.fetch_chunks_by_ids([cid])[cid])
    item = searcher(db, mode="hybrid")._score_row(row["id"], row, process_query("采购订单 苏州审计"))
    assert item["semantic"] == 0 and item["fused"] == 0


def test_service_reports_file_limit_without_losing_original_candidates(db, tmp_path, monkeypatch):
    body_ids = []
    for i in range(35):
        file = index_file(db, tmp_path / f"{i}.txt", "采购订单")
        body_ids.append(db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (file["id"],)).fetchone()[0])
        ingest(db, [ChatMessage("one", "苏州审计", attachments=[
            Attachment(file["name"], path=file["path"])])], source=str(i))
    service = searcher(db, mode="keyword")
    monkeypatch.setattr(service, "_keyword_candidates", lambda *args: dict(zip(body_ids, range(1, 36))))
    results = service.search(SearchQuery("采购订单 苏州审计", scope="chat_files", limit=35))
    diagnostics = service.last_diagnostics
    assert len(results) == 35
    assert diagnostics.companion_checked_files == 32 and diagnostics.companion_requested_files == 35
    assert diagnostics.companion_candidates == 32 and "文件上限" in diagnostics.companion_status
    assert sum(bool(r.supporting_evidence) for r in results) == 32


def test_service_budget_timeout_keeps_initial_results(db, tmp_path, monkeypatch):
    file = index_file(db, tmp_path / "file.txt", "采购订单")
    ingest(db, [ChatMessage("one", "苏州审计", attachments=[Attachment(file["name"], path=file["path"])])])
    cid = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (file["id"],)).fetchone()[0]
    service = searcher(db, mode="keyword")
    monkeypatch.setattr(service, "_keyword_candidates", lambda *args: {cid: 1})
    monkeypatch.setattr(search_module, "recall_companions",
                        lambda *args, **kwargs: recall_companions(*args, budget_ms=0, **kwargs))
    results = service.search(SearchQuery("采购订单 苏州审计", scope="chat_files"))
    assert len(results) == 1 and results[0].file_id == file["id"]
    assert not results[0].supporting_evidence
    assert service.last_diagnostics.companion_candidates == 0
    assert "时间预算" in service.last_diagnostics.companion_status


def test_rebind_between_lookup_and_enrichment_cannot_expand_target_set(db, tmp_path, monkeypatch):
    first = index_file(db, tmp_path / "first.txt", "采购订单")
    second = index_file(db, tmp_path / "second.txt", "其他正文")
    ingest(db, [ChatMessage("one", "苏州审计", attachments=[Attachment(first["name"], path=first["path"])])])
    cid = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (first["id"],)).fetchone()[0]
    attachment = ChatStore(db).attachments()[0]
    service = searcher(db, mode="keyword")
    writer = LocalMindDB(db.db_path)
    monkeypatch.setattr(service, "_keyword_candidates", lambda *args: {cid: 1})
    def rebind(*args, **kwargs):
        report = recall_companions(*args, **kwargs)
        ChatStore(writer).bind(attachment["id"], second["id"])
        return report
    monkeypatch.setattr(search_module, "recall_companions", rebind)
    try:
        results = service.search(SearchQuery("采购订单 苏州审计", scope="chat_files"))
    finally:
        writer.close()
    assert [r.file_id for r in results] == [first["id"]]
    # The search retains the original association throughout its read snapshot.
    assert results[0].supporting_evidence
    assert service.last_diagnostics.companion_candidates == 1
    assert not service.last_diagnostics.index_current
    assert ChatStore(db).attachments()[0]["file_id"] == second["id"]


@pytest.mark.parametrize("keep_valid_chat", [True, False])
def test_recent_stale_sources_do_not_make_existing_body_look_missing(db, tmp_path, monkeypatch, keep_valid_chat):
    file = index_file(db, tmp_path / "file.txt", "采购订单")
    for i in range(4):
        ingest(db, [ChatMessage("one", "苏州审计" if i == 0 else "无关内容",
                                sent_at=f"2026-09-{i + 1:02d}", attachments=[
            Attachment(file["name"], path=file["path"])])], source=str(i))
    first = db.conn.execute("SELECT min(id) FROM chat_attachments").fetchone()[0]
    db.conn.execute("UPDATE chat_attachments SET linked_hash='outdated' WHERE id!=?", (first,))
    db.conn.commit()
    service = searcher(db, mode="keyword")
    if not keep_valid_chat:
        body = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (file["id"],)).fetchone()[0]
        stale_chat = db.conn.execute("SELECT chunk_id FROM chat_units WHERE attachment_id!=? LIMIT 1", (first,)).fetchone()[0]
        monkeypatch.setattr(service, "_keyword_candidates", lambda *args: {body: 1, stale_chat: 2})
    results = service.search(SearchQuery("采购订单 苏州审计", scope="chat_files"))
    assert results[0].supporting_evidence
    assert results[0].attachment_id == first
    assert service.last_diagnostics.companion_requested_files == (0 if keep_valid_chat else 1)
    assert service.last_diagnostics.companion_candidates == (0 if keep_valid_chat else 1)
