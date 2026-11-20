from datetime import datetime

import pytest

from localmind.models import SearchQuery
from localmind.query_processing import process_query
from localmind import searcher as search_module
from test_search_evidence import attach
from test_weflow_v04 import db, index_file, searcher, CountingEmbedding


@pytest.mark.parametrize("query", [
    "正文包含去年预算说明",
    "不要去年发的付款材料",
    "去年或今年发送的付款材料",
    "去年预算.xlsx",
    "去年预算.xlsx文件",
    "查找标题为“去年预算”的资料",
    "包含去年收到的发票金额的说明",
    "没有今年发送的合同",
    "上个月之前发送的合同",
    "从去年到九月的付款记录",
    "去年起的付款记录",
    "《去年发来的合同》",
    r"D:\去年发来的合同.txt",
    "今年计划与去年预算的对比",
    "包含上个月这个词的记录",
    "今年以来的付款记录",
])
def test_content_literal_and_ambiguous_dates_do_not_silently_filter(query):
    processed = process_query(query, now=datetime(2026, 10, 21))
    assert processed.modified_after is None and processed.modified_before is None


def test_body_year_does_not_exclude_a_recently_sent_wechat_file(db, tmp_path, monkeypatch):
    target = index_file(db, tmp_path / "budget.txt", "去年预算说明：星桥项目付款凭证")
    attach(db, target, source="目标群", text="请核对随附的项目材料", sent_at="2026-10-10T00:00:00Z")
    original = search_module.process_query
    monkeypatch.setattr(search_module, "process_query", lambda text, **kw: original(text, now=datetime(2026, 10, 21), **kw))
    results = searcher(db, mode="keyword").search(
        SearchQuery("正文包含去年预算说明", scope="chat_files", evidence="content"))
    assert [result.file_id for result in results] == [target["id"]]


@pytest.mark.parametrize("bounds", [
    {"modified_after": "2026-10-01"},
    {"modified_before": "2025-01-01"},
])
def test_explicit_partial_bounds_are_not_completed_from_the_query(db, monkeypatch, bounds):
    original = search_module.process_query
    monkeypatch.setattr(search_module, "process_query", lambda text, **kw: original(text, now=datetime(2026, 10, 21), **kw))
    service = searcher(db, mode="keyword")
    service.search(SearchQuery("去年付款凭证", **bounds))
    assert service.last_diagnostics.modified_after == bounds.get("modified_after")
    assert service.last_diagnostics.modified_before == bounds.get("modified_before")


@pytest.mark.parametrize("query,after,before", [
    ("上个月付款凭证", "2026-09-01", "2026-10-01"),
    ("请帮我找上个月的付款凭证", "2026-09-01", "2026-10-01"),
    ("同事上个月发来的付款凭证", "2026-09-01", "2026-10-01"),
    ("去年修改的预算.xlsx", "2025-01-01", "2026-01-01"),
    ("李工今年提供的图纸", "2026-01-01", "2027-01-01"),
    ("本月合同", "2026-10-01", "2026-11-01"),
    ("这个月保存的照片", "2026-10-01", "2026-11-01"),
    ("上个月发来的“付款凭证”", "2026-09-01", "2026-10-01"),
])
def test_simple_leading_dates_and_explicit_actions_keep_relative_ranges(query, after, before):
    value = process_query(query, now=datetime(2026, 10, 21))
    assert value.modified_after == after + "T00:00:00"
    assert value.modified_before == (before + "T00:00:00" if before else None)


def test_january_rollover_and_disabled_inference_preserve_literal_time_terms():
    value = process_query("上个月付款凭证", now=datetime(2026, 1, 2))
    assert value.modified_after == "2025-12-01T00:00:00"
    assert value.modified_before == "2026-01-01T00:00:00"
    literal = process_query("去年预算", infer_dates=False)
    assert literal.modified_after is None and literal.modified_before is None
    assert "去年" in literal.terms
    assert "去年" in process_query("正文包含去年预算说明").terms


@pytest.mark.parametrize("now,query,after,before", [
    ("2026-12-31T23:59:00+08:00", "本月合同", "2026-12-01T00:00:00+08:00", "2027-01-01T00:00:00+08:00"),
    ("2024-02-29T10:00:00+08:00", "今年合同", "2024-01-01T00:00:00+08:00", "2025-01-01T00:00:00+08:00"),
    ("2026-01-01T00:01:00+08:00", "去年合同", "2025-01-01T00:00:00+08:00", "2026-01-01T00:00:00+08:00"),
])
def test_calendar_periods_are_bounded_and_preserve_explicit_clock_timezone(now, query, after, before):
    value = process_query(query, now=datetime.fromisoformat(now))
    assert (value.modified_after, value.modified_before) == (after, before)


def test_current_month_excludes_future_sources_and_reuses_frozen_range(db, tmp_path, monkeypatch):
    from dataclasses import replace

    files = [index_file(db, tmp_path / f"month-{month}.txt", "验收合同") for month in (9, 10, 11)]
    for file, month in zip(files, (9, 10, 11)):
        attach(db, file, source=f"{month}月", text="验收合同", sent_at=f"2026-{month:02d}-12T04:00:00Z")
    now = [datetime(2026, 10, 21)]
    monkeypatch.setattr(search_module, "process_query", lambda text: process_query(text, now=now[0]))
    service = searcher(db, mode="keyword")
    results = service.search(SearchQuery("本月验收合同", scope="chat_files"))
    assert [result.file_id for result in results] == [files[1]["id"]]
    frozen = results[0].source_query
    now[0] = datetime(2026, 11, 2)
    assert [result.file_id for result in service.search(
        replace(frozen, text="本月验收合同", limit=50))] == [files[1]["id"]]
    assert [result.file_id for result in service.search(SearchQuery("本月验收合同", scope="chat_files"))] == [files[2]["id"]]


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
@pytest.mark.parametrize("evidence", ["content", "chat"])
def test_content_dates_reach_both_wechat_retrieval_routes(db, tmp_path, monkeypatch, mode, evidence):
    embed = CountingEmbedding()
    target = index_file(db, tmp_path / "budget.txt", "去年预算说明与付款凭证", embed)
    attach(db, target, source="目标群", text="正文包含去年预算说明", sent_at="2026-10-10T00:00:00Z",
           embedding=embed)
    original = search_module.process_query
    monkeypatch.setattr(search_module, "process_query", lambda text: original(text, now=datetime(2026, 10, 21)))
    service = searcher(db, embed, mode=mode)
    results = service.search(SearchQuery("正文包含去年预算说明", scope="chat_files", evidence=evidence))
    assert [result.file_id for result in results] == [target["id"]]
    assert service.last_diagnostics.modified_after is None
    assert results[0].source_query.modified_after is None
    assert "目标群" in results[0].chat_context


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_all_time_and_explicit_dates_invalidate_scope_without_rebuilding_vectors(db, tmp_path, monkeypatch, mode):
    embed = CountingEmbedding()
    old = index_file(db, tmp_path / "old.txt", "去年付款凭证", embed)
    recent = index_file(db, tmp_path / "recent.txt", "去年付款凭证", embed)
    attach(db, old, source="旧来源", text="去年付款凭证", sent_at="2025-06-10T00:00:00Z", embedding=embed)
    attach(db, recent, source="新来源", text="去年付款凭证", sent_at="2026-10-10T00:00:00Z", embedding=embed)
    original = search_module.process_query
    monkeypatch.setattr(search_module, "process_query",
                        lambda text, **kw: original(text, now=datetime(2026, 10, 21), **kw))
    service = searcher(db, embed, mode=mode)
    vector_index = service.vector_index._faiss_index
    query = SearchQuery("去年付款凭证", scope="chat_files")
    assert [result.file_id for result in service.search(query)] == [old["id"]]
    assert query.modified_after is None and query.modified_before is None
    query.infer_dates = False
    assert {result.file_id for result in service.search(query)} == {old["id"], recent["id"]}
    assert service.last_diagnostics.modified_after is None and service.last_diagnostics.modified_before is None
    for inference in (False, True):
        query.infer_dates = inference
        query.modified_after = "2026-01-01"
        results = service.search(query)
        assert [result.file_id for result in results] == [recent["id"]]
        assert results[0].source_query.modified_before is None
        query.modified_after = None
        query.modified_before = "2026-01-01"
        assert [result.file_id for result in service.search(query)] == [old["id"]]
        query.modified_before = None
    assert service.vector_index._faiss_index is vector_index


def test_cli_can_disable_date_inference(db, tmp_path, monkeypatch, capsys):
    from localmind import cli

    target = index_file(db, tmp_path / "cli-time.txt", "去年预算说明与付款凭证")
    attach(db, target, source="测试群", text="去年预算", sent_at="2026-10-10T00:00:00Z")
    monkeypatch.setattr("sys.argv", ["localmind-index", "--db", str(db.db_path), "search", "去年预算",
                                   "--mode", "keyword", "--scope", "chat_files",
                                   "--no-rerank", "--no-date-inference"])
    cli.main()
    assert target["name"] in capsys.readouterr().out
