import pytest

from localmind.clause_semantics import coverage_ranks, file_clause_evidence
from localmind.models import SearchQuery
from localmind.query_processing import process_query, semantic_clauses, semantic_clause_sources
from localmind.searcher import SearchCancelled, reciprocal_rank_fusion
from test_clause_semantics import corpus, item  # noqa: F401
from test_search_snapshot import document


QUERY = "聊天里说杭州轮岗交接时要用的材料；文件内容是服务器房间停电后的供电保障"


@pytest.mark.parametrize("text,source", [
    ("内容是服务器断电后如何持续供电", "content"),
    ("文件内容是服务器断电后如何持续供电", "content"),
    ("文档的正文中提到怎样处理断电", "content"),
    ("正文里写的是断电保障方案", "content"),
    ("聊天里说这份留作内部讨论", "chat"),
    ("群聊里提到这是客户确认的版本", "chat"),
    ("聊天记录中要求把这个留给培训", "chat"),
    ("聊天消息里讨论新人交接问题", "chat"),
    ("群消息确认这份可以用于演示", "chat"),
    ("签字认可之后以这一版为准", "all"),
    ("我想找包含聊天记录的文档", "all"),
    ("“内容是服务器故障”的标题", "all"),
    ("关于群聊消息的隐私说明", "all"),
    ("正文排版和内容设计的规范", "all"),
])
def test_only_explicit_source_prefixes_are_routed(text, source):
    assert semantic_clause_sources((text,)) == (source,)


def test_nonconjunctive_query_still_cannot_enable_source_routing():
    for text in ("正文是停电保障，或者聊天里说新人培训",
                 "不要聊天里提到的资料，正文是备用电源计划"):
        assert semantic_clauses(process_query(text)) == ()


def test_content_description_cannot_use_a_chat_only_winner():
    clauses = semantic_clauses(process_query(QUERY))
    items = [item(1, 10, "chat"), item(2, 10), item(3, 11, "chat")]
    result = file_clause_evidence(items, {1: (.9, .99), 2: (.1, .6), 3: (.99, .99)}, clauses)
    assert result[10][0] == pytest.approx(2 * .9 * .6 / 1.5)
    assert [part["chunk_id"] for part in result[10][1]] == [1, 2]
    assert result[11] == (0, ())


def test_explicit_chat_description_does_not_use_file_body_or_stale_links():
    clauses = ("正文是备用电池的维护计划", "聊天里说下周给新人演示")
    items = [item(1, 10), item(2, 10, "chat", stale=True), item(3, 11)]
    result = file_clause_evidence(items, {1: (.9, 1), 2: (.1, 1), 3: (1, 1)}, clauses)
    assert result[10] == result[11] == (0, ())


def test_coverage_rank_counts_targets_not_fragments_and_shares_ties():
    assert coverage_ranks({7: (.8, ()), 8: (.8, ()), 9: (.5, ()), 10: (0, ())}) == {7: 1, 8: 1, 9: 3}
    assert coverage_ranks({}) == {}


def test_source_routes_are_scoped_cached_and_explain_actual_fragments(corpus, monkeypatch):
    db, service, _, index, fid, _, _, body, chat = corpus
    with db.conn:
        db.conn.execute("UPDATE chat_sources SET conversation='专属交接群' WHERE source_key='target'")
    service.enable_clause_semantics = True
    calls = []
    search = index.search
    def traced(*args, **kwargs):
        calls.append(kwargs.get("allowed_ids"))
        return search(*args, **kwargs)
    monkeypatch.setattr(index, "search", traced)
    result = service.search(SearchQuery(QUERY, scope="chat_files", conversation="专属交接群"))
    assert result[0].file_id == fid
    assert calls[1] == {chat} and calls[2] == {body}
    assert all(part.source_type == "chat" for part in result[0].supporting_evidence if part.query_clause.startswith("聊天"))
    assert all(part.source_type != "chat" for part in result[0].supporting_evidence if part.query_clause.startswith("文件内容"))
    assert "聊天 / 正文" in service.last_diagnostics.clause_status
    cached = service._source_ids
    service.search(SearchQuery(QUERY, scope="chat_files", conversation="专属交接群"))
    assert service._source_ids is cached and index.rebuilds == 1
    _, new_id, _ = document(db, db.db_path.parent / "new.txt", "新资料", (.1, .1, .8, 0))
    service.search(SearchQuery(QUERY, scope="chat_files", conversation="专属交接群"))
    assert service._source_ids is not cached and new_id in service._source_ids["content"]
    assert new_id not in calls[-1]


def test_fusion_uses_the_updated_semantic_target_rank(corpus, monkeypatch):
    _, service, _, _, _, _, _, _, _ = corpus
    service.enable_clause_semantics = True
    recorded = []
    original = service._rank_candidates
    def capture(batch):
        recorded.extend(dict(item) for item in batch.items)
        return original(batch)
    monkeypatch.setattr(service, "_rank_candidates", capture)
    service.search(SearchQuery(QUERY, scope="chat_files"))
    by_file = {}
    for candidate in recorded:
        by_file.setdefault(candidate["row"]["file_id"], []).append(candidate)
    ranks = coverage_ranks({fid: (group[0]["semantic"], ()) for fid, group in by_file.items()})
    for fid, group in by_file.items():
        keyword = min((item["keyword_rank"] for item in group if item["keyword_rank"] is not None), default=None)
        assert all(item["fused"] == pytest.approx(reciprocal_rank_fusion(keyword, ranks.get(fid))) for item in group)


def test_cancelled_source_cache_is_not_published(corpus, monkeypatch):
    _, service, _, _, _, _, _, _, _ = corpus
    def cancel():
        raise SearchCancelled()
    monkeypatch.setattr(service, "_check_cancelled", cancel)
    with pytest.raises(SearchCancelled):
        service._clause_source_ids("content")
    assert service._source_ids == {}
