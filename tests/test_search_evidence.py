from datetime import datetime
from unittest.mock import Mock

import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore, context_label
from localmind.models import SearchQuery
from localmind import searcher as search_module
from test_weflow_v04 import db, index_file, ingest, searcher, CountingEmbedding


def attach(db, file, *, source, text, sent_at="2026-09-12T04:00:00Z", embedding=None):
    return ingest(db, [ChatMessage(source, text, "同事", sent_at,
                                  [Attachment(file["name"], path=file["path"])])],
                  source=source, conversation=source, embedding=embedding)


def test_separate_chat_and_content_evidence_and_reuse(db, tmp_path):
    embed = CountingEmbedding()
    first = index_file(db, tmp_path / "first.txt", "星桥计划的正文材料", embed)
    second = index_file(db, tmp_path / "second.txt", "日常维护的正文材料", embed)
    attach(db, first, source="甲群", text="普通交付附件", embedding=embed)
    attach(db, second, source="乙群", text="这是星桥计划的讨论材料", embedding=embed)
    service = searcher(db, embed, mode="keyword")
    vector_index = service.vector_index._faiss_index
    query = SearchQuery("星桥计划", scope="chat_files", evidence="content")
    results = service.search(query)
    assert [r.file_id for r in results] == [first["id"]]
    assert results[0].source_type != "chat"
    query.evidence = "chat"
    results = service.search(query)
    assert [r.file_id for r in results] == [second["id"]]
    assert results[0].source_type == "chat" and "乙群" in results[0].chat_context
    query.evidence = "all"
    assert {r.file_id for r in service.search(query)} == {first["id"], second["id"]}
    service.retrieval_mode = "semantic"
    query.evidence = "content"
    assert all(r.source_type != "chat" for r in service.search(query))
    query.evidence = "chat"
    assert all(r.source_type == "chat" for r in service.search(query))
    assert service.vector_index._faiss_index is vector_index


def test_date_filters_before_candidates_and_uses_attachment_time(db, tmp_path):
    embed = CountingEmbedding()
    target = index_file(db, tmp_path / "target.txt", "共同验收凭证", embed)
    messages = [ChatMessage(f"noise-{i}", "共同验收凭证", sent_at="2026-10-01T00:00:00Z",
                            attachments=[Attachment(f"noise-{i}.txt")]) for i in range(180)]
    ingest(db, messages, source="干扰群", embedding=embed)
    attach(db, target, source="目标群", text="共同验收凭证", embedding=embed)
    service = searcher(db, embed)
    query = SearchQuery("共同验收凭证", scope="chat_files", limit=5,
                        modified_after="2026-09-01T00:00:00Z", modified_before="2026-10-01T00:00:00Z")
    for mode in ("keyword", "semantic", "hybrid"):
        service.retrieval_mode = mode
        results = service.search(query)
        assert len(results) == 1 and results[0].file_id == target["id"]
        assert "目标群" in results[0].chat_context
        assert service.last_diagnostics.keyword_candidates <= 2
        assert service.last_diagnostics.semantic_candidates <= 2
    query.modified_after = query.modified_before
    assert not service.search(query)


def test_body_sources_filter_dates_before_selecting_recent_three(db, tmp_path, monkeypatch):
    file = index_file(db, tmp_path / "monthly.txt", "星桥付款凭证")
    attach(db, file, source="九月来源", text="旧月份发送记录")
    for day in range(1, 6):
        attach(db, file, source=f"十月来源{day}", text="最新发送记录",
               sent_at=f"2026-10-0{day}T00:00:00Z")
    original = search_module.process_query
    monkeypatch.setattr(search_module, "process_query", lambda text: original(text, now=datetime(2026, 10, 20)))
    service = searcher(db, mode="keyword")
    result = service.search(SearchQuery("上个月星桥付款凭证", scope="chat_files", evidence="content"))[0]
    assert result.file_id == file["id"]
    assert "九月来源" in result.chat_context and "十月来源" not in result.chat_context
    result = service.search(SearchQuery("星桥付款凭证", scope="chat_files", evidence="content"))[0]
    assert "十月来源5" in result.chat_context


def test_subsecond_dates_offsets_and_half_open_interval(db, tmp_path):
    file = index_file(db, tmp_path / "boundary.txt", "边界记录")
    for key, moment in (("before", "2026-09-30T16:00:00Z"),
                        ("inside", "2026-10-01T00:00:00.000001+08:00"),
                        ("after", "2026-09-30T16:00:00.000002Z")):
        attach(db, file, source=key, text="边界记录", sent_at=moment)
    result = searcher(db, mode="keyword").search(SearchQuery(
        "边界记录", scope="chat_files", evidence="content",
        modified_after="2026-09-30T16:00:00.000001Z",
        modified_before="2026-10-01T00:00:00.000002+08:00",
    ))[0]
    assert "inside" in result.chat_context and "after" not in result.chat_context


def test_file_dates_keep_local_timestamp_semantics(db, tmp_path):
    file = index_file(db, tmp_path / "local.txt", "修改记录")
    moment = datetime.fromisoformat(file["modified_at"])
    epoch = moment.timestamp()
    after = datetime.fromtimestamp(epoch - 1).astimezone().isoformat()
    before = datetime.fromtimestamp(epoch + 1).astimezone().isoformat()
    service = searcher(db, mode="keyword")
    query = SearchQuery("修改记录", scope="files", modified_after=after, modified_before=before)
    assert service.search(query)[0].file_id == file["id"]
    query.modified_after = before
    assert not service.search(query)


def test_platform_filter_uses_chat_date_for_linked_body(db, tmp_path):
    file = index_file(db, tmp_path / "platform.txt", "正文独有代码")
    db.conn.execute("UPDATE files SET modified_at='2026-10-20T12:00:00' WHERE id=?", (file["id"],))
    db.conn.commit()
    attach(db, file, source="日期来源", text="普通交付")
    results = searcher(db, mode="keyword").search(SearchQuery(
        "正文独有代码", scope="all", chat_platform="wechat",
        modified_after="2026-09-01", modified_before="2026-10-01"))
    assert len(results) == 1 and results[0].source_type == "txt"


def test_extension_and_path_filters_before_limit_and_cache_key(db, tmp_path):
    for i in range(110):
        index_file(db, tmp_path / "noise" / f"n{i}.txt", "统一验收说明")
    target = index_file(db, tmp_path / "target" / "target.md", "统一验收说明")
    service = searcher(db, mode="keyword")
    query = SearchQuery("统一验收说明", scope="files", file_type=".MD", limit=5)
    assert [r.file_id for r in service.search(query)] == [target["id"]]
    query.file_type = None
    query.path_contains = str(tmp_path / "target")
    assert [r.file_id for r in service.search(query)] == [target["id"]]
    query.path_contains = str(tmp_path / "missing")
    assert not service.search(query)
    assert service.search("统一验收说明")


def test_sources_are_distinct_attachments_not_repeated_chunks(db, tmp_path):
    file = index_file(db, tmp_path / "same.txt", "付款凭证")
    for day in range(1, 5):
        attach(db, file, source=f"来源{day}", text="很长的聊天记录。" * 600,
               sent_at=f"2026-09-0{day}T00:00:00Z")
    cid = db.conn.execute("SELECT id FROM chunks WHERE file_id=?", (file["id"],)).fetchone()[0]
    row = ChatStore(db).enrich(db.fetch_chunks_by_ids([cid]), SearchQuery("付款", scope="chat_files"))[cid]
    refs = row["chat_refs"]
    assert len(refs) == 3 and len({r["message_id"] for r in refs}) == 3
    assert [r["conversation"] for r in refs] == ["来源4", "来源3", "来源2"]
    assert "节选" in context_label(refs[0])
    assert len(context_label(refs[0])) < 5000
    assert db.conn.execute("SELECT MAX(length(body)) FROM chat_messages").fetchone()[0] > 4000


@pytest.mark.parametrize("query", [
    SearchQuery("test", scope="files", evidence="chat"),
    SearchQuery("test", scope="chat_files", evidence="unknown"),
    SearchQuery("test", scope="chat_files", modified_after="not-a-date"),
])
def test_invalid_filters_fail_explicitly(db, query):
    with pytest.raises(ValueError):
        searcher(db).search(query)


def test_cli_exposes_evidence_mode_and_dates(db, tmp_path, monkeypatch, capsys):
    from localmind import cli
    file = index_file(db, tmp_path / "cli.txt", "正文独有凭证")
    attach(db, file, source="命令行测试", text="普通附件")
    monkeypatch.setattr("sys.argv", ["localmind-index", "--db", str(db.db_path), "search", "正文独有凭证",
        "--scope", "chat_files", "--evidence", "content", "--mode", "keyword", "--no-rerank",
        "--after", "2026-09-01", "--before", "2026-10-01"])
    cli.main()
    output = capsys.readouterr().out
    assert "cli.txt" in output and "排序分" in output and "/100" not in output
    monkeypatch.setattr("sys.argv", ["localmind-index", "--db", str(db.db_path), "search", "test", "--evidence", "content"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_failed_scope_install_invalidates_cache_and_restores_previous_scope(db, tmp_path, monkeypatch):
    file = index_file(db, tmp_path / "cache.txt", "缓存验证")
    attach(db, file, source="来源", text="缓存验证")
    service = searcher(db, mode="keyword")
    query = SearchQuery("缓存验证", scope="chat_files", evidence="content")
    service.search(query)
    previous_ids = service._allowed_ids
    install = db.set_search_scope
    def fail(_ids):
        raise RuntimeError("synthetic scope failure")
    monkeypatch.setattr(db, "set_search_scope", fail)
    query.evidence = "chat"
    with pytest.raises(RuntimeError):
        service.search(query)
    assert service._scope_key is None and service._allowed_ids is previous_ids
    monkeypatch.setattr(db, "set_search_scope", install)
    query.evidence = "content"
    assert service.search(query)[0].file_id == file["id"]
    query.evidence = "content"
    assert service.search(query)[0].file_id == file["id"]


def test_empty_scope_does_not_encode_or_load_query_model(db, tmp_path):
    file = index_file(db, tmp_path / "empty-scope.txt", "正文内容")
    attach(db, file, source="已有来源", text="普通附件")
    service = searcher(db)
    service.embedding_service.encode_query = Mock(side_effect=AssertionError("empty scope must not encode"))
    results = service.search(SearchQuery("模糊描述", scope="chat_files", conversation="不存在的会话"))
    assert not results and "筛选范围" in service.last_diagnostics.semantic_status
    service.embedding_service.encode_query.assert_not_called()
