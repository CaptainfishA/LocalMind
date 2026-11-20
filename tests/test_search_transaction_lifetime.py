from __future__ import annotations

from types import SimpleNamespace

import pytest

from localmind.models import SearchQuery
from localmind import file_freshness
from test_search_snapshot import document, setup


def commit_and_checkpoint(writer, name):
    with writer.conn:
        writer.conn.execute("UPDATE files SET name=? WHERE id=1", (name,))
    writer.conn.execute("PRAGMA busy_timeout=0")
    return tuple(writer.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())


def test_embedding_preparation_does_not_pin_wal_and_recall_sees_new_commit(setup, monkeypatch):
    reader, writer, service, _, _, _, _ = setup
    states = []
    encode = service.embedding_service.encode_query

    def checked_encode(text):
        states.append((reader.conn.in_transaction, commit_and_checkpoint(writer, "during-encoding.txt")))
        return encode(text)

    monkeypatch.setattr(service.embedding_service, "encode_query", checked_encode)
    result = service.search(SearchQuery("付款", scope="files"))[0]
    assert len(states) == 1
    assert not states[0][0]
    assert states[0][1][0] == 0
    assert result.name == "during-encoding.txt"
    assert result.semantic_score == pytest.approx(1)
    assert service.last_diagnostics.index_current


def test_reranker_does_not_pin_wal_and_preserves_collected_evidence(setup):
    reader, writer, service, _, _, _, record = setup
    states = []

    def checked_rerank(_query, passages):
        states.append((reader.conn.in_transaction, commit_and_checkpoint(writer, "during-rerank.txt")))
        assert "付款条款" in passages[0]
        return [0.8] * len(passages)

    service.reranker_service = SimpleNamespace(
        enabled=True, recommended_top_k=30, model_name="synthetic", load_error=None,
        runtime=SimpleNamespace(name="cpu"), rerank=checked_rerank,
    )
    result = service.search("付款")[0]
    assert states and not states[0][0]
    assert states[0][1][0] == 0
    assert result.name == record.name and "付款条款" in result.snippet
    assert not service.last_diagnostics.index_current


@pytest.mark.parametrize("case", ["keyword", "empty_scope", "empty_index"])
def test_ineligible_queries_do_not_prepare_embedding(setup, monkeypatch, case):
    _, writer, service, _, _, _, _ = setup
    calls = []
    monkeypatch.setattr(service.embedding_service, "encode_query", lambda text: calls.append(text))
    query = SearchQuery("付款", scope="files")
    if case == "keyword":
        service.retrieval_mode = "keyword"
    elif case == "empty_scope":
        query.path_contains = "no-such-path"
    else:
        writer.clear_index()
    results = service.search(query)
    assert not calls
    assert service.last_diagnostics.timings_ms["embedding"] == 0
    assert bool(results) == (case == "keyword")


def test_index_model_changes_during_preparation_are_checked_without_second_encoding(setup, monkeypatch):
    reader, writer, service, _, _, _, _ = setup
    encode = service.embedding_service.encode_query
    calls = []

    def change_model(text):
        calls.append(reader.conn.in_transaction)
        with writer.conn:
            writer.conn.execute("UPDATE vectors SET model_name='different-model'")
        return encode(text)

    monkeypatch.setattr(service.embedding_service, "encode_query", change_model)
    assert service.search("付款") == []
    assert calls == [False]
    assert "模型不一致" in service.last_diagnostics.semantic_status
    assert service.last_diagnostics.indexed_embedding_model == "different-model"
    assert service.last_diagnostics.index_current
    assert not reader.conn.in_transaction


def test_index_cleared_during_preparation_does_not_resurrect_old_candidates(setup, monkeypatch):
    reader, writer, service, _, _, _, _ = setup
    encode = service.embedding_service.encode_query

    def clear(text):
        writer.clear_index()
        return encode(text)

    monkeypatch.setattr(service.embedding_service, "encode_query", clear)
    assert service.search("付款") == []
    assert not service.vector_index.ids
    assert service.last_diagnostics.index_current
    assert not reader.conn.in_transaction


def test_embedding_failure_is_attempted_once_and_keyword_fallback_survives(setup, monkeypatch):
    reader, _, service, _, fid, _, _ = setup
    calls = []
    service.retrieval_mode = "hybrid"

    def fail(_text):
        calls.append(reader.conn.in_transaction)
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(service.embedding_service, "encode_query", fail)
    assert service.search("付款条款")[0].file_id == fid
    assert calls == [False]
    assert "synthetic model failure" in service.last_diagnostics.semantic_status
    assert service.last_diagnostics.semantic_candidates == 0
    assert not reader.conn.in_transaction


def test_file_metadata_io_does_not_pin_wal(setup, monkeypatch):
    reader, writer, service, _, _, _, _ = setup
    states = []
    inspect = file_freshness.inspect_file

    def checked_inspect(record, **kwargs):
        states.append((reader.conn.in_transaction, commit_and_checkpoint(writer, "during-stat.txt")))
        return inspect(record, **kwargs)

    monkeypatch.setattr(file_freshness, "inspect_file", checked_inspect)
    assert service.search("付款")
    assert states and not states[0][0] and states[0][1][0] == 0
    assert not service.last_diagnostics.index_current


def test_callers_outer_transaction_is_preserved_and_reported(setup, monkeypatch):
    reader, _, service, _, _, _, _ = setup
    encode = service.embedding_service.encode_query
    states = []
    reader.conn.execute("BEGIN")
    reader.index_revisions()
    try:
        monkeypatch.setattr(service.embedding_service, "encode_query",
                            lambda text: states.append(reader.conn.in_transaction) or encode(text))
        assert service.search("付款")
        assert states == [True]
        assert reader.conn.in_transaction and service.last_diagnostics.caller_transaction
    finally:
        reader.conn.rollback()


def test_query_preprocessing_also_runs_before_read_snapshot(setup, monkeypatch):
    from localmind import searcher as module

    reader, _, service, _, _, _, _ = setup
    process = module.process_query
    states = []
    monkeypatch.setattr(module, "process_query",
                        lambda text: states.append(reader.conn.in_transaction) or process(text))
    assert service.search("付款")
    assert states == [False]


def test_metadata_committed_during_model_preparation_does_not_rebuild_vectors(setup, monkeypatch):
    reader, writer, service, index, _, _, _ = setup
    encode = service.embedding_service.encode_query
    rebuilds = index.rebuilds

    def update(text):
        with writer.conn:
            writer.conn.execute("UPDATE files SET name='latest.txt'")
        return encode(text)

    monkeypatch.setattr(service.embedding_service, "encode_query", update)
    assert service.search("付款")[0].name == "latest.txt"
    assert index.rebuilds == rebuilds
    assert not reader.conn.in_transaction


def test_nonempty_scope_without_its_own_vectors_skips_model_loading(setup, tmp_path, monkeypatch):
    _, writer, service, _, fid, cid, _ = setup
    document(writer, tmp_path / "different.txt", "另一文件")
    with writer.conn:
        writer.conn.execute("DELETE FROM vectors WHERE chunk_id=?", (cid,))
    calls = []
    service.retrieval_mode = "hybrid"
    monkeypatch.setattr(service.embedding_service, "encode_query", lambda text: calls.append(text))
    result = service.search(SearchQuery("付款", scope="files", path_contains="original.txt"))
    assert result[0].file_id == fid and not calls
    assert service.last_diagnostics.timings_ms["embedding"] == 0
    assert service.last_diagnostics.semantic_status == "当前筛选范围没有语义向量"
