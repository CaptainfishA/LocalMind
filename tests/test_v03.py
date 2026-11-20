from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService, SentenceTransformerBackend
from localmind.evaluation.dataset import build_text_subset, load_dataset
from localmind.evaluation.fixtures import build_fixtures
from localmind.evaluation.metrics import ranking_metrics
from localmind.evaluation.runner import run_evaluation
from localmind.indexer import IndexingService
from localmind.monitor import FileEvent, apply_file_events
from localmind.ocr import OcrService
from localmind.reranker import RerankerService
from localmind.scanner import build_file_record
from localmind.searcher import SearchService


def indexer(db):
    return IndexingService(db, embedding_service=EmbeddingService(force_hash=True),
                           ocr_service=OcrService(enabled=False))


def test_metrics_distinguish_hit_from_recall_and_dedupe():
    result = ranking_metrics(["bad", "a", "a", "b"], {"a": 3, "b": 1, "c": 2})
    assert result["hit_at_1"] == 0
    assert result["hit_at_5"] == 1
    assert result["recall_at_5"] == pytest.approx(2 / 3)
    assert result["mrr_at_10"] == 0.5
    assert 0 < result["ndcg_at_10"] < 1
    assert result["judged_at_10"] == pytest.approx(2 / 3)
    assert ranking_metrics(["a", "c", "b"], {"a": 3, "b": 1, "c": 2})["ndcg_at_10"] == 1
    with pytest.raises(ValueError):
        ranking_metrics([], {})


def test_subset_keeps_all_positives_and_freezes_splits(tmp_path):
    docs = [{"id": str(i), "text": f"办公资料采购合同 {i}"} for i in range(30)]
    queries = [{"id": str(i), "text": f"合同内容 {i}"} for i in range(6)]
    qrels = [{"query_id": str(i), "file_id": str(i), "relevance": 1} for i in range(6)]
    qrels.append({"query_id": "1", "file_id": "0", "relevance": 1})
    a = build_text_subset(docs, queries, qrels, tmp_path / "a", query_count=6, corpus_size=15)
    b = build_text_subset(docs, queries, qrels, tmp_path / "b", query_count=6, corpus_size=15)
    ma, ca, qa, ra = load_dataset(a)
    mb, *_ = load_dataset(b)
    assert ma == mb
    assert len(ca) == 15
    assert {"0", "1", "2", "3", "4", "5"} <= {d["file_id"] for d in ca}
    splits = {q["query_id"]: q["split"] for q in qa}
    assert splits["0"] == splits["1"]
    assert set(splits.values()) == {"dev", "test"}
    assert len(ra) == 7
    with pytest.raises(ValueError, match="drop positives"):
        build_text_subset(docs, queries, qrels, tmp_path / "bad", query_count=6, corpus_size=2)
    (a / ca[0]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Document changed"):
        load_dataset(a)


def test_office_fixture_and_eval_end_to_end(tmp_path):
    dataset = build_fixtures(tmp_path / "dataset", projects=2)
    duplicate = build_fixtures(tmp_path / "duplicate", projects=2)
    manifest, corpus, queries, _ = load_dataset(dataset)
    other, *_ = load_dataset(duplicate)
    assert manifest == other
    assert len(corpus) == 14 and len(queries) == 12
    report = run_evaluation(dataset, tmp_path / "result", mode="keyword")
    assert report["index"]["files_indexed"] == 14
    assert report["index"]["errors"] == 0
    assert report["metrics"]["hit_at_5"] > 0.5
    assert (tmp_path / "result" / "queries.jsonl").exists()
    with pytest.raises(ValueError, match="outside"):
        run_evaluation(dataset, dataset / "result", mode="keyword")


def test_cancelled_scan_keeps_unvisited_indexes(tmp_path):
    docs = tmp_path / "files"
    docs.mkdir()
    for i in range(5):
        (docs / f"{i}.txt").write_text("原始付款内容", encoding="utf-8")
    db = LocalMindDB(tmp_path / "index.db")
    try:
        worker = indexer(db)
        worker.scan_and_index([docs])
        (docs / "4.txt").unlink()
        checks = 0

        def progress(_):
            nonlocal checks
            checks += 1

        summary = worker.scan_and_index([docs], progress=progress, should_cancel=lambda: checks >= 2)
        assert summary.cancelled and summary.files_removed == 0
        assert db.stats().files_total == 5
        assert db.conn.execute("SELECT status FROM scan_jobs ORDER BY id DESC LIMIT 1").fetchone()[0] == "cancelled"
        assert worker.scan_and_index([docs]).files_removed == 1
    finally:
        db.close()


def test_same_second_same_size_edit_is_detected(tmp_path):
    docs = tmp_path / "files"
    docs.mkdir()
    file = docs / "data.txt"
    file.write_text("旧款合同", encoding="utf-8")
    os.utime(file, ns=(1700000000120000000, 1700000000120000000))
    db = LocalMindDB(tmp_path / "index.db")
    try:
        worker = indexer(db)
        worker.scan_and_index([docs])
        file.write_text("新款合同", encoding="utf-8")
        os.utime(file, ns=(1700000000450000000, 1700000000450000000))
        assert worker.scan_and_index([docs]).files_indexed == 1
        assert "新款" in db.conn.execute("SELECT chunk_text FROM chunks").fetchone()[0]
    finally:
        db.close()


def test_file_changes_during_parse_preserves_old_content(tmp_path):
    docs = tmp_path / "files"
    docs.mkdir()
    file = docs / "data.txt"
    file.write_text("原始内容", encoding="utf-8")
    db = LocalMindDB(tmp_path / "index.db")
    try:
        worker = indexer(db)
        worker.scan_and_index([docs])
        file.write_text("新内容待解析", encoding="utf-8")
        original_parse = worker.parser.parse

        def changing_parse(path):
            result = original_parse(path)
            file.write_text("又发生了一次内容变化", encoding="utf-8")
            return result

        worker.parser.parse = changing_parse
        result = worker.scan_and_index([docs])
        assert result.errors == 1
        assert "原始内容" in db.conn.execute("SELECT chunk_text FROM chunks").fetchone()[0]
    finally:
        db.close()


def test_atomic_recreate_in_monitor_batch_updates_index(tmp_path):
    file = tmp_path / "atomic.txt"
    file.write_text("原始资料", encoding="utf-8")
    db_path = tmp_path / "index.db"
    db = LocalMindDB(db_path)
    try:
        indexer(db).index_path(file)
        file.write_text("重新创建的报价内容", encoding="utf-8")
        result = apply_file_events(db_path, [FileEvent("deleted", str(file)), FileEvent("created", str(file))],
                                   enable_ocr=False, enable_embeddings=False)
        assert result.indexed == 1 and result.removed == 0
        assert "重新创建" in db.conn.execute("SELECT chunk_text FROM chunks").fetchone()[0]
    finally:
        db.close()


def test_cuda_oom_shrinks_then_falls_back_without_model_change(monkeypatch):
    import sys

    class OOM(RuntimeError):
        pass

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Model:
        device = "cuda"
        sizes = []

        def encode(self, items, batch_size, **kwargs):
            self.sizes.append(batch_size)
            if self.device == "cuda":
                raise OOM("out of memory")
            return np.ones((len(items), 4), dtype="float32")

        def to(self, device):
            self.device = device
            return self

        def float(self):
            return self

    model = Model()
    backend = SentenceTransformerBackend(model, "same-model", device="cuda")
    result = backend.encode(["中文查询"])
    assert model.sizes == [64, 32, 16, 8, 4, 2, 1, 1]
    assert backend.device == "cpu" and backend.name == "same-model"
    assert np.linalg.norm(result[0]) == pytest.approx(1)


def test_keyword_mode_never_loads_embedding(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        embedding = EmbeddingService(force_hash=True)
        service = SearchService(db, embedding_service=embedding, retrieval_mode="keyword",
                                reranker_service=RerankerService(enabled=False))
        service.search("合同")
        assert embedding.load_source == "not_loaded"
        assert set(service.last_diagnostics.timings_ms) == {"query", "keyword", "semantic", "companion", "clause", "fusion", "rerank", "aggregate", "file_check", "snapshot", "embedding"}
    finally:
        db.close()


def test_ocr_reenabled_retries_unchanged_file(tmp_path):
    from localmind.models import TextChunk

    file = tmp_path / "image.png"
    file.write_bytes(b"fake-image-fixture")
    db = LocalMindDB(tmp_path / "index.db")
    try:
        worker = indexer(db)
        worker.parser.parse = lambda _: ([], "empty", "disabled")
        worker.scan_and_index([tmp_path], include_embeddings=False)
        worker.ocr_service.enabled = True
        worker.parser.parse = lambda _: ([TextChunk(chunk_text="发票编号123", source_type="ocr_image")], "parsed", "done")
        assert worker.scan_and_index([tmp_path], include_embeddings=False).files_indexed == 1
        assert "发票" in db.conn.execute("SELECT chunk_text FROM chunks").fetchone()[0]
        file.write_bytes(b"changed-image-fixture")
        worker.parser.parse = lambda _: ([], "empty", "failed")
        assert worker.scan_and_index([tmp_path], include_embeddings=False).errors == 1
        assert "发票" in db.conn.execute("SELECT chunk_text FROM chunks").fetchone()[0]
    finally:
        db.close()


def test_comparison_rejects_different_test_sets(tmp_path):
    from localmind.evaluation.compare import compare_reports

    before = {"dataset": {"checksums": {"queries": "a"}}, "split": "test", "query_count": 3,
              "metrics": {"hit_at_1": 0.5}, "performance": {"hot_p95_ms": 100}}
    after = {**before, "metrics": {"hit_at_1": 1.0}}
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(before), encoding="utf-8")
    right.write_text(json.dumps(after), encoding="utf-8")
    assert compare_reports(left, right)["metrics"]["hit_at_1"]["delta"] == 0.5
    after["split"] = "dev"
    right.write_text(json.dumps(after), encoding="utf-8")
    with pytest.raises(ValueError, match="same frozen dataset"):
        compare_reports(left, right)
