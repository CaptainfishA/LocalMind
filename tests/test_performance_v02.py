from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from localmind.cache_manager import (
    CACHE_MARKER,
    clear_model_cache,
    model_cache_stats,
    prepare_custom_cache_dir,
)
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.indexer import IndexingService
from localmind.ocr import OcrService
from localmind.reranker import RerankerService
from localmind.runtime import ResolvedRerankerRuntime
from localmind.runtime import RuntimeCapabilities, resolve_reranker_runtime
from localmind.vector_index import LocalVectorIndex


class CountingEmbedding:
    active_model_name = "test-batch-embedding"
    load_error = None

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.batch_sizes.append(len(texts))
        vectors = np.ones((len(texts), 8), dtype="float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_scan_batches_embeddings_across_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for index in range(12):
        (docs / f"文档-{index}.txt").write_text(f"第 {index} 份客户报价与付款信息。", encoding="utf-8")

    embedding = CountingEmbedding()
    db = LocalMindDB(tmp_path / "localmind.db")
    summary = IndexingService(
        db,
        embedding_service=embedding,  # type: ignore[arg-type]
        ocr_service=OcrService(enabled=False),
    ).scan_and_index([docs])

    assert summary.files_indexed == 12
    assert embedding.batch_sizes == [12]
    assert db.stats().vectors_total == 12


def test_embedding_tries_local_cache_before_network(tmp_path: Path, monkeypatch) -> None:
    calls: list[bool] = []

    class FakeSentenceTransformer:
        def __init__(self, *args, local_files_only: bool, **kwargs) -> None:
            calls.append(local_files_only)

        def encode(self, items, **kwargs):
            return np.ones((len(items), 4), dtype="float32")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    service = EmbeddingService(cache_dir=tmp_path / "cache", device="cpu")
    service.encode_query("付款信息")

    assert calls == [True]
    assert service.load_source == "local_cache"


def test_custom_cache_clear_only_removes_managed_cache(tmp_path: Path) -> None:
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    cache_dir = prepare_custom_cache_dir(tmp_path)
    (cache_dir / "model.bin").write_bytes(b"1234")
    nested = cache_dir / "nested"
    nested.mkdir()
    (nested / "tokenizer.json").write_text("{}", encoding="utf-8")

    before = model_cache_stats(cache_dir)
    removed = clear_model_cache(cache_dir)

    assert before.files == 2
    assert removed.bytes_total == before.bytes_total
    assert unrelated.exists()
    assert (cache_dir / CACHE_MARKER).exists()
    assert model_cache_stats(cache_dir).files == 0


def test_runtime_resolution_prefers_accelerated_backends() -> None:
    cuda_caps = RuntimeCapabilities(True, True, "Test GPU", False, ())
    onnx_caps = RuntimeCapabilities(True, False, None, True, ("CPUExecutionProvider",))
    cpu_caps = RuntimeCapabilities(True, False, None, False, ())

    assert resolve_reranker_runtime("auto", cuda_caps).name == "cuda"
    assert resolve_reranker_runtime("auto", onnx_caps).name == "onnx"
    assert resolve_reranker_runtime("auto", cpu_caps).name == "cpu"
    assert cpu_caps.default_reranker_enabled is False


def test_reranker_does_not_apply_sigmoid_twice() -> None:
    class FakeCrossEncoder:
        def predict(self, pairs, **kwargs):
            return np.asarray([0.01, 0.95], dtype="float32")

    service = RerankerService(enabled=True)
    service._model = FakeCrossEncoder()
    service._runtime = ResolvedRerankerRuntime(
        requested="cpu",
        name="cpu",
        backend="torch",
        device="cpu",
        provider=None,
        available=True,
        message="test",
    )

    scores = service.rerank("query", ["bad", "good"])
    assert scores is not None
    assert scores == pytest.approx([0.01, 0.95])


def test_ocr_v3_initialization_uses_supported_arguments(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))
    service = OcrService(cache_dir=tmp_path / "cache", engine="onnxruntime")

    assert service.check_ready()
    assert service.readiness_status == "ready"
    assert calls == [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "lang": "ch",
            "engine": "onnxruntime",
            "device": "cpu",
        }
    ]


def test_vector_index_is_rebuilt_only_when_explicitly_refreshed(tmp_path: Path) -> None:
    class CountingVectorIndex(LocalVectorIndex):
        def __init__(self) -> None:
            super().__init__()
            self.rebuilds = 0

        def rebuild(self, ids, vectors) -> None:
            self.rebuilds += 1
            super().rebuild(ids, vectors)

    db = LocalMindDB(tmp_path / "localmind.db")
    vector_index = CountingVectorIndex()
    from localmind.reranker import RerankerService
    from localmind.searcher import SearchService

    service = SearchService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        reranker_service=RerankerService(enabled=False),
        vector_index=vector_index,
    )
    service.search("付款")
    service.search("报价")
    assert vector_index.rebuilds == 1

    service.refresh_vector_index()
    assert vector_index.rebuilds == 2
