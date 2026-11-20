from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .cache_manager import prepare_model_cache_dir, resolve_model_cache_dir
from .config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class HashEmbeddingBackend:
    name = "localmind-hash-embedding"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def encode(self, texts: Iterable[str], *, is_query: bool = False) -> np.ndarray:
        vectors = [self._encode_one(text) for text in texts]
        if not vectors:
            return np.empty((0, self.dim), dtype="float32")
        return normalize_vectors(np.vstack(vectors))

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype="float32")
        compact = "".join(text.lower().split())
        tokens = set()
        for size in (1, 2, 3):
            for i in range(max(0, len(compact) - size + 1)):
                tokens.add(compact[i : i + size])
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dim
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vector[index] += sign
        return vector


class EmbeddingService:
    def __init__(
        self,
        *,
        model_name: str = EMBEDDING_MODEL,
        cache_dir: str | Path | None = None,
        force_hash: bool = False,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = resolve_model_cache_dir(cache_dir)
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported embedding device: {device}")
        self.device_request = device
        self.device = "cpu"
        self.local_files_only = local_files_only
        self.runtime_note = ""
        self._backend: object | None = None
        self._load_error: str | None = None
        self.load_source = "not_loaded"
        self.load_seconds = 0.0
        self.force_hash = force_hash

    @property
    def active_model_name(self) -> str:
        backend = self._ensure_backend()
        return getattr(backend, "name", self.model_name)

    @property
    def load_error(self) -> str | None:
        self._ensure_backend()
        return self._load_error

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        backend = self._ensure_backend()
        result = backend.encode(texts, is_query=False)  # type: ignore[attr-defined]
        self._sync_runtime(backend)
        return result

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        backend = self._ensure_backend()
        result = backend.encode(texts, is_query=True)  # type: ignore[attr-defined]
        self._sync_runtime(backend)
        return result

    def _sync_runtime(self, backend: object) -> None:
        self.device = getattr(backend, "device", "cpu")
        self.runtime_note = getattr(backend, "runtime_note", self.runtime_note)

    def _ensure_backend(self) -> object:
        if self._backend is not None:
            return self._backend
        if self.force_hash:
            self._backend = HashEmbeddingBackend()
            self.load_source = "hash"
            return self._backend
        started = time.perf_counter()
        try:
            self.cache_dir = prepare_model_cache_dir(self.cache_dir)
            os.environ["HF_HOME"] = str(self.cache_dir)
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self.cache_dir),
                    device="cpu",
                    local_files_only=True,
                )
                self.load_source = "local_cache"
            except OSError:
                if self.local_files_only:
                    raise
                model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self.cache_dir),
                    device="cpu",
                    local_files_only=False,
                )
                self.load_source = "download_or_refresh"
            if self.device_request != "cpu":
                import torch

                if torch.cuda.is_available():
                    try:
                        model.to("cuda")
                        model.half()
                        self.device = "cuda"
                    except RuntimeError as exc:
                        model.to("cpu").float()
                        self.runtime_note = f"CUDA 初始化失败，使用 CPU：{exc}"
                elif self.device_request == "cuda":
                    self.runtime_note = "CUDA 不可用，使用 CPU"
            self._backend = SentenceTransformerBackend(model, self.model_name, device=self.device)
            self._backend.runtime_note = self.runtime_note
            self.load_seconds = time.perf_counter() - started
            return self._backend
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            self._load_error = str(exc)
            self._backend = HashEmbeddingBackend()
            self.load_source = "hash_fallback"
            self.load_seconds = time.perf_counter() - started
            return self._backend


class SentenceTransformerBackend:
    query_instruction = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model: object, model_name: str, *, device: str = "cpu") -> None:
        self.model = model
        self.name = model_name
        self.device = device
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.runtime_note = ""

    def encode(self, texts: Iterable[str], *, is_query: bool = False) -> np.ndarray:
        items = list(texts)
        if not items:
            dim = self.model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]
            return np.empty((0, dim), dtype="float32")
        if is_query:
            items = [self.query_instruction + item for item in items]
        elif hasattr(self.model, "tokenizer"):
            # A character limit is not a tokenizer guarantee. Never silently lose a tail.
            limit = int(self.model.max_seq_length)
            encoded = self.model.tokenizer(items, truncation=False, padding=False, verbose=False)
            if any(len(ids) > limit for ids in encoded["input_ids"]):
                raise ValueError(f"文本块超过语义模型 {limit} token 上限；未生成截断向量，请更新文本分块")
        while True:
            try:
                vectors = self.model.encode(  # type: ignore[attr-defined]
                    items,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                break
            except RuntimeError as exc:
                import torch

                if self.device != "cuda" or not isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                if self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    self.runtime_note = f"显存不足，批次缩小至 {self.batch_size}"
                else:
                    self.model.to("cpu").float()  # type: ignore[attr-defined]
                    self.device = "cpu"
                    self.runtime_note = "显存不足，已回退 CPU（保持相同语义模型）"
                torch.cuda.empty_cache()
        return normalize_vectors(np.asarray(vectors, dtype="float32"))
