from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .cache_manager import prepare_model_cache_dir, resolve_model_cache_dir
from .config import RERANKER_MODEL
from .runtime import ResolvedRerankerRuntime, resolve_reranker_runtime


class RerankerService:
    def __init__(
        self,
        *,
        model_name: str = RERANKER_MODEL,
        cache_dir: str | Path | None = None,
        runtime: str = "auto",
        enabled: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = resolve_model_cache_dir(cache_dir)
        self.runtime_request = runtime
        self._runtime: ResolvedRerankerRuntime | None = None
        self.enabled = enabled
        self.local_files_only = local_files_only
        self._model: object | None = None
        self._load_attempted = False
        self._load_error: str | None = None
        self.load_source = "not_loaded"
        self.load_seconds = 0.0

    @property
    def active_model_name(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._model is not None:
            return self.model_name
        return f"{self.model_name}（未加载 / {self.runtime.name.upper()}）"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def runtime(self) -> ResolvedRerankerRuntime:
        if self._runtime is None:
            self._runtime = resolve_reranker_runtime(self.runtime_request)
        return self._runtime

    @property
    def recommended_top_k(self) -> int:
        return 30 if self.runtime.name == "cuda" else 10

    @property
    def runtime_status(self) -> str:
        return self.runtime.message

    def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        if not passages or not self._ensure_model():
            return None
        pairs = [(query, passage) for passage in passages]
        try:
            tokenizer = getattr(self._model, "tokenizer", None)
            limit = getattr(self._model, "max_seq_length", None)
            if limit is None:
                limit = getattr(self._model, "max_length", None)
            if limit is None:
                limit = getattr(tokenizer, "model_max_length", None)
            if callable(tokenizer) and limit is not None:
                encoded = tokenizer([query] * len(passages), passages,
                                    truncation=False, padding=False, verbose=False)
                if any(len(ids) > int(limit) for ids in encoded["input_ids"]):
                    raise ValueError(f"精排输入超过模型 {limit} token 上限，本次保留基础排序")
            predictions = self._model.predict(  # type: ignore[attr-defined]
                pairs,
                batch_size=16 if self.runtime.name == "cuda" else 4,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            values = np.asarray(predictions, dtype="float32")
            if not np.all(np.isfinite(values)):
                raise ValueError("Reranker returned non-finite scores")
            if (values.ndim not in (1, 2) or values.shape[0] != len(passages)
                    or values.ndim == 2 and not values.shape[1]):
                raise ValueError("Reranker returned an invalid score shape")
            self._load_error = None
            if values.ndim == 2 and values.shape[1] > 1:
                shifted = values - values.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
                return probabilities[:, -1].astype(float).tolist()
            scores = values.reshape(-1)
            if np.all((scores >= 0.0) & (scores <= 1.0)):
                return scores.astype(float).tolist()
            logits = scores
            logits = np.clip(logits, -30.0, 30.0)
            return (1.0 / (1.0 + np.exp(-logits))).astype(float).tolist()
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            self._load_error = str(exc)
            return None

    def _ensure_model(self) -> bool:
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        if not self.runtime.available:
            self._load_error = self.runtime.message
            return False
        started = time.perf_counter()
        try:
            self.cache_dir = prepare_model_cache_dir(self.cache_dir)
            os.environ["HF_HOME"] = str(self.cache_dir)
            from sentence_transformers import CrossEncoder

            model_reference = self.model_name
            onnx_cache = self.cache_dir / "onnx" / self.model_name.replace("/", "--")
            if self.runtime.name == "onnx" and (onnx_cache / "config.json").exists():
                model_reference = str(onnx_cache)
            model_kwargs: dict[str, object] = {}
            if self.runtime.name == "cuda":
                model_kwargs["torch_dtype"] = "float16"
            if self.runtime.provider:
                model_kwargs["provider"] = self.runtime.provider
            kwargs: dict[str, object] = {
                "cache_folder": str(self.cache_dir),
                "device": self.runtime.device,
                "backend": self.runtime.backend,
                "model_kwargs": model_kwargs,
            }
            try:
                self._model = CrossEncoder(model_reference, local_files_only=True, **kwargs)
                self.load_source = (
                    "local_onnx_cache" if model_reference != self.model_name else "local_cache"
                )
            except Exception:
                if self.local_files_only:
                    raise
                self._model = CrossEncoder(self.model_name, local_files_only=False, **kwargs)
                self.load_source = "download_or_refresh"
            if self.runtime.name == "onnx" and model_reference == self.model_name:
                try:
                    onnx_cache.parent.mkdir(parents=True, exist_ok=True)
                    self._model.save_pretrained(str(onnx_cache))  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.load_seconds = time.perf_counter() - started
            return True
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            self._load_error = str(exc)
            self.load_seconds = time.perf_counter() - started
            return False
