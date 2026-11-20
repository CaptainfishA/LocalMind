from __future__ import annotations

import importlib.util
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .cache_manager import prepare_model_cache_dir, resolve_model_cache_dir
from .config import PDF_OCR_PAGE_LIMIT
from .models import OcrResult, utc_now_iso


class OcrService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        use_gpu: bool = False,
        cache_dir: str | Path | None = None,
        engine: str = "auto",
    ) -> None:
        self.enabled = enabled
        self.use_gpu = use_gpu
        self.cache_dir = resolve_model_cache_dir(cache_dir)
        self.engine = self._resolve_engine(engine)
        self._engine: Any | None = None
        self._load_error: str | None = None
        self._load_attempted = False
        self.load_seconds = 0.0

    @property
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        self._ensure_engine()
        return self._engine is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def readiness_status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._engine is not None:
            return "ready"
        if self._load_error is not None:
            return "error"
        if importlib.util.find_spec("paddleocr") is None:
            return "missing"
        return "unchecked"

    @property
    def readiness_message(self) -> str:
        messages = {
            "disabled": "OCR 已关闭",
            "ready": f"PaddleOCR / {self.engine} 已就绪（{self.load_seconds:.1f} 秒）",
            "missing": "未安装 PaddleOCR",
            "unchecked": "PaddleOCR 已安装，尚未检查模型",
        }
        return messages.get(self.readiness_status, f"OCR 初始化失败：{self._load_error}")

    def check_ready(self) -> bool:
        self._ensure_engine()
        return self._engine is not None

    def _ensure_engine(self) -> None:
        if self._engine is not None or self._load_attempted or not self.enabled:
            return
        self._load_attempted = True
        started = time.perf_counter()
        try:
            self.cache_dir = prepare_model_cache_dir(self.cache_dir)
            paddle_cache = self.cache_dir / "paddlex"
            paddle_cache.mkdir(parents=True, exist_ok=True)
            os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache)
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddleocr import PaddleOCR

            try:
                import paddlex.utils.cache as paddlex_cache

                paddlex_cache.CACHE_DIR = str(paddle_cache)
                paddlex_cache.FUNC_CACHE_DIR = str(paddle_cache / "func_ret")
                paddlex_cache.FILE_LOCK_DIR = str(paddle_cache / "locks")
                paddlex_cache.TEMP_DIR = str(paddle_cache / "temp")
            except Exception:
                pass

            try:
                self._engine = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang="ch",
                    engine=self.engine,
                    device="gpu" if self.use_gpu else "cpu",
                )
            except TypeError:
                self._engine = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=self.use_gpu)
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            self._load_error = str(exc)
        finally:
            self.load_seconds = time.perf_counter() - started

    @staticmethod
    def _resolve_engine(requested: str) -> str:
        normalized = requested.lower().strip()
        if normalized != "auto":
            return normalized
        return "onnxruntime" if importlib.util.find_spec("onnxruntime") else "paddle"

    def ocr_image(self, image_path: str | Path, *, page_no: int | None = None) -> OcrResult:
        if not self.enabled:
            return OcrResult(text="", status="disabled", page_no=page_no, created_at=utc_now_iso())
        self._ensure_engine()
        if self._engine is None:
            return OcrResult(
                text="",
                status="skipped",
                page_no=page_no,
                error=self._load_error or "PaddleOCR is not available",
                created_at=utc_now_iso(),
            )
        try:
            raw = self._predict(image_path)
            text, confidence = self._extract_text_and_confidence(raw)
            return OcrResult(
                text=text,
                status="done" if text else "empty",
                page_no=page_no,
                confidence=confidence,
                created_at=utc_now_iso(),
            )
        except Exception as exc:  # pragma: no cover - depends on OCR runtime
            return OcrResult(
                text="",
                status="failed",
                page_no=page_no,
                error=str(exc),
                created_at=utc_now_iso(),
            )

    def ocr_pdf_pages(self, pdf_path: str | Path, *, max_pages: int = PDF_OCR_PAGE_LIMIT) -> list[OcrResult]:
        self._ensure_engine()
        if self._engine is None:
            return [
                OcrResult(
                    text="",
                    status="skipped",
                    error=self._load_error or "PaddleOCR is not available",
                    created_at=utc_now_iso(),
                )
            ]
        try:
            import pymupdf
        except Exception as exc:
            return [OcrResult(text="", status="failed", error=str(exc), created_at=utc_now_iso())]

        results: list[OcrResult] = []
        with pymupdf.open(pdf_path) as doc:
            page_count = min(doc.page_count, max_pages)
            with tempfile.TemporaryDirectory(prefix="localmind_pdf_ocr_") as tmp:
                tmp_dir = Path(tmp)
                for index in range(page_count):
                    page = doc.load_page(index)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    image_path = tmp_dir / f"page-{index + 1}.png"
                    pix.save(image_path)
                    results.append(self.ocr_image(image_path, page_no=index + 1))
        return results

    def _predict(self, image_path: str | Path) -> Any:
        if hasattr(self._engine, "predict"):
            return self._engine.predict(str(image_path))
        return self._engine.ocr(str(image_path), cls=False)

    def _extract_text_and_confidence(self, raw: Any) -> tuple[str, float | None]:
        texts: list[str] = []
        confidences: list[float] = []

        def walk(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, dict):
                for key in ("rec_texts", "texts"):
                    value = item.get(key)
                    if isinstance(value, list):
                        texts.extend(str(part) for part in value if part)
                for key in ("rec_scores", "scores"):
                    value = item.get(key)
                    if isinstance(value, list):
                        confidences.extend(float(part) for part in value if part is not None)
                return
            if isinstance(item, str):
                texts.append(item)
                return
            if isinstance(item, (list, tuple)):
                if len(item) >= 2 and isinstance(item[1], (list, tuple)) and item[1]:
                    candidate = item[1]
                    if isinstance(candidate[0], str):
                        texts.append(candidate[0])
                        if len(candidate) > 1:
                            try:
                                confidences.append(float(candidate[1]))
                            except (TypeError, ValueError):
                                pass
                        return
                for child in item:
                    walk(child)

        walk(raw)
        confidence = sum(confidences) / len(confidences) if confidences else None
        return "\n".join(texts).strip(), confidence
