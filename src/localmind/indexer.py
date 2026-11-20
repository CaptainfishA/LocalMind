from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import INDEX_CHUNK_BATCH_SIZE, INDEX_FILE_BATCH_SIZE
from .db import LocalMindDB
from .embeddings import EmbeddingService
from .models import FileRecord, TextChunk
from .ocr import OcrService
from .parsers import DocumentParser
from .scanner import build_file_record, hash_file, scan_folder
from .vector_index import LocalVectorIndex
from .chunker import TEXT_INDEX_VERSION, rechunk

ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class ScanSummary:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    errors: int = 0
    cancelled: bool = False


@dataclass(slots=True)
class PreparedFileIndex:
    record: FileRecord
    chunks: list[TextChunk]
    parse_status: str
    ocr_status: str
    reused_text: bool = False


class IndexingService:
    def __init__(
        self,
        db: LocalMindDB,
        *,
        embedding_service: EmbeddingService | None = None,
        ocr_service: OcrService | None = None,
        vector_index: LocalVectorIndex | None = None,
        maintain_vector_index: bool = False,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self.ocr_service = ocr_service or OcrService()
        self.parser = DocumentParser(ocr_service=self.ocr_service)
        self.vector_index = vector_index or LocalVectorIndex()
        self.maintain_vector_index = maintain_vector_index
        if self.maintain_vector_index:
            self.rebuild_vector_index()

    def rebuild_vector_index(self) -> None:
        model_name = self.db.get_setting("embedding_model")
        ids, vectors = self.db.load_vectors(model_name=model_name)
        self.vector_index.rebuild(ids, vectors)

    def scan_and_index(
        self,
        roots: list[str | Path],
        *,
        include_embeddings: bool = True,
        progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScanSummary:
        summary = ScanSummary()
        if should_cancel and should_cancel():
            summary.cancelled = True
            return summary
        embedding_model = self._prepare_embedding_model(progress) if include_embeddings else None

        for root in roots:
            if should_cancel and should_cancel():
                summary.cancelled = True
                break
            root_path = str(Path(root))
            job_id = self.db.create_scan_job(root_path)
            status = "done"
            root_seen: set[str] = set()
            traversal_errors: list[OSError] = []
            root_seen_count = 0
            root_indexed = 0
            root_errors = 0
            prepared_batch: list[PreparedFileIndex] = []
            prepared_chunks = 0

            def flush_batch() -> None:
                nonlocal prepared_batch, prepared_chunks, root_indexed, root_errors, status
                if not prepared_batch:
                    return
                if should_cancel and should_cancel():
                    summary.cancelled = True
                    prepared_batch = []
                    prepared_chunks = 0
                    return
                indexed, failures = self._commit_prepared_batch(
                    prepared_batch,
                    include_embeddings=include_embeddings,
                    embedding_model_name=embedding_model,
                    progress=progress,
                )
                root_indexed += indexed
                summary.files_indexed += indexed
                for failed_record, error in failures:
                    root_errors += 1
                    summary.errors += 1
                    status = "done_with_errors"
                    self._record_file_error(failed_record, error, progress)
                prepared_batch = []
                prepared_chunks = 0

            try:
                for record in scan_folder(root, on_error=traversal_errors.append):
                    if should_cancel and should_cancel():
                        summary.cancelled = True
                        break
                    root_seen.add(record.path)
                    root_seen_count += 1
                    summary.files_seen += 1
                    if progress:
                        progress(f"检查：{record.path}")
                    try:
                        prepared = self._prepare_file(
                            record,
                            include_embeddings=include_embeddings,
                            embedding_model_name=embedding_model,
                            progress=progress,
                        )
                        if prepared is None:
                            summary.files_skipped += 1
                            continue
                        prepared_batch.append(prepared)
                        prepared_chunks += len(prepared.chunks)
                        if (
                            len(prepared_batch) >= INDEX_FILE_BATCH_SIZE
                            or prepared_chunks >= INDEX_CHUNK_BATCH_SIZE
                        ):
                            flush_batch()
                    except Exception as exc:
                        root_errors += 1
                        summary.errors += 1
                        status = "done_with_errors"
                        self._record_file_error(record, exc, progress)

                flush_batch()

                if summary.cancelled or (should_cancel and should_cancel()):
                    summary.cancelled = True
                    status = "cancelled"
                    break

                if traversal_errors:
                    status = "done_with_errors"
                    root_errors += len(traversal_errors)
                    summary.errors += len(traversal_errors)
                    if progress:
                        progress("部分目录无权访问，已保留这些目录原有的索引。")
                else:
                    removed = self.db.remove_missing_files(root_path, root_seen)
                    summary.files_removed += removed
                    if removed and progress:
                        progress(f"移除 {removed} 个已不存在文件的索引。")
                self.db.mark_folder_scanned(root_path)
            except Exception:
                status = "failed"
                raise
            finally:
                self.db.finish_scan_job(
                    job_id,
                    status=status,
                    files_seen=root_seen_count,
                    files_indexed=root_indexed,
                    errors=root_errors,
                )

        self.db.set_setting("ocr_runtime_status", self.ocr_service.readiness_status)
        self.db.set_setting("ocr_load_error", self.ocr_service.load_error or "")
        if self.maintain_vector_index:
            self.rebuild_vector_index()
        return summary

    def index_path(
        self,
        path: str | Path,
        *,
        include_embeddings: bool = True,
        force_recheck: bool = True,
        progress: ProgressCallback | None = None,
    ) -> bool:
        record = build_file_record(path)
        if record is None:
            return False
        model_name = self._prepare_embedding_model(progress) if include_embeddings else None
        changed = self.index_file(
            record,
            include_embeddings=include_embeddings,
            embedding_model_name=model_name,
            force_recheck=force_recheck,
            progress=progress,
        )
        if changed and self.maintain_vector_index:
            self.rebuild_vector_index()
        return changed

    def index_paths(
        self,
        paths: list[str | Path],
        *,
        include_embeddings: bool = True,
        force_recheck: bool = True,
        progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScanSummary:
        summary = ScanSummary()
        if should_cancel and should_cancel():
            summary.cancelled = True
            return summary
        embedding_model = self._prepare_embedding_model(progress) if include_embeddings else None
        prepared_batch: list[PreparedFileIndex] = []
        prepared_chunks = 0

        def flush_batch() -> None:
            nonlocal prepared_batch, prepared_chunks
            if not prepared_batch:
                return
            if should_cancel and should_cancel():
                summary.cancelled = True
                prepared_batch = []
                prepared_chunks = 0
                return
            pending = prepared_batch
            indexed, failures = self._commit_prepared_batch(
                pending,
                include_embeddings=include_embeddings,
                embedding_model_name=embedding_model,
                progress=progress,
                should_cancel=should_cancel,
            )
            summary.files_indexed += indexed
            if should_cancel and should_cancel():
                summary.cancelled = True
            failed_paths = {record.path for record, _ in failures}
            if progress and not summary.cancelled:
                for item in pending:
                    if item.record.path not in failed_paths:
                        progress(f"更新索引：{item.record.path}")
            for record, error in failures:
                summary.errors += 1
                self._record_file_error(record, error, progress)
            prepared_batch = []
            prepared_chunks = 0

        for path in paths:
            if should_cancel and should_cancel():
                summary.cancelled = True
                break
            record = build_file_record(path)
            if record is None:
                summary.files_skipped += 1
                continue
            summary.files_seen += 1
            try:
                prepared = self._prepare_file(
                    record,
                    include_embeddings=include_embeddings,
                    embedding_model_name=embedding_model,
                    force_recheck=force_recheck,
                    progress=progress,
                )
                if prepared is None:
                    summary.files_skipped += 1
                    continue
                prepared_batch.append(prepared)
                prepared_chunks += len(prepared.chunks)
                if (
                    len(prepared_batch) >= INDEX_FILE_BATCH_SIZE
                    or prepared_chunks >= INDEX_CHUNK_BATCH_SIZE
                ):
                    flush_batch()
            except Exception as exc:
                summary.errors += 1
                self._record_file_error(record, exc, progress)
        flush_batch()
        self.db.set_setting("ocr_runtime_status", self.ocr_service.readiness_status)
        self.db.set_setting("ocr_load_error", self.ocr_service.load_error or "")
        if self.maintain_vector_index and summary.files_indexed:
            self.rebuild_vector_index()
        return summary

    def index_file(
        self,
        record: FileRecord,
        *,
        include_embeddings: bool = True,
        embedding_model_name: str | None = None,
        force_recheck: bool = False,
        progress: ProgressCallback | None = None,
    ) -> bool:
        prepared = self._prepare_file(
            record,
            include_embeddings=include_embeddings,
            embedding_model_name=embedding_model_name,
            force_recheck=force_recheck,
            progress=progress,
        )
        if prepared is None:
            return False
        indexed, failures = self._commit_prepared_batch(
            [prepared],
            include_embeddings=include_embeddings,
            embedding_model_name=embedding_model_name,
            progress=progress,
        )
        if failures:
            raise failures[0][1]
        return bool(indexed)

    def _prepare_file(
        self,
        record: FileRecord,
        *,
        include_embeddings: bool,
        embedding_model_name: str | None,
        force_recheck: bool = False,
        progress: ProgressCallback | None = None,
    ) -> PreparedFileIndex | None:
        previous = self.db.get_file_by_path(record.path)
        vectors_ready = bool(
            previous
            and embedding_model_name
            and self.db.file_has_vectors(int(previous["id"]), embedding_model_name)
        )
        metadata_unchanged = bool(
            previous
            and int(previous["size"]) == record.size
            and str(previous["modified_at"]) == record.modified_at
        )
        index_complete = bool(
            previous
            and previous["parse_status"] in {"parsed", "empty"}
            and previous["text_index_version"] == TEXT_INDEX_VERSION
            and (not self.ocr_service.enabled or previous["ocr_status"] in {"done", "empty", "not_required"})
            and (not include_embeddings or vectors_ready or previous["parse_status"] == "empty")
        )
        if metadata_unchanged and index_complete and not force_recheck:
            return None

        record.file_hash = hash_file(Path(record.path))
        same_content = bool(
            previous
            and previous["file_hash"] is not None
            and previous["file_hash"] == record.file_hash
            and previous["parse_status"] in {"parsed", "empty"}
        )
        if same_content and index_complete:
            self.db.update_file_metadata(record)
            return None

        if same_content and (not self.ocr_service.enabled or previous["ocr_status"] in {"done", "empty", "not_required"}):
            if progress:
                progress(f"复用已解析文字并更新语义分块：{record.name}")
            rows = self.db.conn.execute("SELECT * FROM chunks WHERE file_id=? ORDER BY id", (previous["id"],))
            chunks = [TextChunk(**{key: row[key] for key in TextChunk.__dataclass_fields__}) for row in rows]
            return PreparedFileIndex(record, rechunk(chunks), previous["parse_status"], previous["ocr_status"], True)

        if progress:
            progress(f"解析：{record.name}")
        chunks, parse_status, ocr_status = self.parser.parse(record.path)
        if self.ocr_service.enabled and ocr_status in {"failed", "skipped"}:
            raise RuntimeError(self.ocr_service.load_error or f"OCR 未完成：{ocr_status}")
        latest = build_file_record(record.path)
        if latest is None or (latest.size, latest.modified_at) != (record.size, record.modified_at):
            raise RuntimeError("文件在解析期间发生变化，请稍后重试")
        chunks = [chunk for chunk in chunks if chunk.chunk_text.strip()]
        if not chunks:
            parse_status = "empty"
        return PreparedFileIndex(record, chunks, parse_status, ocr_status)

    def _commit_prepared_batch(
        self,
        prepared: list[PreparedFileIndex],
        *,
        include_embeddings: bool,
        embedding_model_name: str | None,
        progress: ProgressCallback | None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[int, list[tuple[FileRecord, Exception]]]:
        if not prepared:
            return 0, []
        all_chunks = [chunk for item in prepared for chunk in item.chunks]
        embeddings: np.ndarray | None = None
        active_model: str | None = None
        if include_embeddings and all_chunks:
            if progress:
                progress(f"批量生成向量：{len(prepared)} 个文件，{len(all_chunks)} 个文本块")
            try:
                embeddings = self.embedding_service.encode_documents(
                    [chunk.chunk_text for chunk in all_chunks]
                )
            except Exception as exc:
                return 0, [(item.record, exc) for item in prepared]
            active_model = self.embedding_service.active_model_name
            self.db.set_setting("embedding_device_active", getattr(self.embedding_service, "device", "cpu"))
            self.db.set_setting("embedding_runtime_note", getattr(self.embedding_service, "runtime_note", ""))
            if embedding_model_name and active_model != embedding_model_name:
                error = RuntimeError(
                    f"Embedding 模型在索引过程中发生变化：{embedding_model_name} -> {active_model}"
                )
                return 0, [(item.record, error) for item in prepared]

        indexed = 0
        failures: list[tuple[FileRecord, Exception]] = []
        offset = 0
        for item in prepared:
            if should_cancel and should_cancel():
                break
            chunk_count = len(item.chunks)
            item_embeddings = None
            if embeddings is not None and chunk_count:
                item_embeddings = embeddings[offset : offset + chunk_count]
            offset += chunk_count
            try:
                latest = build_file_record(item.record.path)
                if latest is None or (latest.size, latest.modified_at) != (item.record.size, item.record.modified_at):
                    raise RuntimeError("文件在生成向量期间发生变化，请稍后重试")
                self.db.replace_file_index(
                    item.record,
                    item.chunks,
                    parse_status=item.parse_status,
                    ocr_status=item.ocr_status,
                    embeddings=item_embeddings,
                    model_name=active_model,
                    preserve_ocr_results=item.reused_text,
                )
                indexed += 1
            except Exception as exc:
                failures.append((item.record, exc))
        return indexed, failures

    def _record_file_error(
        self,
        record: FileRecord,
        error: Exception,
        progress: ProgressCallback | None,
    ) -> None:
        file_id = self.db.upsert_file(record)
        self.db.update_file_status(file_id, parse_status="failed", error=str(error))
        if progress:
            progress(f"失败：{record.path} ({error})")

    def remove_path(self, path: str | Path) -> bool:
        removed = self.db.delete_file_by_path(path)
        if removed and self.maintain_vector_index:
            self.rebuild_vector_index()
        return removed

    def _prepare_embedding_model(self, progress: ProgressCallback | None) -> str:
        if progress:
            progress("检查本地语义模型...")
        active_model = self.embedding_service.active_model_name
        indexed_models = self.db.vector_models()
        load_error = self.embedding_service.load_error
        if load_error and indexed_models and active_model not in indexed_models:
            self.db.set_setting("embedding_load_error", load_error)
            raise RuntimeError(
                "语义模型加载失败，已保留现有向量索引；请检查模型缓存或网络后重试。"
            )
        if indexed_models and set(indexed_models) != {active_model}:
            self.db.clear_vectors()
            if progress:
                progress("Embedding 模型已变化，正在为文件重建语义向量。")
        self.db.set_setting("embedding_model", active_model)
        self.db.set_setting("embedding_load_error", load_error or "")
        return active_model
