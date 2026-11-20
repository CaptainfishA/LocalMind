from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import SUPPORTED_EXTENSIONS
from .db import LocalMindDB
from .embeddings import EmbeddingService
from .indexer import IndexingService
from .ocr import OcrService


@dataclass(frozen=True, slots=True)
class FileEvent:
    event_type: str
    src_path: str
    dest_path: str | None = None


@dataclass(slots=True)
class IncrementalUpdateSummary:
    indexed: int = 0
    removed: int = 0
    skipped: int = 0
    errors: int = 0


def is_supported_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def apply_file_events(
    db_path: str | Path,
    events: list[FileEvent],
    *,
    enable_ocr: bool = True,
    enable_embeddings: bool = True,
    progress: Callable[[str], None] | None = None,
    embedding_service: EmbeddingService | None = None,
    ocr_service: OcrService | None = None,
) -> IncrementalUpdateSummary:
    summary = IncrementalUpdateSummary()
    deletes: set[str] = set()
    upserts: set[str] = set()

    for event in events:
        if event.event_type == "moved":
            if is_supported_path(event.src_path):
                deletes.add(event.src_path)
            if event.dest_path and is_supported_path(event.dest_path):
                upserts.add(event.dest_path)
        elif event.event_type == "deleted":
            if is_supported_path(event.src_path):
                deletes.add(event.src_path)
        elif event.event_type in {"created", "modified"} and is_supported_path(event.src_path):
            upserts.add(event.src_path)

    # Atomic saves can delete and recreate the same path within one event batch.
    for path in list(deletes):
        if Path(path).is_file():
            deletes.remove(path)
            upserts.add(path)
    upserts.difference_update(deletes)
    db = LocalMindDB(db_path)
    try:
        indexer = IndexingService(
            db,
            embedding_service=embedding_service or EmbeddingService(),
            ocr_service=ocr_service or OcrService(enabled=enable_ocr),
        )
        for path in sorted(deletes):
            if indexer.remove_path(path):
                summary.removed += 1
                if progress:
                    progress(f"监控移除：{path}")
        existing_upserts: list[str] = []
        for path in sorted(upserts):
            if Path(path).exists():
                existing_upserts.append(path)
            elif indexer.remove_path(path):
                summary.removed += 1
        if existing_upserts:
            batch = indexer.index_paths(
                existing_upserts,
                include_embeddings=enable_embeddings,
                force_recheck=True,
                progress=progress,
            )
            summary.indexed += batch.files_indexed
            summary.skipped += batch.files_skipped
            summary.errors += batch.errors
    finally:
        db.close()
    return summary


class FileChangeMonitor:
    def __init__(
        self,
        roots: list[str | Path],
        callback: Callable[[list[FileEvent]], None],
        *,
        debounce_seconds: float = 1.5,
    ) -> None:
        self.roots = [str(Path(root)) for root in roots]
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._observer = None
        self._queue: queue.Queue[FileEvent] = queue.Queue()
        self._stop_event = threading.Event()
        self._dispatch_thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return bool(self._observer and self._observer.is_alive())

    def start(self) -> bool:
        if self.running:
            return True
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception as exc:
            self.error = f"watchdog 不可用：{exc}"
            return False

        monitor = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event) -> None:
                if not event.is_directory:
                    monitor._queue.put(FileEvent("created", event.src_path))

            def on_modified(self, event) -> None:
                if not event.is_directory:
                    monitor._queue.put(FileEvent("modified", event.src_path))

            def on_deleted(self, event) -> None:
                if not event.is_directory:
                    monitor._queue.put(FileEvent("deleted", event.src_path))

            def on_moved(self, event) -> None:
                if not event.is_directory:
                    monitor._queue.put(FileEvent("moved", event.src_path, event.dest_path))

        observer = Observer()
        scheduled = 0
        handler = Handler()
        for root in self.roots:
            if Path(root).is_dir():
                observer.schedule(handler, root, recursive=True)
                scheduled += 1
        if not scheduled:
            self.error = "没有可监控的有效文件夹"
            return False

        self._stop_event.clear()
        observer.start()
        self._observer = observer
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="LocalMindFileMonitor",
            daemon=True,
        )
        self._dispatch_thread.start()
        self.error = None
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5)
            self._dispatch_thread = None

    def _dispatch_loop(self) -> None:
        pending: list[FileEvent] = []
        last_event_at = 0.0
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.25)
                pending.append(event)
                last_event_at = time.monotonic()
            except queue.Empty:
                pass
            if pending and time.monotonic() - last_event_at >= self.debounce_seconds:
                batch = _dedupe_events(pending)
                pending = []
                self.callback(batch)


def _dedupe_events(events: list[FileEvent]) -> list[FileEvent]:
    latest: dict[tuple[str, str, str | None], FileEvent] = {}
    for event in events:
        key = (event.event_type, event.src_path.lower(), (event.dest_path or "").lower() or None)
        latest[key] = event
    return list(latest.values())
