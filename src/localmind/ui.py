from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from threading import Event
from pathlib import Path

from .cache_manager import (
    clear_model_cache,
    format_bytes,
    model_cache_stats,
    prepare_custom_cache_dir,
    prepare_model_cache_dir,
)
from .db import LocalMindDB
from .embeddings import EmbeddingService
from .indexer import IndexingService, ScanSummary
from .index_notifications import IndexRevisionProbe
from .models import SearchQuery, SearchResult
from .monitor import FileChangeMonitor, FileEvent, IncrementalUpdateSummary, apply_file_events
from .ocr import OcrService
from .reranker import RerankerService
from .runtime import detect_runtime_capabilities, resolve_reranker_runtime
from .searcher import SearchCancelled, SearchDiagnostics, SearchService
from .chat_ui import WeFlowPanel
from .ui_formatting import display_time, plain_tooltip, result_source_label


def _import_qt():
    try:
        from PySide6.QtCore import QEvent, QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot
        from PySide6.QtGui import QCloseEvent, QGuiApplication
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QButtonGroup,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFrame,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSplitter,
            QStyle,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        return locals()
    except Exception as exc:  # pragma: no cover - UI dependency
        raise RuntimeError("PySide6 is required to launch the desktop UI. Install dependencies first.") from exc


qt = _import_qt()
QObject = qt["QObject"]
QEvent = qt["QEvent"]
Qt = qt["Qt"]
QThread = qt["QThread"]
QTimer = qt["QTimer"]
QMetaObject = qt["QMetaObject"]
Signal = qt["Signal"]
Slot = qt["Slot"]
QCloseEvent = qt["QCloseEvent"]
QGuiApplication = qt["QGuiApplication"]
QAbstractItemView = qt["QAbstractItemView"]
QApplication = qt["QApplication"]
QButtonGroup = qt["QButtonGroup"]
QCheckBox = qt["QCheckBox"]
QComboBox = qt["QComboBox"]
QFileDialog = qt["QFileDialog"]
QFrame = qt["QFrame"]
QHBoxLayout = qt["QHBoxLayout"]
QHeaderView = qt["QHeaderView"]
QLabel = qt["QLabel"]
QLineEdit = qt["QLineEdit"]
QListWidget = qt["QListWidget"]
QMainWindow = qt["QMainWindow"]
QMessageBox = qt["QMessageBox"]
QPlainTextEdit = qt["QPlainTextEdit"]
QProgressBar = qt["QProgressBar"]
QPushButton = qt["QPushButton"]
QScrollArea = qt["QScrollArea"]
QSplitter = qt["QSplitter"]
QStyle = qt["QStyle"]
QTabWidget = qt["QTabWidget"]
QTableWidget = qt["QTableWidget"]
QTableWidgetItem = qt["QTableWidgetItem"]
QTextEdit = qt["QTextEdit"]
QVBoxLayout = qt["QVBoxLayout"]
QWidget = qt["QWidget"]


from . import __version__
from .desktop.theme import APP_STYLESHEET
from .desktop.views import DesktopViews
from .desktop.widgets import icon, result_icon
from .desktop.detail_text import context_text, evidence_text


class SearchWorker(QObject):
    finished = Signal(object, object, object)
    failed = Signal(str)
    cancelled = Signal(float)
    topics_finished = Signal(object, object, str)

    def __init__(self, db_path: str, cache_dir: str, reranker_runtime: str, embedding_device: str = "auto") -> None:
        super().__init__()
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.reranker_runtime = reranker_runtime
        self.embedding_service = EmbeddingService(cache_dir=cache_dir, device=embedding_device)
        self.reranker_service = RerankerService(
            cache_dir=cache_dir,
            runtime=reranker_runtime,
            enabled=False,
        )
        self.db: LocalMindDB | None = None
        self.service: SearchService | None = None
        self.index_dirty = True
        self.cancel_event = Event()
        from .chat_topics import TopicOrganizer
        self.topic_organizer = TopicOrganizer()

    @Slot(object)
    def configure(self, payload: dict[str, str]) -> None:
        self.topic_organizer.clear()
        self.cache_dir = payload["cache_dir"]
        self.reranker_runtime = payload["reranker_runtime"]
        if self.db is not None:
            self.db.close()
        self.db = None
        self.service = None
        self.embedding_service = EmbeddingService(cache_dir=self.cache_dir, device=payload.get("embedding_device", "auto"))
        self.reranker_service = RerankerService(
            cache_dir=self.cache_dir,
            runtime=self.reranker_runtime,
            enabled=False,
        )
        self.index_dirty = True

    @Slot()
    def mark_index_dirty(self) -> None:
        self.topic_organizer.clear()
        self.index_dirty = True
        if self.service is not None:
            self.service.clear_refinement_cache()

    @Slot()
    def close(self) -> None:
        self.topic_organizer.clear()
        if self.db is not None:
            self.db.close()
        self.db = None
        self.service = None

    @Slot(object)
    def organize_topics(self, request):
        if request["cancel"].is_set():
            self.topics_finished.emit(request, None, "主题整理已取消")
            return
        try:
            data, notice = self.topic_organizer.organize(self.db_path, request["result"], request["groups"],
                self.embedding_service, excluded=request["excluded"], cancelled=request["cancel"].is_set)
        except Exception:
            data, notice = None, "主题整理未完成，保留原分组；可检查本地模型后重试"
        self.topics_finished.emit(request, data, notice)

    @Slot(object)
    def process(self, payload: dict[str, object]) -> None:
        started = time.perf_counter()
        refresh_ms = 0.0
        try:
            if self.cancel_event.is_set():
                raise SearchCancelled()
            mode = payload.get("retrieval_mode", "hybrid")
            if mode not in {"hybrid", "semantic", "keyword"}:
                raise ValueError("无效的检索模式")
            self.reranker_service.enabled = bool(payload["enable_reranker"])
            refresh_started = time.perf_counter()
            reused_service = self.service is not None
            if self.db is None:
                self.db = LocalMindDB(self.db_path)
            if self.service is None:
                self.service = SearchService(
                    self.db,
                    embedding_service=self.embedding_service,
                    reranker_service=self.reranker_service,
                    should_cancel=self.cancel_event.is_set,
                )
            elif self.index_dirty:
                self.service.refresh_vector_index(force=False)
            if self.index_dirty:
                refresh_ms = (time.perf_counter() - refresh_started) * 1000
            self.index_dirty = False
            self.service.retrieval_mode = mode
            search_started = time.perf_counter()
            results = self.service.search(payload["query"])  # type: ignore[arg-type]
            if self.cancel_event.is_set():
                raise SearchCancelled()
            search_ms = (time.perf_counter() - search_started) * 1000
            timings = {
                "total_ms": (time.perf_counter() - started) * 1000,
                "search_ms": search_ms,
                "index_refresh_ms": refresh_ms,
                "service_reused": reused_service,
                "index_revision": payload.get("index_revision", 0),
            }
            self.finished.emit(results, self.service.last_diagnostics, timings)
        except SearchCancelled:
            self.cancelled.emit((time.perf_counter() - started) * 1000)
        except Exception as exc:
            if self.cancel_event.is_set():
                self.cancelled.emit((time.perf_counter() - started) * 1000)
            else:
                self.failed.emit(str(exc))


class ScanWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        db_path: str,
        folders: list[str],
        enable_ocr: bool,
        enable_embeddings: bool,
        cache_dir: str,
        embedding_device: str = "auto",
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.folders = folders
        self.enable_ocr = enable_ocr
        self.enable_embeddings = enable_embeddings
        self.cache_dir = cache_dir
        self.embedding_device = embedding_device
        self.cancel_event = Event()

    @Slot()
    def run(self) -> None:
        db = None
        try:
            db = LocalMindDB(self.db_path)
            indexer = IndexingService(
                db,
                embedding_service=EmbeddingService(cache_dir=self.cache_dir, device=self.embedding_device),
                ocr_service=OcrService(enabled=self.enable_ocr, cache_dir=self.cache_dir),
            )
            summary = indexer.scan_and_index(
                self.folders,
                include_embeddings=self.enable_embeddings,
                progress=self.progress.emit,
                should_cancel=self.cancel_event.is_set,
            )
            self.finished.emit(summary)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if db is not None:
                db.close()


class IncrementalWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)

    def __init__(self, db_path: str, cache_dir: str, embedding_device: str = "auto") -> None:
        super().__init__()
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.embedding_service = EmbeddingService(cache_dir=cache_dir, device=embedding_device)
        self.ocr_service = OcrService(cache_dir=cache_dir)

    @Slot(object)
    def configure(self, payload: dict[str, str]) -> None:
        self.cache_dir = payload["cache_dir"]
        self.embedding_service = EmbeddingService(cache_dir=self.cache_dir, device=payload.get("embedding_device", "auto"))
        self.ocr_service = OcrService(cache_dir=self.cache_dir)

    @Slot(object)
    def process(self, payload: dict[str, object]) -> None:
        try:
            self.ocr_service.enabled = bool(payload["ocr"])
            summary = apply_file_events(
                self.db_path,
                payload["events"],  # type: ignore[arg-type]
                enable_ocr=bool(payload["ocr"]),
                enable_embeddings=bool(payload["embeddings"]),
                progress=self.progress.emit,
                embedding_service=self.embedding_service,
                ocr_service=self.ocr_service,
            )
        except Exception as exc:
            summary = IncrementalUpdateSummary(errors=1)
            self.progress.emit(f"增量更新失败：{exc}")
        self.finished.emit(summary)


class OcrCheckWorker(QObject):
    finished = Signal(str, str, float)

    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        self.cache_dir = cache_dir

    @Slot()
    def run(self) -> None:
        service = OcrService(cache_dir=self.cache_dir)
        service.check_ready()
        self.finished.emit(
            service.readiness_status,
            service.load_error or "",
            service.load_seconds,
        )


@dataclass(frozen=True, slots=True)
class _SearchRequest:
    query: SearchQuery
    mode: str
    reranker: bool


class MainWindow(DesktopViews, QMainWindow):
    monitor_events = Signal(object)
    search_requested = Signal(object)
    topics_requested = Signal(object)
    search_index_dirty = Signal()
    search_configure = Signal(object)
    incremental_requested = Signal(object)
    incremental_configure = Signal(object)

    def __init__(self, db: LocalMindDB) -> None:
        super().__init__()
        if QApplication.instance().style().objectName().lower() != "fusion":
            QApplication.setStyle("Fusion")
        self.db = db
        self._topic_jobs = {}
        self.runtime_capabilities = detect_runtime_capabilities()
        self.cache_dir = prepare_model_cache_dir(self.db.get_setting("model_cache_dir"))
        self.db.set_setting("model_cache_dir", str(self.cache_dir))
        self.reranker_runtime = self.db.get_setting("reranker_runtime", "auto") or "auto"
        self.embedding_device = self.db.get_setting("embedding_device", "auto") or "auto"
        if self.embedding_device not in {"auto", "cuda", "cpu"}:
            self.embedding_device = "auto"
        self.pending_file_events: list[FileEvent] = []
        self.monitor_events.connect(self._queue_file_events)
        if self.db.get_setting("reranker_enabled") is None:
            self.db.set_setting(
                "reranker_enabled",
                "1" if self.runtime_capabilities.default_reranker_enabled else "0",
            )
        self.results: list[SearchResult] = []
        self.search_busy = False
        self._search_cancel_requested = False
        self._search_started_at = 0.0
        self._prior_has_search = False
        self._index_revision = 0
        self._search_revision = 0
        self._has_search = False
        self._results_stale = False
        self._displayed_request: _SearchRequest | None = None
        self._pending_request: _SearchRequest | None = None
        self._preserved_selection = None
        self._results_db_revision: int | None = None
        self._last_index_check_revision: int | None = None
        self._index_watch_error = False
        self._closing = False
        self.index_revision_probe = IndexRevisionProbe(self.db.db_path)
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.file_monitor: FileChangeMonitor | None = None
        self.incremental_busy = False
        self.incremental_jobs = 0
        self.ocr_check_thread: QThread | None = None
        self.ocr_check_worker: OcrCheckWorker | None = None

        self.search_thread = QThread(self)
        self.search_worker = SearchWorker(
            str(self.db.db_path),
            str(self.cache_dir),
            self.reranker_runtime,
            self.embedding_device,
        )
        self.search_worker.moveToThread(self.search_thread)
        self.search_requested.connect(self.search_worker.process)
        self.topics_requested.connect(self.search_worker.organize_topics)
        self.topics_requested.connect(self._track_topics)
        self.search_worker.topics_finished.connect(self._topics_finished)
        self.search_index_dirty.connect(self.search_worker.mark_index_dirty)
        self.search_index_dirty.connect(self._mark_results_stale)
        self.search_configure.connect(self.search_worker.configure)
        self.search_worker.finished.connect(self._search_finished)
        self.search_worker.failed.connect(self._search_failed)
        self.search_worker.cancelled.connect(self._search_cancelled)
        self.search_thread.start()

        self.incremental_thread = QThread(self)
        self.incremental_worker = IncrementalWorker(str(self.db.db_path), str(self.cache_dir), self.embedding_device)
        self.incremental_worker.moveToThread(self.incremental_thread)
        self.incremental_requested.connect(self.incremental_worker.process)
        self.incremental_configure.connect(self.incremental_worker.configure)
        self.incremental_worker.progress.connect(self._append_monitor_log)
        self.incremental_worker.finished.connect(self._incremental_finished)
        self.incremental_thread.start()

        self.setWindowTitle(f"LocalMind {__version__}")
        self.setWindowIcon(icon("scan-search", "#237b63", 32))
        self.resize(1380, 860)
        self.setMinimumSize(980, 650)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_shell()
        self._load_watched_folders()
        self.refresh_stats()
        self._maybe_show_privacy_notice()
        self.index_watch_timer = QTimer(self)
        self.index_watch_timer.setInterval(2000)
        self.index_watch_timer.timeout.connect(self._poll_index_updates)
        self.index_watch_timer.start()
        QTimer.singleShot(0, self._startup_reconcile)


    def _maybe_show_privacy_notice(self) -> None:
        if self.db.get_setting("privacy_accepted_v02") == "1":
            return
        message = QMessageBox(self)
        message.setWindowTitle("LocalMind 隐私确认")
        message.setText("LocalMind 只处理你主动选择的文件夹，索引和 AI 模型均保存在本地。")
        message.setInformativeText("可实时监听所选目录的文件变化，启动补扫可在文件夹页关闭。原文件不会被修改。")
        message.setStandardButtons(QMessageBox.StandardButton.Ok)
        message.exec()
        self.db.set_setting("privacy_accepted_v02", "1")

    def _load_watched_folders(self) -> None:
        for folder in self.db.list_watched_folders():
            if Path(folder).is_dir():
                self.folder_list.addItem(folder)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择要索引的文件夹")
        if not folder:
            return
        existing = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        if folder not in existing:
            self.folder_list.addItem(folder)
            self.db.add_watched_folder(folder)
            self._sync_monitor()
            self.refresh_stats()

    def remove_folder(self) -> None:
        for item in self.folder_list.selectedItems():
            self.db.remove_watched_folder(item.text())
            self.folder_list.takeItem(self.folder_list.row(item))
        self._sync_monitor()
        self.refresh_stats()

    def _update_evidence_filter(self) -> None:
        active = self.scope_filter.currentData() == "chat_files"
        self.evidence_filter.setVisible(active)
        self.evidence_filter.setEnabled(active and not self.search_busy)

    def copy_chat_context(self) -> None:
        result = self.selected_result()
        if not result:
            return
        self.show_selected_result()
        if self.source_copy_button.isEnabled():
            QApplication.clipboard().setText(self.context_preview.toPlainText())

    @Slot()
    def _mark_results_stale(self) -> None:
        self._index_revision += 1
        if self._has_search:
            self._results_stale = True
            self._update_result_notice()
            self.show_selected_result()

    def _update_result_notice(self) -> None:
        if self._index_watch_error:
            text = "结果待刷新；暂时无法核对最新索引" if self._results_stale else "暂时无法核对索引更新"
        else:
            text = "索引已更新，检索结果待刷新"
        self.result_notice_label.setText(text)
        self.result_notice.setVisible(self._has_search and (self._results_stale or self._index_watch_error))
        self._update_result_controls()

    @Slot()
    def _poll_index_updates(self) -> None:
        if (self._closing or not self._has_search or self._results_db_revision is None
                or not self.isVisible() or self.isMinimized() or self.db.conn.in_transaction):
            return
        scroll_positions = None
        try:
            revision = self.index_revision_probe.read()
            if revision != self._results_db_revision and (
                    revision != self._last_index_check_revision or self._index_watch_error):
                self._results_stale = True
                scroll = self.preview.verticalScrollBar().value()
                old_context = self.context_preview.toPlainText()
                context_scroll = self.context_preview.verticalScrollBar().value()
                info_scroll = self.info_preview.verticalScrollBar().value()
                self.show_selected_result()
                scroll_positions = [(self.preview, scroll), (self.info_preview, info_scroll)]
                if old_context == self.context_preview.toPlainText():
                    scroll_positions.append((self.context_preview, context_scroll))
            self._last_index_check_revision = revision
            self._index_watch_error = False
        except (sqlite3.Error, OSError):
            self._index_watch_error = True
        # External commits do not invalidate a search already in flight. Its actual
        # snapshot revision is checked at delivery; no queued FAISS rebuild is needed.
        self._update_result_notice()
        # The notice can resize the previews, so restore positions after it changes.
        for preview, position in scroll_positions or ():
            scrollbar = preview.verticalScrollBar()
            scrollbar.setValue(position)
            # QPlainTextEdit can lay out wrapped blocks during the first scroll.
            if scrollbar.value() != position:
                scrollbar.setValue(position)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._poll_index_updates)
        QTimer.singleShot(0, self._resize_workspace)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if (event.type() in (QEvent.Type.ActivationChange, QEvent.Type.WindowStateChange)
                and getattr(self, "_has_search", False)):
            QTimer.singleShot(0, self._poll_index_updates)

    def _result_attachment_details(self, result):
        """Keep actions and refreshed context tied to the displayed search target."""
        if not result.attachment_id:
            return None, result.chat_context
        from .chats import ChatStore, chat_source_reference

        store = ChatStore(self.db)
        rows = store.attachments(attachment_id=result.attachment_id)
        attachment = rows[0] if rows else None
        if attachment is None or attachment["message_id"] != result.chat_message_id:
            return None, ""
        if result.chat_source and chat_source_reference(attachment).identity != result.chat_source.identity:
            return None, ""
        if result.result_kind == "file":
            if attachment["file_id"] != result.file_id or attachment["local_path"] != result.path:
                return None, ""
        elif result.result_kind == "attachment_pending" and attachment["file_id"] is not None:
            return None, ""
        context = store.attachment_context(
            result.attachment_id,
            file_id=result.file_id if result.result_kind == "file" else None,
            query=result.source_query,
            source=result.chat_source,
            file_path=result.path if result.result_kind == "file" else None,
        )
        return (attachment, context) if context else (None, "")

    def _context_result(self, result):
        """Switch only the context view; file actions retain the original result."""
        from .desktop.detail_text import evidence_sources, source_title

        sources = evidence_sources(result)
        selected = self.source_selector.currentData()
        current = [self.source_selector.itemData(i) for i in range(self.source_selector.count())]
        if current != sources:
            self.source_selector.blockSignals(True)
            try:
                self.source_selector.clear()
                for index, source in enumerate(sources):
                    title = source_title(source)
                    self.source_selector.addItem(f"{index + 1} · {title}", source)
                    self.source_selector.setItemData(index, plain_tooltip(title), Qt.ItemDataRole.ToolTipRole)
                chosen = next((i for i, source in enumerate(sources)
                               if selected and source.identity == selected.identity), 0)
                self.source_selector.setCurrentIndex(chosen if sources else -1)
            finally:
                self.source_selector.blockSignals(False)
        self.source_selector.setVisible(len(sources) > 1)
        selected = self.source_selector.currentData()
        self.source_selector.setToolTip(plain_tooltip(source_title(selected)) if selected else "")
        if selected is None or selected == result.chat_source:
            return result
        return replace(result, attachment_id=selected.attachment_id, chat_message_id=selected.message_id,
                       chat_source=selected, chat_context="")

    def _change_context_source(self, _index):
        self.show_selected_result()

    def _guard_attachment_action(self, result) -> bool:
        attachment, _ = self._result_attachment_details(result)
        if attachment is not None:
            return True
        self.search_index_dirty.emit()
        return False

    def show_source_history(self) -> None:
        from .chats import ChatStore
        from .source_history_ui import SourceHistoryDialog

        result = self.selected_result()
        if result and result.result_kind == "file" and result.attachment_id:
            context_result = self._context_result(result)
            dialog = SourceHistoryDialog(ChatStore(self.db), context_result, self)
            try:
                dialog.exec()
            finally:
                dialog.deleteLater()

    def _current_search_request(self, limit=20) -> _SearchRequest:
        query = SearchQuery(
            text=self.search_input.text().strip(),
            file_type=self.type_filter.currentData() or None, limit=limit,
            scope=self.scope_filter.currentData(),
            conversation=self.conversation_filter.text().strip() or None,
            sender=self.sender_filter.text().strip() or None,
            evidence=self.evidence_filter.currentData() if self.scope_filter.currentData() == "chat_files" else "all",
            infer_dates=self.date_mode_filter.currentData(),
        )
        mode = next((mode for mode, button in self.retrieval_buttons.items() if button.isChecked()), "hybrid")
        return _SearchRequest(query, mode, self.enable_reranker.isChecked())

    def _matching_search_controls(self) -> bool:
        request = self._displayed_request
        if request is None:
            return False
        # Relative dates are frozen from the accepted search, not recomputed on expansion.
        original_controls = replace(request, query=replace(request.query, modified_after=None, modified_before=None))
        return self._current_search_request(request.query.limit) == original_controls

    def _update_result_controls(self) -> None:
        if hasattr(self, "reset_filters_button"):
            self.reset_filters_button.setEnabled(not self.search_busy and not self.chat_panel.busy)
        request = self._displayed_request
        limit = request.query.limit if request else None
        count = f"{len(self.results)} 项"
        if self.results and all(r.result_kind == "message" for r in self.results):
            visible = sum(not self.result_table.isRowHidden(i) for i in range(len(self.results)))
            count = (f"{visible} 段讨论 · {len(self.results)} 条结果" if self.discussion_mode.currentData()
                     and getattr(self, "_discussion_groups", ()) else f"{len(self.results)} 条结果")
        self.result_count_label.setText(count + (f" · 上限 {limit}" if limit else ""))
        next_limit = next((value for value in (50, 100) if limit is not None and value > limit), None)
        reason = ""
        if request is None:
            reason = "尚无可扩大的查询"
        elif self.search_busy or self.chat_panel.busy:
            reason = "当前任务进行中"
        elif self._results_stale or self._index_watch_error:
            reason = "先刷新或核对当前索引状态"
        elif not self._matching_search_controls():
            reason = "查询或筛选已修改，请先搜索"
        elif next_limit is None:
            reason = "已达到本次结果上限 100；不表示所有相关文件都已找到"
        self.expand_results_button.setEnabled(not reason)
        self.expand_results_button.setToolTip(reason or
            f"按原查询与筛选将结果上限扩大至 {next_limit} 项；会重新召回和排序，可能没有新增结果")

    def expand_search(self) -> None:
        self._poll_index_updates()
        self._update_result_controls()
        if not self.expand_results_button.isEnabled():
            return
        request = self._displayed_request
        limit = next(value for value in (50, 100) if value > request.query.limit)
        self._start_search(replace(request, query=replace(request.query, limit=limit)), preserve_selection=True)

    def run_search(self) -> None:
        self._start_search(self._current_search_request())

    def refresh_search(self) -> None:
        same_query = self._matching_search_controls()
        limit = self._displayed_request.query.limit if same_query else 20
        self._start_search(self._current_search_request(limit), preserve_selection=same_query)

    def _start_search(self, request: _SearchRequest, *, preserve_selection=False) -> None:
        if not request.query.text or self.search_busy:
            return
        if self.chat_panel.busy:
            self.search_status.setText("微信索引任务进行中，搜索暂不可用。")
            return
        self._pending_request = request
        self._prior_has_search = self._has_search
        self._search_cancel_requested = False
        self._search_started_at = time.perf_counter()
        self.search_worker.cancel_event.clear()
        selected = self.selected_result()
        self._preserved_selection = self._result_identity(selected) if preserve_selection and selected else None
        self.db.set_setting("reranker_enabled", "1" if request.reranker else "0")
        search_query = request.query
        self.db.set_setting("search_infer_dates", "1" if search_query.infer_dates else "0")
        if search_query.scope == "chat_files":
            self.db.set_setting("chat_search_evidence", search_query.evidence)
        self.search_button.setEnabled(False)
        self.search_input.setEnabled(False)
        mode = request.mode
        self.db.set_setting("search_retrieval_mode", mode)
        self.enable_reranker.setEnabled(False)
        for button in self.retrieval_buttons.values():
            button.setEnabled(False)
        for widget in (self.scope_filter, self.evidence_filter, self.type_filter, self.date_mode_filter,
                       self.conversation_filter, self.sender_filter):
            widget.setEnabled(False)
        self.search_busy = True
        self.search_content.setCurrentWidget(self.search_splitter)
        self.cancel_search_button.setEnabled(True)
        self._has_search = True
        self._search_revision = self._index_revision
        self.refresh_results_button.setEnabled(False)
        self._update_result_controls()
        self.search_status.setText("正在检索与排序…")
        self.header_status.setText("搜索中")
        self.search_requested.emit(
            {"query": search_query, "enable_reranker": request.reranker, "retrieval_mode": mode,
             "index_revision": self._search_revision}
        )

    def cancel_search(self) -> None:
        if not self.search_busy or self._search_cancel_requested:
            return
        self._search_cancel_requested = True
        # Event.set is thread-safe; a queued Qt slot would wait behind the search.
        self.search_worker.cancel_event.set()
        self.cancel_search_button.setEnabled(False)
        self.search_status.setText("正在取消，等待当前处理步骤结束…")
        self.header_status.setText("取消中")

    def _search_cancelled(self, elapsed_ms: float) -> None:
        self._has_search = self._prior_has_search
        self.search_status.setText(f"搜索已取消 · 用时 {elapsed_ms:.0f} ms")
        self._update_result_notice()
        self._finish_search_ui()

    def _search_finished(
        self,
        results: list[SearchResult],
        diagnostics: SearchDiagnostics,
        timings: dict[str, object],
    ) -> None:
        from .file_freshness import STATE_LABELS, WARNING_STATES

        if self._search_cancel_requested:
            self._search_cancelled(float(timings["total_ms"]))
            return
        if timings.get("index_revision", self._search_revision) != self._index_revision:
            self.search_status.setText("搜索期间索引发生变化，本次结果未采用。")
            self._finish_search_ui()
            return
        self._has_search = True
        if self._pending_request is not None:
            request = self._pending_request
            self._displayed_request = replace(request, query=replace(
                request.query, modified_after=diagnostics.modified_after, modified_before=diagnostics.modified_before))
        else:
            self._displayed_request = None
        revision = self.db.index_revisions()[0]
        self._results_db_revision = diagnostics.index_revision if diagnostics.index_revision is not None else revision
        self._last_index_check_revision = revision
        self._results_stale = not diagnostics.index_current or (
            self._results_db_revision != revision)
        self._index_watch_error = False
        self._update_result_notice()
        self.results = results
        self._discussion_groups = ()
        self.result_table.blockSignals(True)
        self.result_table.setRowCount(len(results))
        for row_index, result in enumerate(results):
            self.result_table.setRowHidden(row_index, False)
            values = [
                result.name,
                result_source_label(result),
                "消息" if result.result_kind == "message" else result.ext.upper() or "未知",
                f"{result.score:.3f}",
                result.match_reason,
                display_time(result.modified_at),
                result.path if result.result_kind == "file" else "未定位" if result.attachment_id else "聊天记录",
            ]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                tooltip = value
                if col_index == 0:
                    item.setIcon(result_icon(result.ext, result.result_kind))
                    item.setData(Qt.ItemDataRole.UserRole, result.match_reason or values[6])
                    tooltip += "\n" + values[6]
                    if result.file_state in WARNING_STATES:
                        tooltip += "\n" + STATE_LABELS[result.file_state]
                elif col_index == 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    tooltip += "（排序分，非正确概率）"
                elif col_index == 5:
                    tooltip = ("文件修改时间：" if result.result_kind == "file" else "消息发送时间：") + (result.modified_at or "未知")
                item.setToolTip(plain_tooltip(tooltip))
                self.result_table.setItem(row_index, col_index, item)
        self.search_content.setCurrentWidget(self.search_splitter)
        self.result_table.empty_title = "暂无匹配结果"
        if results:
            selected_row = next((i for i, result in enumerate(results)
                                 if self._result_identity(result) == self._preserved_selection), 0)
            self.result_table.selectRow(selected_row)
            self.result_table.blockSignals(False)
            self.result_table.scrollToItem(self.result_table.item(selected_row, 0))
            self.show_selected_result()
        else:
            self.result_table.blockSignals(False)
            self._clear_result_details("没有找到结果。")
        timing_text = f"耗时 {float(timings['total_ms']):.0f} ms"
        self.search_status.setToolTip(diagnostics.semantic_status + "\n" + "\n".join(
            f"{name}: {value:.1f} ms" for name, value in diagnostics.timings_ms.items()
        ))
        self.search_status.setToolTip(self.search_status.toolTip()
                                     + f"\n数据库读取区间：{diagnostics.snapshot_hold_ms:.1f} ms"
                                     + ("（外层事务仍由调用方管理）" if diagnostics.caller_transaction else ""))
        if diagnostics.clause_status:
            self.search_status.setToolTip(self.search_status.toolTip() + "\n" + diagnostics.clause_status)
        if diagnostics.refinement_status:
            self.search_status.setToolTip(self.search_status.toolTip() + "\n" + diagnostics.refinement_status
                + f"（检查 {diagnostics.refinement_selected} 段，新编码 {diagnostics.refinement_encoded}，"
                  f"缓存 {diagnostics.refinement_cache_hits}，细化 {diagnostics.refinement_improved} 个候选）")
        if diagnostics.companion_status:
            self.search_status.setToolTip(self.search_status.toolTip() + "\n" + diagnostics.companion_status
                                         + f"（{diagnostics.companion_checked_files}/{diagnostics.companion_requested_files} 个文件）")
            timing_text += f"｜补充 {diagnostics.companion_candidates} 段"
            if diagnostics.companion_checked_files < diagnostics.companion_requested_files:
                timing_text += "（互补未完成）"
        if float(timings["index_refresh_ms"]) >= 1:
            timing_text += f"（索引刷新 {float(timings['index_refresh_ms']):.0f} ms）"
        warning_count = sum(result.file_state in WARNING_STATES for result in results)
        if warning_count:
            timing_text += f"｜索引待核对 {warning_count} 项"
        mode_label = {"hybrid": "综合", "semantic": "语义", "keyword": "关键词"}[diagnostics.retrieval_mode]
        if diagnostics.retrieval_mode == "hybrid" and diagnostics.ranking_profile == "descriptive":
            mode_label += " · 描述优先"
        if diagnostics.clause_count:
            mode_label += " · 分句语义"
        if diagnostics.retrieval_mode != "keyword" and not diagnostics.semantic_candidates:
            mode_label += " · 无语义结果"
        if diagnostics.scope == "chat_files":
            mode_label += "｜" + {"all": "全部线索", "content": "文件正文", "chat": "聊天线索"}[diagnostics.evidence]
        diagnostic_summary = (
            f"{mode_label}｜候选 {diagnostics.total_candidates} 个｜关键词 {diagnostics.keyword_candidates}｜"
            f"语义 {diagnostics.semantic_candidates}｜精排：{diagnostics.reranker_status}｜{timing_text}"
        )
        self.search_status.setToolTip(diagnostic_summary + "\n" + self.search_status.toolTip())
        if diagnostics.identifier:
            timing_text += (f"｜编号完整命中 {diagnostics.identifier_matches} 项" if diagnostics.identifier_matches
                            else "｜候选中未见完整编号")
            self.search_status.setToolTip(self.search_status.toolTip()
                + f"\n编号线索：{diagnostics.identifier}；仅核对本次候选的文件名与正文/聊天片段，不含会话/发送者字段，不是全库不存在的证明。"
                + ("综合/关键词按完整编号优先，组内按排序分。" if diagnostics.retrieval_mode != "semantic"
                   else "语义模式仍按语义分排序。"))
        self.search_status.setText(f"{len(results)} 项结果  ·  {mode_label}  ·  {timing_text}")
        bounds = []
        if diagnostics.modified_after:
            bounds.append(display_time(diagnostics.modified_after) + " 起")
        if diagnostics.modified_before:
            bounds.append(display_time(diagnostics.modified_before) + " 前")
        applied_query = self._displayed_request.query if self._displayed_request else None
        chat_time = diagnostics.scope in {"chat_files", "messages"} or bool(applied_query and (
            applied_query.chat_platform or applied_query.conversation or applied_query.sender))
        time_label = "发送时间" if chat_time else "修改时间" if diagnostics.scope == "files" else "检索时间"
        self.search_time_status.setText(time_label + "：" + ("，".join(bounds) if bounds else "不限"))
        self.search_time_status.setToolTip(
            "已显示结果的时间范围。起点包含，终点不包含，按本机时区展示。\n"
            "微信消息/附件或按聊天身份筛选时使用发送时间；普通本地文件使用修改时间。")
        self.search_time_status.show()
        self._finish_search_ui()

    def _search_failed(self, error: str) -> None:
        if self._search_cancel_requested:
            self._search_cancelled((time.perf_counter() - self._search_started_at) * 1000)
            return
        if self._search_revision != self._index_revision:
            self.search_status.setText("搜索期间索引发生变化，本次结果未采用。")
            self._finish_search_ui()
            return
        self.search_status.setText(f"搜索失败：{error}")
        self._finish_search_ui()
        QMessageBox.critical(self, "搜索失败", error)

    def _finish_search_ui(self) -> None:
        self.search_busy = False
        self._search_cancel_requested = False
        self.cancel_search_button.setEnabled(False)
        self._pending_request = None
        self._preserved_selection = None
        self.enable_reranker.setEnabled(True)
        self.refresh_results_button.setEnabled(True)
        self.search_button.setEnabled(True)
        self.search_input.setEnabled(True)
        for button in self.retrieval_buttons.values():
            button.setEnabled(True)
        for widget in (self.scope_filter, self.type_filter, self.date_mode_filter,
                       self.conversation_filter, self.sender_filter):
            widget.setEnabled(True)
        self._update_evidence_filter()
        self._update_result_controls()
        self.search_input.setFocus()
        self.header_status.setText("本地模式")

    @staticmethod
    def _result_identity(result):
        if result.result_kind == "file":
            return result.result_kind, result.file_id, result.path
        return result.result_kind, result.chat_message_id, result.attachment_id, result.path

    def show_selected_result(self) -> None:
        from .file_freshness import STATE_LABELS, WARNING_STATES

        result = self.selected_result()
        if not result:
            self._clear_result_details()
            return
        identity = (result.file_id, result.chunk_id, result.attachment_id,
                    result.chat_source.identity if result.chat_source else None)
        detail_changed = getattr(self, "_detail_identity", None) != identity
        if detail_changed:
            self._detail_identity = identity
            self.source_selector.blockSignals(True)
            self.source_selector.clear()
            self.source_selector.blockSignals(False)
            for toggle in (self.evidence_expand, self.context_expand, self.technical_toggle):
                toggle.setChecked(False)
            self.info_preview.hide()
            self.file_summary.show()
            self.technical_toggle.setText("检索技术详情")
        self.detail_title.setText(result.name)
        self.detail_title.setToolTip(plain_tooltip(result.name))
        if result.identifier_query:
            state = "编号完整命中" if result.identifier_matched else "未见完整编号"
            self.identifier_notice.setText(f"{state} · {result.identifier_query}")
            self.identifier_notice.setToolTip(plain_tooltip(self.identifier_notice.text()
                + "\n仅核对文件名与正文/聊天片段，不含会话/发送者字段；不代表满足全部条件，也不证明全库不存在。"))
            self.identifier_notice.setProperty("state", "matched" if result.identifier_matched else "partial")
            self.identifier_notice.style().unpolish(self.identifier_notice)
            self.identifier_notice.style().polish(self.identifier_notice)
            self.identifier_notice.show()
        else:
            self.identifier_notice.clear()
            self.identifier_notice.setToolTip("")
            self.identifier_notice.hide()
        attachment, context = self._result_attachment_details(result)
        attachment_valid = attachment is not None
        if result.attachment_id and not attachment_valid and not self._results_stale:
            self.search_index_dirty.emit()
            return
        context_result = self._context_result(result)
        context_attachment = attachment
        if context_result is not result:
            context_attachment, context = self._result_attachment_details(context_result)
        context_missing = bool(context_result.attachment_id) and context_attachment is None
        for button in self.file_actions:
            button.setEnabled(not self.chat_panel.busy and (result.result_kind == "file"
                              or result.result_kind == "attachment_pending" and attachment_valid))
        path_label = result.path if result.result_kind == "file" else (
            (attachment["local_path"] or "点击打开时定位") if attachment_valid else
            "关联已变化，待重新搜索" if result.attachment_id else "聊天记录")
        self.bind_result_button.setEnabled(attachment_valid and not self.chat_panel.busy)
        self.index_result_button.setEnabled(attachment_valid and not self.chat_panel.busy)
        reranker = "未启用" if result.reranker_score is None else f"{result.reranker_score:.3f}"
        time_label = "修改时间" if result.result_kind == "file" else "发送时间"
        semantic_label = "分句语义覆盖" if any(part.query_clause for part in result.supporting_evidence) else "语义"
        meta = [
            f"文件：{result.name}",
            f"路径：{path_label}",
            f"{time_label}：{display_time(result.modified_at)}",
            f"时间原始值：{result.modified_at or '未知'}",
            f"排序分：{result.score:.3f}",
            f"匹配原因：{result.match_reason}",
            f"关键词：{result.keyword_score:.3f}   {semantic_label}：{result.semantic_score:.3f}",
            f"融合：{result.fused_score:.3f}   AI 精排：{reranker}",
            f"来源：{result_source_label(result)}",
            f"解析标识：{result.source_type}",
        ]
        if result.page_no:
            meta.append(f"页码：{result.page_no}")
        if result.sheet_name:
            meta.append(f"Sheet：{result.sheet_name}")
        if result.result_kind == "file":
            meta.append("文件状态：" + STATE_LABELS.get(result.file_state, STATE_LABELS["unchecked"]))
        if attachment_valid:
            from .chat_indexing import content_status
            meta.append("正文索引：" + content_status(attachment, include_embeddings=self.enable_embedding.isChecked(),
                                                    model_name=self.db.get_setting("embedding_model")))
        elif result.attachment_id:
            meta.append("聊天来源：已移除、关联或筛选信息已变化")
        warnings = []
        if result.file_state in WARNING_STATES:
            warnings.append(STATE_LABELS[result.file_state])
        if result.attachment_id and not attachment_valid:
            warnings.append("聊天来源已移除或关联已变化")
        elif context_missing:
            warnings.append("所选聊天线索已移除或关联已变化")
        if self._results_stale:
            warnings.append("上次搜索快照，排名与命中片段待刷新")
        self.detail_warning.setText("\n".join(warnings))
        self.detail_warning.setVisible(bool(warnings))
        if warnings:
            meta.append("检索状态：" + "；".join(warnings))
        snippet, shortened = evidence_text(result, self.evidence_expand.isChecked())
        self.preview.setPlainText(snippet)
        self._evidence_has_more = shortened or self.evidence_expand.isChecked()
        self.evidence_expand.setVisible(self._evidence_has_more and self.preview_tabs.currentIndex() == 0)
        expanded = self.evidence_expand.isChecked()
        self.evidence_expand.setToolTip("收起匹配详情" if expanded else "展开匹配详情")
        self.evidence_expand.setAccessibleName(self.evidence_expand.toolTip())
        self.evidence_expand.setIcon(icon("chevron-down" if expanded else "chevron-right"))
        self.info_preview.setPlainText("\n\n".join(meta))
        self.file_summary.setPlainText(f"文件位置\n{path_label}\n\n{time_label}\n{display_time(result.modified_at)}"
                                       + ("\n\n" + next((line for line in meta if line.startswith("正文索引：")), "")
                                          if attachment_valid else ""))
        self.preview_tabs.setTabEnabled(2, True)
        self._highlight_evidence(result)
        readable_context, context_shortened = context_text(
            context, context_attachment, self.context_expand.isChecked(), include_header=self.source_selector.count() <= 1)
        self.context_preview.setPlainText(readable_context)
        self.context_preview.setPlaceholderText("这条来源已移除或关联已变化，请重新搜索" if context_missing else "暂无聊天来源")
        self.context_expand.setVisible(bool(context) or self.context_expand.isChecked())
        self.context_expand.setToolTip("返回聊天记录" if self.context_expand.isChecked() else "来源文本")
        self.context_expand.setAccessibleName(self.context_expand.toolTip())
        self.context_expand.setIcon(icon("chevron-down" if self.context_expand.isChecked() else "chevron-right"))
        has_sources = bool(context) or self.source_selector.count() > 1 or bool(result.chat_source and not result.attachment_id)
        self.preview_tabs.setTabEnabled(1, has_sources)
        if not has_sources and self.preview_tabs.currentIndex() == 1:
            self.preview_tabs.setCurrentIndex(0)
        self.source_copy_button.setEnabled(bool(context))
        self.source_history_button.setEnabled(result.result_kind == "file" and result.attachment_id is not None)
        self.chat_timeline.set_result(result, self.results if result.result_kind == "message" else (),
                                      revision=self.db.index_revisions())
        self.context_preview.setVisible(self.context_expand.isChecked())
        self.chat_timeline.setVisible(True)
        self.chat_timeline.set_raw_mode(self.context_expand.isChecked())
        self.source_copy_button.setVisible(self.context_expand.isChecked())
        self.source_selector.setVisible(self.context_expand.isChecked() and self.source_selector.count() > 1)
        if detail_changed and result.result_kind == "message" and has_sources:
            self.preview_tabs.setCurrentIndex(1)
        self._show_file_preview(result, attachment)

    def _detail_tab_changed(self, index):
        self.evidence_expand.setVisible(index == 0 and self._evidence_has_more)
        if not hasattr(self, "file_preview"):
            return
        self._set_file_preview_visible(self.file_preview._target is not None)

    def _track_topics(self, request):
        self._topic_jobs[id(request)] = request["cancel"]

    def _topics_finished(self, request, data, notice):
        self._topic_jobs.pop(id(request), None)
        import shiboken6
        receiver = request["receiver"]()
        if receiver is not None and shiboken6.isValid(receiver):
            receiver.topics_finished(request, data, notice)

    def _read_alpha_context(self, result, group, row, kind):
        from .alpha_chat_context import anchor_record, link_known_attachments
        from .desktop.chat_transcript import AlphaContextDialog
        from .search_filters import epoch
        from .weflow import WeFlowClient, WeFlowError

        token = self.chat_panel.token.text().strip()
        if not token:
            QMessageBox.information(self, "Alpha 连接", "请先在微信资料页填写 Alpha Token；本地聊天记录仍可使用。")
            return
        try:
            anchor = anchor_record(self.db.db_path, result, group, row["id"])
            client = WeFlowClient(self.chat_panel.url.text(), token=token, timeout=15, account=anchor["account_id"])
            if client.base_url != anchor["base_url"] or self.chat_panel.accounts.currentData() not in (None, anchor["account_id"]):
                raise WeFlowError("请在微信资料页连接建立此索引的 Alpha 地址和账号。")
        except Exception as exc:
            QMessageBox.information(self, "暂时无法读取", str(exc) if isinstance(exc, WeFlowError) else "来源已变化，请重新搜索后重试。")
            return
        finally:
            token = ""
        scope = ("所选合并转发的子记录" if kind == "forward" else
                 "选定会话中锚点前后各最多 20 条候选消息，再按前后 15 分钟及当前日期范围筛选展示。\n候选可能包含时间范围外的消息")
        if QMessageBox.question(self, "读取 Alpha 聊天记录", "仅从本机 Alpha 读取" + scope + "。\n不写入索引，不下载附件，不发送消息。继续？") != QMessageBox.StandardButton.Yes:
            client._token = ""
            return
        query = result.source_query
        chat_dates = query is not None and (query.scope in {"chat_files", "messages"} or query.chat_platform or query.conversation or query.sender)
        options = {"kind": kind, "start": epoch(query.modified_after) if chat_dates else None,
                   "end": epoch(query.modified_before) if chat_dates else None}

        def guard():
            return anchor_record(self.db.db_path, result, group, row["id"]) == anchor

        dialog = AlphaContextDialog(client, anchor, options=options, guard=guard,
            enrich=lambda batch: link_known_attachments(self.db.db_path, group, anchor, batch),
            attachment=lambda remote, att: self._open_timeline_attachment(remote["local_message"], att), parent=self)
        try:
            dialog.exec()
        finally:
            client._token = ""
            dialog.deleteLater()

    def _open_timeline_attachment(self, message, attachment):
        from PySide6.QtWidgets import QDialog
        from .chats import ChatStore
        from .desktop.file_preview import FilePreview
        from .desktop.widgets import button

        def current():
            rows = ChatStore(self.db).attachments(attachment_id=attachment["id"])
            if not rows:
                return None
            row = rows[0]
            if (row["message_id"] != message["id"] or row["external_id"] != message["external_id"]
                    or row["name"] != attachment["name"] or row["file_id"] != attachment["file_id"]
                    or row["origin"] != attachment["origin"]
                    or row["file_id"] is not None and row["local_path"] != attachment["path"]
                    or any(row[k] != message[k] for k in ("platform", "account", "source_key"))):
                return None
            return row

        actual = current()
        if actual is None:
            QMessageBox.information(self, "附件已变化", "请刷新聊天记录后重试。")
            return
        if not actual["file_id"]:
            self.chat_panel.locate_attachment(actual["id"], "open")
            return
        path = actual["local_path"]
        dialog = QDialog(self)
        dialog.setWindowTitle("聊天附件 · " + actual["name"])
        dialog.resize(900, 700)
        layout = QVBoxLayout(dialog)
        title = QLabel(actual["name"])
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        layout.addWidget(title)
        preview = FilePreview(dialog, expanded=True)
        preview.set_target(path)
        layout.addWidget(preview, 1)
        def open_file():
            verified = current()
            if verified and verified["local_path"] == path:
                self.perform_file_action(path, "open")
            else:
                QMessageBox.information(dialog, "附件已变化", "文件关联已变化，请关闭预览并刷新。")
        layout.addWidget(button("打开文件", "external-link", open_file, primary=True))
        try:
            dialog.exec()
        finally:
            preview.shutdown()
            dialog.deleteLater()

    def _apply_discussion_groups(self, groups=None):
        if groups is not None:
            self._discussion_groups = groups
        available = bool(self.results) and all(r.result_kind == "message" for r in self.results)
        self.discussion_mode.setVisible(available)
        active = available and self.discussion_mode.currentData()
        current = self.result_table.currentRow()
        selected = current
        self.result_table.blockSignals(True)
        for i, result in enumerate(self.results):
            self.result_table.setRowHidden(i, False)
            self.result_table.item(i, 0).setText(result.name)
            self.result_table.item(i, 0).setData(Qt.ItemDataRole.UserRole, result.match_reason)
        if active:
            for group in getattr(self, "_discussion_groups", ()):
                identities = {(r.message_id, r.attachment_id) for r in group.refs}
                members = [i for i, r in enumerate(self.results) if (r.chat_message_id, r.attachment_id) in identities]
                if not members:
                    continue
                first = members[0]
                item = self.result_table.item(first, 0)
                item.setText(group.title)
                item.setData(Qt.ItemDataRole.UserRole, f"{group.conversation} · {len(group.hit_ids)} 条命中消息")
                for i in members[1:]:
                    self.result_table.setRowHidden(i, True)
                if current in members:
                    selected = first
        if selected != current:
            self.result_table.selectRow(selected)
        self.result_table.blockSignals(False)
        self._update_result_controls()
        if selected != current:
            self.show_selected_result()

    def _toggle_evidence(self):
        self.show_selected_result()

    def _toggle_context(self):
        self.show_selected_result()

    def _toggle_technical(self):
        expanded = self.technical_toggle.isChecked()
        self.file_summary.setVisible(not expanded)
        self.info_preview.setVisible(expanded)
        self.technical_toggle.setText("收起技术详情" if expanded else "检索技术详情")

    def _show_file_preview(self, result, attachment):
        if result.result_kind == "file":
            page = next((part.page_no for part in result.supporting_evidence
                         if part.source_type != "chat" and part.page_no), result.page_no)
            sheet = next((part.sheet_name for part in result.supporting_evidence
                          if part.source_type != "chat" and part.sheet_name), result.sheet_name)
            self.file_preview.set_target(result.path, page=page, sheet=sheet, revision=self._index_revision,
                                         anchor=result.preview_anchor)
        elif result.result_kind == "attachment_pending" and attachment:
            from .wechat_files import AttachmentLocator
            cached = self.db.conn.execute("SELECT * FROM chat_attachment_locations WHERE attachment_id=?",
                                          (result.attachment_id,)).fetchone()
            if cached and cached["signature"] == AttachmentLocator.signature(attachment) and cached["sha256"]:
                self.file_preview.set_target(cached["path"], revision=self._index_revision, expected_hash=cached["sha256"])
            else:
                self.file_preview.clear("尚未定位到文件，可先打开或关联本地文件")
        else:
            self.file_preview.clear("尚未建立本地文件关联，可先定位或关联文件" if result.attachment_id else "此结果是聊天消息，没有可预览文件")
        available = self.file_preview._target is not None
        self._set_file_preview_visible(available)
        source_label = result_source_label(result)
        file_type = result.ext.upper()
        meta = source_label + (" · " + file_type if file_type and file_type not in source_label.upper() else "")
        if result.result_kind == "attachment_pending":
            meta += " · " + ("已记录路径" if available else "待定位")
        self.detail_meta.setText(meta)
        command = "定位并打开" if result.result_kind == "attachment_pending" and not available else "打开文件"
        self.file_actions[0].setText(command)
        self.file_actions[0].setAccessibleName(command)

    def _set_file_preview_visible(self, visible):
        visible = visible and (self.preview_tabs.currentIndex() != 1 or self.context_expand.isChecked())
        if visible == (not self.file_preview.isHidden()):
            return
        if not visible:
            sizes = self.detail_splitter.sizes()
            if all(sizes):
                self._file_preview_sizes = sizes
        self.file_preview.setVisible(visible)
        if visible:
            self.detail_splitter.setSizes(getattr(self, "_file_preview_sizes", [320, 180]))

    def _clear_result_details(self, message=""):
        self._detail_identity = None
        self.detail_title.setText("文件详情")
        self.detail_title.setToolTip("")
        self.detail_meta.clear()
        self.identifier_notice.clear()
        self.identifier_notice.setToolTip("")
        self.identifier_notice.hide()
        self.preview.setPlainText(message)
        self.preview.setExtraSelections([])
        self.context_preview.clear()
        self.chat_timeline.clear()
        self.discussion_mode.hide()
        self.source_selector.blockSignals(True)
        self.source_selector.clear()
        self.source_selector.blockSignals(False)
        self.source_selector.hide()
        self.source_selector.setToolTip("")
        self.context_preview.setPlaceholderText("")
        self.info_preview.clear()
        self.file_summary.clear()
        self.file_preview.clear()
        self._set_file_preview_visible(False)
        self.file_actions[0].setText("打开文件")
        self.file_actions[0].setAccessibleName("打开文件")
        for toggle in (self.evidence_expand, self.context_expand, self.technical_toggle):
            toggle.setChecked(False)
        self._evidence_has_more = False
        self.evidence_expand.hide()
        self.context_expand.hide()
        self.info_preview.hide()
        self.file_summary.show()
        self.technical_toggle.setText("检索技术详情")
        self.detail_warning.clear()
        self.detail_warning.hide()
        self.preview_tabs.setCurrentIndex(0)
        self.preview_tabs.setTabEnabled(1, False)
        self.preview_tabs.setTabEnabled(2, False)
        for button in (*self.file_actions, self.source_copy_button, self.source_history_button,
                       self.bind_result_button, self.index_result_button):
            button.setEnabled(False)

    def _highlight_evidence(self, result):
        from PySide6.QtCore import QRegularExpression
        from PySide6.QtGui import QColor, QFont
        from .identifiers import identifier_pattern

        selections = []
        document = self.preview.document()
        for evidence in result.supporting_evidence:
            if len(selections) >= 128:
                break
            label = "聊天线索" if evidence.source_type == "chat" else "文件正文"
            heading = document.find(label)
            if not heading.isNull():
                selection = QTextEdit.ExtraSelection()
                selection.cursor = heading
                selection.format.setFontWeight(QFont.Weight.DemiBold)
                selection.format.setForeground(QColor("#137b75"))
                selections.append(selection)
            for term in evidence.matched_terms:
                position = 0
                pattern = (QRegularExpression(identifier_pattern(term).pattern,
                           QRegularExpression.PatternOption.CaseInsensitiveOption)
                           if term == result.identifier_query else term)
                while term and len(selections) < 128:
                    found = document.find(pattern, position)
                    if found.isNull():
                        break
                    selection = QTextEdit.ExtraSelection()
                    selection.cursor = found
                    selection.format.setBackground(QColor("#fff2a8"))
                    selection.format.setForeground(QColor("#25313b"))
                    selections.append(selection)
                    position = found.selectionEnd()
        self.preview.setExtraSelections(selections)

    def bind_selected_attachment(self) -> None:
        result = self.selected_result()
        if result and result.attachment_id and self._guard_attachment_action(result):
            self.chat_panel.bind_attachment(result.attachment_id)

    def index_selected_attachment(self):
        result = self.selected_result()
        if result and result.attachment_id and self._guard_attachment_action(result):
            self.chat_panel.index_attachment_contents([result.attachment_id])

    def selected_result(self) -> SearchResult | None:
        row = self.result_table.currentRow()
        if row < 0 or row >= len(self.results) or not self.result_table.selectionModel().hasSelection():
            return None
        return self.results[row]

    def open_selected_file(self) -> None:
        self.selected_file_action("open")

    def open_selected_folder(self) -> None:
        self.selected_file_action("folder")

    def copy_selected_path(self) -> None:
        self.selected_file_action("copy")

    def selected_file_action(self, action):
        result = self.selected_result()
        if not result:
            return
        if result.result_kind == "file":
            self.perform_file_action(result.path, action)
        elif result.result_kind == "attachment_pending" and self._guard_attachment_action(result):
            self.chat_panel.locate_attachment(result.attachment_id, action)

    def perform_file_action(self, path, action):
        from .wechat_files import local_file
        from .weflow import WeFlowError
        try:
            actual = local_file(path)
            if action == "copy":
                QGuiApplication.clipboard().setText(str(actual))
            elif action == "folder":
                if sys.platform.startswith("win"):
                    subprocess.Popen(["explorer", "/select,", str(actual)])
                else:
                    subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(actual.parent)])
            elif action == "open":
                if actual.suffix.lower() not in {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx", ".pptx",
                                                ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                                                ".mp3", ".mp4", ".wav", ".zip", ".7z", ".rar"}:
                    if QMessageBox.question(self, "确认打开文件", "此文件类型可能执行程序或脚本。确定使用系统关联程序打开吗？",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                        return
                os.startfile(str(actual))  # type: ignore[attr-defined]
        except Exception as exc:
            message = str(exc) if isinstance(exc, WeFlowError) else "无法打开文件；文件可能已移动、被占用，或系统没有关联程序。"
            QMessageBox.warning(self, "文件操作失败", message)

    def _startup_reconcile(self) -> None:
        self._sync_monitor()
        if self.startup_scan.isChecked() and self.folder_list.count():
            self.start_scan(automatic=True)

    def start_scan(self, *, automatic: bool = False) -> None:
        folders = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        if automatic:
            folders = [folder for folder in folders if Path(folder).is_dir() and Path(folder).parent != Path(folder)]
        if not folders:
            if automatic:
                return
            QMessageBox.warning(self, "需要文件夹", "请先添加至少一个文件夹。")
            return
        if self.scan_thread and self.scan_thread.isRunning():
            return
        if self.incremental_busy or self.chat_panel.busy:
            if automatic:
                QTimer.singleShot(1000, self._startup_reconcile)
            else:
                QMessageBox.information(self, "索引更新中", "请等待当前增量更新完成。")
            return
        if any(Path(folder).parent == Path(folder) for folder in folders):
            answer = QMessageBox.question(
                self,
                "确认全盘扫描",
                "检测到磁盘根目录。全文解析、OCR 和语义索引可能持续很久，是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.db.set_setting("ocr_enabled", "1" if self.enable_ocr.isChecked() else "0")
        self.db.set_setting("embedding_enabled", "1" if self.enable_embedding.isChecked() else "0")
        self.progress_log.clear()
        self.progress_bar.show()
        self.scan_button.setEnabled(False)
        self.cancel_scan_button.setEnabled(True)
        self.header_status.setText("建立索引中")
        self.monitor_status.setText("扫描期间的文件变化已排队")
        self.scan_thread = QThread()
        self.scan_worker = ScanWorker(
            str(self.db.db_path),
            folders,
            self.enable_ocr.isChecked(),
            self.enable_embedding.isChecked(),
            str(self.cache_dir),
            self.embedding_device,
        )
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.progress.connect(self.progress_log.appendPlainText)
        self.scan_worker.finished.connect(self._scan_finished)
        self.scan_worker.failed.connect(self._scan_failed)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.failed.connect(self.scan_thread.quit)
        self.scan_worker.finished.connect(self.scan_worker.deleteLater)
        self.scan_worker.failed.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self._scan_cleanup)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.start()

    def cancel_scan(self) -> None:
        if self.scan_worker is not None:
            self.scan_worker.cancel_event.set()
            self.cancel_scan_button.setEnabled(False)
            self.header_status.setText("正在取消扫描")

    def _scan_cleanup(self) -> None:
        self.scan_thread = None
        self.scan_worker = None
        self.scan_button.setEnabled(True)
        self.cancel_scan_button.setEnabled(False)
        self._queue_file_events([])
        self._sync_monitor()

    def _scan_finished(self, summary: ScanSummary) -> None:
        self.progress_bar.hide()
        self.header_status.setText("本地模式")
        self.progress_log.appendPlainText(
            f"{'已取消' if summary.cancelled else '完成'}：发现 {summary.files_seen}，更新 {summary.files_indexed}，"
            f"跳过 {summary.files_skipped}，移除 {summary.files_removed}，错误 {summary.errors}"
        )
        self.search_index_dirty.emit()
        self.refresh_stats()

    def _scan_failed(self, error: str) -> None:
        self.progress_bar.hide()
        self.header_status.setText("本地模式")
        self.search_index_dirty.emit()
        QMessageBox.critical(self, "扫描失败", error)

    def _monitor_toggled(self, checked: bool) -> None:
        self.db.set_setting("monitor_enabled", "1" if checked else "0")
        self._sync_monitor()

    def _sync_monitor(self) -> None:
        if self.scan_thread is not None:
            return
        folders = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        if (self.enable_monitor.isChecked() and self.file_monitor is not None
                and self.file_monitor.running and self.file_monitor.roots == folders):
            self.monitor_status.setText(f"正在监控 {len(folders)} 个文件夹")
            return
        if self.file_monitor is not None:
            self.file_monitor.stop()
            self.file_monitor = None
        folders = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        if not self.enable_monitor.isChecked() or not folders:
            self.monitor_status.setText("监控已关闭" if folders else "等待添加文件夹")
            return
        self.file_monitor = FileChangeMonitor(folders, self.monitor_events.emit)
        if self.file_monitor.start():
            self.monitor_status.setText(f"正在监控 {len(folders)} 个文件夹")
        else:
            self.monitor_status.setText(self.file_monitor.error or "监控启动失败")

    @Slot(object)
    def _queue_file_events(self, events: list[FileEvent]) -> None:
        if not self.enable_monitor.isChecked():
            self.pending_file_events.clear()
            return
        self.pending_file_events.extend(events)
        if self.scan_thread is not None or self.incremental_busy or self.chat_panel.busy or not self.pending_file_events:
            return
        pending, self.pending_file_events = self.pending_file_events, []
        self.incremental_jobs += 1
        self.incremental_busy = True
        self.incremental_requested.emit({
            "events": pending, "ocr": self.enable_ocr.isChecked(),
            "embeddings": self.enable_embedding.isChecked(),
        })

    def _append_monitor_log(self, text: str) -> None:
        self.progress_log.appendPlainText(text)

    def _incremental_finished(self, summary: IncrementalUpdateSummary) -> None:
        self.incremental_jobs = max(0, self.incremental_jobs - 1)
        self.incremental_busy = self.incremental_jobs > 0
        self._queue_file_events([])
        if summary.indexed or summary.removed or summary.errors:
            self.progress_log.appendPlainText(
                f"增量完成：更新 {summary.indexed}，移除 {summary.removed}，错误 {summary.errors}"
            )
            if summary.indexed or summary.removed:
                self.search_index_dirty.emit()
            self.refresh_stats()

    def clear_index(self) -> None:
        if self._model_task_running():
            QMessageBox.information(self, "索引正在使用", "请取消扫描或等待当前任务完成后清空。")
            return
        answer = QMessageBox.question(
            self,
            "确认清空",
            "确定清空本地全文、OCR 和向量索引吗？原文件和已选择文件夹不会被修改。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.enable_monitor.setChecked(False)
        self.pending_file_events.clear()
        self.startup_scan.setChecked(False)
        self.chat_panel.auto.setChecked(False)
        self.chat_panel.reset_target()
        self.chat_panel.active_scope = None
        self.chat_panel.scope_label.setText("指定会话" if self.chat_panel.single_session.isChecked() else "选择会话")
        self.db.clear_index()
        self.search_index_dirty.emit()
        self._has_search = False
        self._displayed_request = self._pending_request = None
        self._preserved_selection = None
        self._results_stale = False
        self._results_db_revision = None
        self._last_index_check_revision = None
        self._index_watch_error = False
        self._update_result_notice()
        self.search_status.setText("索引已清空")
        self.search_time_status.clear()
        self.search_time_status.hide()
        self.results = []
        self._update_result_controls()
        self.result_table.setRowCount(0)
        self._clear_result_details()
        self.chat_panel.refresh()
        self.refresh_stats()

    def refresh_stats(self) -> None:
        stats = self.db.stats()
        self.library_status.setText(f"{stats.files_total:,} 个索引文件")
        self.search_content.setCurrentWidget(self.empty_library if not stats.chunks_total and not self._has_search
                                             else self.search_splitter)
        for key, field in self.metric_labels.items():
            field.setText(f"{getattr(stats, key):,}")
        mb = stats.db_size_bytes / 1024 / 1024
        embedding_model = self.db.get_setting("embedding_model", "尚未建立")
        embedding_error = self.db.get_setting("embedding_load_error", "")
        model_state = embedding_model
        if embedding_error:
            model_state += "（已降级，详情见扫描记录）"
        self.stats_label.setText(
            f"已解析 {stats.parsed_files:,} 个文件  ·  待 OCR {stats.ocr_pending:,} 项  ·  {stats.watched_folders} 个监控目录\n"
            f"索引大小 {mb:.2f} MB  ·  最近扫描 {display_time(stats.last_scan_at) if stats.last_scan_at else '尚未扫描'}\n"
            f"Embedding：{model_state}  ·  FTS：{self.db.get_setting('fts_tokenizer', 'unknown')}"
        )
        resolved = resolve_reranker_runtime(self.reranker_runtime, self.runtime_capabilities)
        self.runtime_status_label.setText(resolved.message)
        active_device = self.db.get_setting("embedding_device_active", "尚未运行")
        runtime_note = self.db.get_setting("embedding_runtime_note", "")
        self.embedding_runtime_label.setText(f"最近索引：{active_device}  {runtime_note}")
        cache = model_cache_stats(self.cache_dir)
        self.cache_path_input.setText(str(cache.path))
        self.cache_status_label.setText(
            f"{cache.files} 个缓存文件，共 {format_bytes(cache.bytes_total)}；更换位置不会自动搬移旧缓存。"
        )
        ocr_status = self.db.get_setting("ocr_runtime_status", "unchecked")
        ocr_error = self.db.get_setting("ocr_load_error", "")
        ocr_messages = {
            "ready": "PaddleOCR 模型已就绪",
            "disabled": "OCR 已关闭",
            "missing": "未安装 PaddleOCR",
            "unchecked": "PaddleOCR 尚未检查",
        }
        self.ocr_runtime_label.setText(
            ocr_messages.get(ocr_status or "unchecked", f"初始化失败：{ocr_error or '未知错误'}")
        )

    def _model_task_running(self) -> bool:
        return bool(
            self.search_busy
            or bool(self._topic_jobs)
            or (hasattr(self, "chat_panel") and self.chat_panel.busy)
            or self.incremental_busy
            or bool(self.pending_file_events)
            or (self.scan_thread and self.scan_thread.isRunning())
            or (self.ocr_check_thread and self.ocr_check_thread.isRunning())
        )

    def _configure_model_workers(self) -> None:
        payload = {
            "cache_dir": str(self.cache_dir),
            "reranker_runtime": self.reranker_runtime,
            "embedding_device": self.embedding_device,
        }
        self.search_configure.emit(payload)
        self.incremental_configure.emit(payload)

    def _embedding_device_changed(self) -> None:
        selected = self.embedding_device_combo.currentData()
        if self._model_task_running():
            self.embedding_device_combo.blockSignals(True)
            self.embedding_device_combo.setCurrentIndex(self.embedding_device_combo.findData(self.embedding_device))
            self.embedding_device_combo.blockSignals(False)
            QMessageBox.information(self, "模型正在使用", "请等待当前任务完成后更换设备。")
            return
        self.embedding_device = str(selected)
        self.db.set_setting("embedding_device", self.embedding_device)
        self._configure_model_workers()
        self.refresh_stats()

    def choose_cache_location(self) -> None:
        if self._model_task_running():
            QMessageBox.information(self, "模型正在使用", "请等待当前搜索、索引或 OCR 检查完成。")
            return
        parent = QFileDialog.getExistingDirectory(self, "选择模型缓存的父目录")
        if not parent:
            return
        try:
            self.cache_dir = prepare_custom_cache_dir(parent)
        except Exception as exc:
            QMessageBox.critical(self, "无法使用缓存目录", str(exc))
            return
        self.db.set_setting("model_cache_dir", str(self.cache_dir))
        self._configure_model_workers()
        self.refresh_stats()

    def restore_default_cache_location(self) -> None:
        if self._model_task_running():
            QMessageBox.information(self, "模型正在使用", "请等待当前任务完成后再更换缓存位置。")
            return
        self.cache_dir = prepare_model_cache_dir()
        self.db.set_setting("model_cache_dir", str(self.cache_dir))
        self._configure_model_workers()
        self.refresh_stats()

    def clear_cache(self) -> None:
        if self._model_task_running():
            QMessageBox.information(self, "模型正在使用", "请等待当前任务完成后再清理缓存。")
            return
        stats = model_cache_stats(self.cache_dir)
        answer = QMessageBox.question(
            self,
            "确认清理缓存",
            f"将清理 {stats.path} 中 {format_bytes(stats.bytes_total)} 的模型与微信附件缓存。\n"
            "索引数据库、微信原文件和手动关联的文件不会被删除；模型可能需要重新下载，缓存附件下次需重新定位。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = clear_model_cache(self.cache_dir)
            self._configure_model_workers()
            self.db.set_setting("ocr_runtime_status", "unchecked")
            self.db.set_setting("ocr_load_error", "")
            self.refresh_stats()
            QMessageBox.information(self, "缓存已清理", f"已释放约 {format_bytes(removed.bytes_total)}。")
        except Exception as exc:
            QMessageBox.critical(self, "缓存清理失败", str(exc))

    def _reranker_runtime_changed(self) -> None:
        selected = self.reranker_runtime_combo.currentData()
        if not selected:
            return
        if self._model_task_running():
            self.reranker_runtime_combo.blockSignals(True)
            self.reranker_runtime_combo.setCurrentIndex(self.reranker_runtime_combo.findData(self.reranker_runtime))
            self.reranker_runtime_combo.blockSignals(False)
            QMessageBox.information(self, "模型正在使用", "请等待当前任务完成后更换设备。")
            return
        self.reranker_runtime = str(selected)
        self.db.set_setting("reranker_runtime", self.reranker_runtime)
        if hasattr(self, "cache_path_input"):
            self._configure_model_workers()
            self.refresh_stats()

    def check_ocr_runtime(self) -> None:
        if self._model_task_running():
            QMessageBox.information(self, "任务进行中", "请等待当前任务完成后再检查 OCR。")
            return
        self.ocr_check_button.setEnabled(False)
        self.ocr_runtime_label.setText("正在后台初始化 PaddleOCR；首次检查可能下载模型……")
        self.ocr_check_thread = QThread(self)
        self.ocr_check_worker = OcrCheckWorker(str(self.cache_dir))
        self.ocr_check_worker.moveToThread(self.ocr_check_thread)
        self.ocr_check_thread.started.connect(self.ocr_check_worker.run)
        self.ocr_check_worker.finished.connect(self._ocr_check_finished)
        self.ocr_check_worker.finished.connect(self.ocr_check_thread.quit)
        self.ocr_check_thread.finished.connect(self.ocr_check_worker.deleteLater)
        self.ocr_check_thread.finished.connect(self._ocr_check_cleanup)
        self.ocr_check_thread.finished.connect(self.ocr_check_thread.deleteLater)
        self.ocr_check_thread.start()

    def _ocr_check_finished(self, status: str, error: str, seconds: float) -> None:
        self.db.set_setting("ocr_runtime_status", status)
        self.db.set_setting("ocr_load_error", error)
        self.ocr_check_button.setEnabled(True)
        self.refresh_stats()
        if status == "ready":
            self.ocr_runtime_label.setText(f"PaddleOCR 模型已就绪，初始化耗时 {seconds:.1f} 秒")

    def _ocr_check_cleanup(self) -> None:
        self.ocr_check_thread = None
        self.ocr_check_worker = None

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._closing:
            super().closeEvent(event)
            return
        if self._model_task_running():
            QMessageBox.information(self, "任务进行中", "请等待当前搜索或索引任务完成后再退出。")
            event.ignore()
            return
        if self.file_monitor is not None:
            self.file_monitor.stop()
        if not self.chat_panel.shutdown():
            event.ignore()
            return
        if not self.chat_timeline.shutdown():
            event.ignore()
            return
        QMetaObject.invokeMethod(
            self.search_worker,
            "close",
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        self.search_thread.quit()
        if not self.search_thread.wait(5000):
            QMessageBox.information(self, "搜索服务退出中", "请稍后再次退出。")
            event.ignore()
            return
        self.incremental_thread.quit()
        if not self.incremental_thread.wait(5000):
            QMessageBox.information(self, "增量更新进行中", "请等待当前文件更新完成后再退出。")
            event.ignore()
            return
        self._closing = True
        self.file_preview.shutdown()
        self.index_watch_timer.stop()
        self.index_revision_probe.close()
        try:
            self._save_desktop_layout()
        except (sqlite3.Error, OSError):
            # Layout preferences must not prevent shutdown after workers have stopped.
            pass
        self.db.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow(LocalMindDB())
    window.show()
    sys.exit(app.exec())
