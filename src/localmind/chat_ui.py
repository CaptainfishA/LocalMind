from __future__ import annotations

import time
from pathlib import Path
from threading import Event

from PySide6.QtCore import QDate, QDateTime, QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDateEdit, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QStyle, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QProgressBar, QTabWidget, QToolButton,
)

from .chats import ChatStore, digest, parse_export
from .db import LocalMindDB
from .embeddings import EmbeddingService
from .weflow import SyncCancelled, WeFlowClient, WeFlowError, sync_session
from .weflow_batch import batch_summary, load_batch, sync_all_sessions
from .weflow_scope import load_scope, save_scope
from .chat_scope_ui import ConversationScopeDialog
from .weflow_metadata import AlphaMetadata, MetadataCache
from .wechat_files import AttachmentLocator
from .chat_indexing import content_status, index_attachments
from .ocr import OcrService


class WeFlowWorker(QObject):
    finished = Signal(object)
    progress = Signal(str)
    batch_progress = Signal(object)

    def __init__(self, db_path: str, cancel: Event):
        super().__init__()
        self.db_path = db_path
        self.cancel = cancel
        self.embedding = None
        self.model_config = None
        self.ocr = None
        self.ocr_config = None
        self.metadata_cache = MetadataCache()
        self.metadata_key = None

    @Slot()
    def clear_metadata(self):
        self.metadata_cache = MetadataCache()
        self.metadata_key = None

    @Slot(object)
    def configure(self, _payload):
        self.embedding = None
        self.model_config = None
        self.ocr = None
        self.ocr_config = None

    @Slot(object)
    def process(self, payload: dict):
        started = time.perf_counter()
        db = None
        kind = payload["kind"]
        result = {"kind": kind, "changed": False}
        try:
            if kind == "scope_metadata":
                result["operation"] = payload["operation"]
                result["connection_key"] = digest(payload["url"] + "\0" + payload["token"])
                key = (result["connection_key"], payload.get("api_account"))
                if self.metadata_key != key:
                    self.clear_metadata()
                    self.metadata_key = key
                client = WeFlowClient(payload["url"], payload["token"], account=payload.get("api_account"))
                metadata = AlphaMetadata(client, self.metadata_cache, should_cancel=self.cancel.is_set, progress=self.progress.emit)
                if payload["operation"] == "resolve":
                    value = metadata.resolve(payload["query"])
                elif payload["operation"] == "labels":
                    value = metadata.collect("labels", payload["ids"], force=payload.get("force", False))
                elif payload["operation"] == "groups":
                    value = metadata.groups(payload["ids"], payload["person"], force=payload.get("force", False))
                else:
                    raise ValueError("元数据操作不支持")
                result["value"] = value
            elif kind in {"sessions", "scope"}:
                result["connection_key"] = digest(payload["url"] + "\0" + payload["token"])
                client = WeFlowClient(payload["url"], payload["token"], account=payload.get("api_account"))
                try:
                    result["sessions"] = client.sessions(payload["keyword"], should_cancel=self.cancel.is_set,
                                                         progress=self.progress.emit)
                finally:
                    result.update(accounts=client.available_accounts, account=client.account_id, protocol=client.protocol)
            else:
                db = LocalMindDB(self.db_path)
                store = ChatStore(db)
                if kind in {"sync", "bulk", "import", "attachment_index", "semantic_update"}:
                    config = (payload["cache"], payload["device"], kind == "semantic_update")
                    if self.model_config != config:
                        self.embedding = EmbeddingService(cache_dir=config[0], device=config[1], local_files_only=config[2])
                        self.model_config = config
                    embedding = self.embedding if payload["embeddings"] or kind == "semantic_update" else None
                    if kind in {"attachment_index", "semantic_update"}:
                        ocr_config = (payload["cache"], payload["ocr_enabled"])
                        if self.ocr_config != ocr_config:
                            self.ocr = OcrService(cache_dir=ocr_config[0], enabled=ocr_config[1])
                            self.ocr_config = ocr_config
                        if kind == "semantic_update":
                            from .semantic_maintenance import update_wechat_semantics
                            result.update(update_wechat_semantics(
                                db, embedding_service=embedding, ocr_service=self.ocr,
                                should_cancel=self.cancel.is_set, progress=self.progress.emit))
                        else:
                            result.update(index_attachments(
                                db, payload["attachments"], cache_dir=payload["cache"],
                                embedding_service=self.embedding, ocr_service=self.ocr,
                                include_embeddings=payload["embeddings"], should_cancel=self.cancel.is_set,
                                progress=self.progress.emit))
                    elif kind == "bulk":
                        client = WeFlowClient(payload["url"], payload["token"], account=payload.get("api_account"))
                        try:
                            result.update(sync_all_sessions(
                                db, client, start=payload["start"], incremental=payload["incremental"],
                                max_messages=payload["max_messages"], embedding_service=embedding,
                                should_cancel=self.cancel.is_set, progress=self.progress.emit,
                                on_update=self.batch_progress.emit,
                                scope=payload.get("scope"),
                            ))
                        finally:
                            result.update(accounts=client.available_accounts, account=client.account_id, protocol=client.protocol)
                    elif kind == "sync":
                        result.update(sync_session(
                            db, WeFlowClient(payload["url"], payload["token"], account=payload.get("api_account")), payload["session"],
                            start=payload["start"], incremental=payload["incremental"],
                            max_messages=payload["max_messages"], embedding_service=embedding,
                            should_cancel=self.cancel.is_set, progress=self.progress.emit,
                        ))
                    else:
                        meta, messages = parse_export(payload["path"])
                        if meta.get("platform", "wechat").lower() not in {"wechat", "weixin", "微信", "im"}:
                            raise ValueError("本版仅支持微信数据")
                        result.update(store.ingest(
                            platform="wechat", account=payload["account"], conversation=payload["conversation"],
                            source_key="import:" + digest(str(Path(payload["path"]).resolve())),
                            messages=messages, embedding_service=embedding, should_cancel=self.cancel.is_set,
                        ))
                    loaded = embedding and embedding.load_source != "not_loaded"
                    result["model"] = embedding.active_model_name if loaded else "语义模型未加载" if embedding else "仅关键词"
                    if loaded:
                        db.set_setting("embedding_device_active", embedding.device)
                        db.set_setting("embedding_runtime_note", embedding.runtime_note)
                        db.set_setting("embedding_load_error", embedding.load_error or "")
                elif kind in {"locate", "bind"}:
                    result.update(attachment=payload["attachment"], action=payload.get("action"))
                    locator = AttachmentLocator(
                        db, client_factory=lambda: WeFlowClient(payload["url"], payload["token"], account=payload.get("api_account")),
                        cache_dir=payload["cache"], root=payload.get("file_root", ""),
                        cancel=self.cancel.is_set, progress=self.progress.emit)
                    if kind == "locate":
                        result.update(locator.locate(payload["attachment"]))
                    else:
                        result.update(locator.bind(payload["attachment"], payload["path"]))
                    target = db.get_file_by_path(result["path"]) if kind == "bind" else None
                    if kind == "bind" and target:
                        store.bind(payload["attachment"], int(target["id"]))
                elif kind == "remove":
                    store.remove_source(payload["source"])
                if kind not in {"bulk", "attachment_index", "semantic_update"}:
                    result["changed"] = True
        except Exception as exc:
            # Transport errors are sanitized by WeFlowClient; never log the request/token.
            result["error"] = str(exc) if kind != "scope_metadata" or isinstance(exc, WeFlowError) else f"元数据处理失败（{type(exc).__name__}）"
            result["cancelled"] = isinstance(exc, SyncCancelled) or self.cancel.is_set()
        finally:
            if db:
                db.close()
            payload.pop("token", None)
            result["seconds"] = time.perf_counter() - started
            self.finished.emit(result)


class WeFlowPanel(QScrollArea):
    requested = Signal(object)
    metadata_reset = Signal()

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.busy = False
        self.cancel = Event()
        self.confirmed_target = None
        self.active_scope = None
        self.scope_dialog = None
        self.sources = []
        self.attachments = []
        self.job_rows = []
        self.session_cache = []
        self.task_started = None
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        from .desktop.chat_page import build_chat_page
        build_chat_page(self, host)
        self.controls = [self.url, self.token, self.accounts, self.keyword, self.connect_button, self.sessions, self.start_date,
                         self.max_messages, self.incremental, self.sync_button, self.import_button,
                         self.remove_button, self.bind_button, self.folder_button, self.single_session, self.period, self.auto]
        self.controls += [self.open_attachment_button, self.index_attachments_button, self.choose_root_button,
                          self.clear_root_button, self.semantic_update_button]
        self.attachment_table.cellDoubleClicked.connect(lambda *_: self.open_attachment())
        self.thread = QThread(self)
        self.worker = WeFlowWorker(str(host.db.db_path), self.cancel)
        self.worker.moveToThread(self.thread)
        self.requested.connect(self.worker.process)
        self.metadata_reset.connect(self.worker.clear_metadata)
        host.search_configure.connect(self.worker.configure)
        self.worker.progress.connect(self.show_progress)
        self.worker.batch_progress.connect(self.show_batch_progress)
        self.worker.finished.connect(self.finished)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self.timer = QTimer(self)
        self.timer.setInterval(300000)
        self.timer.timeout.connect(self.auto_sync)
        self.auto.toggled.connect(self.toggle_auto)
        for signal in [self.sessions.currentIndexChanged, self.start_date.dateChanged]:
            signal.connect(self.reset_target)
        self.accounts.currentIndexChanged.connect(self.reset_account)
        self.url.textChanged.connect(self.reset_connection)
        self.token.textChanged.connect(self.reset_connection)
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(1000)
        self.elapsed_timer.timeout.connect(self.update_elapsed)
        self.scope_changed()
        self.refresh()

    def button(self, text, icon, callback):
        from .desktop.widgets import button
        glyphs = {
            QStyle.StandardPixmap.SP_BrowserReload: "refresh-cw",
            QStyle.StandardPixmap.SP_ArrowDown: "download",
            QStyle.StandardPixmap.SP_MediaPause: "pause",
            QStyle.StandardPixmap.SP_DirOpenIcon: "folder",
            QStyle.StandardPixmap.SP_DialogResetButton: "x",
            QStyle.StandardPixmap.SP_DialogOpenButton: "external-link",
            QStyle.StandardPixmap.SP_TrashIcon: "trash-2",
            QStyle.StandardPixmap.SP_FileLinkIcon: "link",
            QStyle.StandardPixmap.SP_FileDialogContentsView: "file-text",
            QStyle.StandardPixmap.SP_FileDialogListView: "file",
        }
        return button(text, glyphs.get(icon, "file"), callback)

    @staticmethod
    def table(headers):
        from .desktop.widgets import DataTable
        table = DataTable(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().hide()
        table.setAlternatingRowColors(False)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        table.verticalHeader().setDefaultSectionSize(48)
        return table

    def toggle_advanced(self, checked):
        self.advanced.setVisible(checked)
        self.advanced_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def scope_changed(self, *_):
        single = self.single_session.isChecked()
        self.scope_label.setText("指定会话" if single else f"已选 {len(self.active_scope['ids'])} 个" if self.active_scope else "选择会话")
        self.keyword.setEnabled(single and not self.busy)
        self.sessions.setEnabled(single and not self.busy)
        self.sync_button.setText("同步此会话" if single else "一键建立索引")
        self.reset_target()

    def change_period(self):
        days = self.period.currentData()
        if days >= 0:
            self.start_date.setDate(QDate.currentDate().addDays(-days) if days else QDate(1970, 1, 1))

    def date_changed(self):
        days = self.period.currentData()
        expected = QDate.currentDate().addDays(-days) if days > 0 else QDate(1970, 1, 1)
        if self.start_date.date() != expected:
            self.period.blockSignals(True)
            self.period.setCurrentIndex(3)
            self.period.blockSignals(False)

    def filter_sessions(self):
        current = self.sessions.currentData()
        self.sessions.blockSignals(True)
        self.sessions.clear()
        keyword = self.keyword.text().strip().casefold()
        for session in self.session_cache:
            if keyword in session["name"].casefold():
                self.sessions.addItem(session["name"], session["id"])
        index = self.sessions.findData(current)
        if index >= 0:
            self.sessions.setCurrentIndex(index)
        self.sessions.blockSignals(False)
        if self.single_session.isChecked() and self.sessions.currentData() != current:
            self.reset_target()

    def set_connection_status(self, text, state="idle"):
        self.connection_status.setText(text)
        self.connection_status.setProperty("state", state)
        self.connection_status.style().unpolish(self.connection_status)
        self.connection_status.style().polish(self.connection_status)

    def pause(self):
        self.cancel.set()
        self.cancel_button.setEnabled(False)
        self.status.setText("正在暂停，等待当前读取或计算结束")

    def update_elapsed(self):
        if self.task_started is not None:
            seconds = int(time.perf_counter() - self.task_started)
            self.elapsed.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")

    def show_progress(self, text):
        if self.cancel.is_set():
            return
        if self.scope_dialog and self.scope_dialog.metadata_busy:
            self.scope_dialog.set_metadata_progress(text)
            return
        current = next((row["name"] for row in self.job_rows if row["status"] == "running"), "")
        self.status.setText(f"{current[:50]} · {text}" if current else text)

    @Slot(object)
    def show_batch_progress(self, event):
        if "sessions" in event:
            self.job_rows = event["sessions"]
            self.task_table.setRowCount(len(self.job_rows))
            indices = range(len(self.job_rows))
        else:
            index = event["index"]
            self.job_rows[index] = event["row"]
            indices = [index]
        labels = {"pending": "待处理", "running": "处理中", "done": "已索引", "empty": "无消息",
                  "skipped": "范围外", "failed": "失败", "partial": "部分完成"}
        for i in indices:
            row = self.job_rows[i]
            count = f"{row.get('received', 0)} / {row.get('failed_messages', 0)}" if "received" in row else ""
            values = [row["name"], labels[row["status"]], count,
                      f"{row['seconds']:.1f} 秒" if "seconds" in row else "", row.get("error", "")]
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                if j == 1:
                    color = {"done": "#147d77", "partial": "#936b12", "failed": "#a33a34", "running": "#245fc5"}.get(row["status"])
                    if color:
                        item.setForeground(QColor(color))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.task_table.setItem(i, j, item)
        total, done = event["total_sessions"], event["completed_sessions"]
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.progress_bar.setProperty("state", event["state"])
        self.progress_bar.style().unpolish(self.progress_bar)
        self.progress_bar.style().polish(self.progress_bar)
        self.batch_counts.setText(f"{done} / {total} 会话 · 成功 {event['succeeded_sessions']} · 部分完成 {event.get('partial_sessions', 0)} · 失败 {event['failed_sessions']} · 待重试消息 {event.get('failed_messages', 0)} · 读取 {event['received']} 条")
        if event["skipped_sessions"]:
            self.batch_counts.setText(self.batch_counts.text() + f" · 范围外 {event['skipped_sessions']}")
        self.snapshot_time.setText("截至 " + QDateTime.fromSecsSinceEpoch(event["end"]).toString("MM-dd HH:mm"))
        self.show_progress("建立索引中" if self.busy else "任务已加载")

    def show_all_attachments(self):
        self.source_table.clearSelection()
        self.source_table.setCurrentCell(-1, -1)
        self.refresh_attachments()
        self.data_tabs.setCurrentIndex(0)

    def target(self):
        return (self.url.text().strip(), self.accounts.currentData(),
                self.sessions.currentData() if self.single_session.isChecked() else self.active_scope["fingerprint"] if self.active_scope else None,
                self.start_date.date().toString("yyyy-MM-dd"))

    def reset_account(self, *_):
        self.metadata_reset.emit()
        if self.scope_dialog is not None:
            self.scope_dialog.reject()
        self.active_scope = None
        self.scope_label.setText("指定会话" if self.single_session.isChecked() else "选择会话")
        self.session_cache = []
        self.sessions.clear()
        self.job_rows = []
        self.task_table.setRowCount(0)
        self.set_connection_status("未连接")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.batch_counts.setText("待开始")
        self.snapshot_time.clear()
        self.reset_target()

    def reset_connection(self, *_):
        self.accounts.blockSignals(True)
        self.accounts.clear()
        self.accounts.addItem("当前账号（自动）", None)
        self.accounts.blockSignals(False)
        self.reset_account()

    def reset_target(self, *_):
        self.confirmed_target = None
        self.auto.setChecked(False)
        if not self.busy:
            self.sync_button.setText("同步此会话" if self.single_session.isChecked() else "一键建立索引")

    def toggle_auto(self, checked):
        if checked and not self.single_session.isChecked() and (not self.active_scope or not self.active_scope["account_id"]):
            self.auto.setChecked(False)
            self.status.setText("请先连接提供稳定账号 ID 的 Alpha，并确认索引范围。")
            return
        if checked and self.confirmed_target != self.target():
            self.auto.setChecked(False)
            self.status.setText("请先完成当前范围的索引，再开启自动更新。")
            return
        self.timer.start() if checked else self.timer.stop()

    def auto_sync(self):
        if not self.host._model_task_running() and self.confirmed_target == self.target():
            self.sync(automatic=True)

    def launch(self, kind, **kwargs):
        if self.host._model_task_running():
            self.status.setText("请等待当前搜索或索引任务完成。")
            return False
        self.busy = True
        self.host._update_result_controls()
        self.cancel.clear()
        self.task_started = time.perf_counter()
        self.elapsed_timer.start()
        self.update_elapsed()
        for widget in self.controls:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(kind in {"sessions", "scope", "scope_metadata", "sync", "bulk", "import", "locate", "attachment_index", "semantic_update"})
        self.progress_bar.setRange(0, 0)
        self.status.setText("读取账号与会话" if kind == "bulk" else "处理中")
        if kind == "bulk":
            self.data_tabs.setCurrentIndex(2)
        self.requested.emit({"kind": kind, "url": "" if kind == "semantic_update" else self.url.text(),
                             "token": "" if kind == "semantic_update" else self.token.text(),
                             "api_account": self.accounts.currentData(),
                             "file_root": self.file_root.text(),
                             "cache": str(self.host.cache_dir), "device": self.host.embedding_device,
                             "ocr_enabled": self.host.enable_ocr.isChecked(),
                             "embeddings": self.host.enable_embedding.isChecked(), **kwargs})
        return True

    def load_sessions(self):
        if not self.accept_privacy():
            return
        self.reset_target()
        self.sessions.clear()
        self.launch("sessions", keyword="")

    def accept_privacy(self, *, all_sessions=False):
        key = "weflow_bulk_privacy_accepted" if all_sessions else "weflow_privacy_accepted"
        if self.host.db.get_setting(key) != "1":
            scope = "你随后确认的会话在所选日期范围内" if all_sessions else "选定会话"
            answer = QMessageBox.question(self, "微信索引隐私", f"将读取{scope}的消息，并将正文、上下文和向量明文存入本地索引。仅连接本机 Alpha / WeFlow，不发送消息、不修改原聊天或文件。是否继续？")
            if answer != QMessageBox.StandardButton.Yes:
                return False
            self.host.db.set_setting(key, "1")
        return True

    def sync(self, *, automatic=False):
        single = self.single_session.isChecked()
        if self.busy or self.scope_dialog is not None:
            return
        if single and not self.sessions.currentData():
            self.status.setText("请先读取并选择一个会话。")
            return
        if not self.token.text().strip():
            self.status.setText("请填写 Alpha / WeFlow API Token。")
            self.token.setFocus()
            return
        if not self.accept_privacy(all_sessions=not single):
            return
        if not single and not automatic:
            self.auto.setChecked(False)
            self.launch("scope", keyword="")
            return
        if not single and (not self.active_scope or self.confirmed_target != self.target()):
            self.status.setText("请重新确认索引范围。")
            return
        start = max(0, int(self.start_date.date().startOfDay().toSecsSinceEpoch()))
        if self.launch("sync" if single else "bulk", session=self.sessions.currentData(), start=start,
                       incremental=self.incremental.isChecked(), max_messages=self.max_messages.value(), scope=self.active_scope):
            self.host.db.set_setting("weflow_index_start", self.start_date.date().toString("yyyy-MM-dd"))

    def choose_scope(self):
        client = WeFlowClient(self.url.text(), account=self.accounts.currentData())
        previous = load_scope(self.host.db, client.base_url, client.account_id)
        context = (self.url.text(), self.token.text(), self.accounts.currentData(), self.start_date.date())
        dialog = ConversationScopeDialog(
            self.session_cache, base_url=client.base_url, account_id=client.account_id,
            account_name=self.accounts.currentText(), period=self.start_date.date().toString("yyyy-MM-dd") + " 至今",
            previous=previous, parent=self,
            metadata_available=bool(client.account_id),
        )
        self.scope_dialog = dialog

        def metadata_request(request):
            if context != (self.url.text(), self.token.text(), self.accounts.currentData(), self.start_date.date()):
                dialog.receive_metadata({"operation": request["operation"], "error": "账号或范围已改变，请重新选择"})
            elif not self.launch("scope_metadata", **request):
                dialog.receive_metadata({"operation": request["operation"], "error": "请等待当前搜索或索引任务完成"})

        dialog.metadata_requested.connect(metadata_request)
        dialog.metadata_cancelled.connect(self.cancel.set)

        def accepted():
            current = (self.url.text(), self.token.text(), self.accounts.currentData(), self.start_date.date())
            if context != current:
                self.status.setText("账号或时间范围已改变，请重新选择。")
                return
            self.active_scope = dialog.selection()
            self.reset_target()
            self.scope_label.setText(f"已选 {len(self.active_scope['ids'])} 个")
            start = max(0, int(self.start_date.date().startOfDay().toSecsSinceEpoch()))
            if self.launch("bulk", start=start, incremental=self.incremental.isChecked(),
                           max_messages=self.max_messages.value(), scope=self.active_scope):
                save_scope(self.host.db, self.active_scope)
                self.host.db.set_setting("weflow_index_start", self.start_date.date().toString("yyyy-MM-dd"))

        def closed(_result):
            self.scope_dialog = None
            dialog.deleteLater()

        dialog.accepted.connect(accepted)
        dialog.finished.connect(closed)
        dialog.open()

    def import_export(self):
        if not self.accept_privacy():
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择微信 ChatLab 导出", "", "ChatLab (*.json *.jsonl)")
        if not path:
            return
        account, ok = QInputDialog.getText(self, "账号标识", "微信账号标签")
        if not ok or not account.strip():
            return
        conversation, ok = QInputDialog.getText(self, "会话标识", "微信会话名称")
        if ok and conversation.strip():
            self.launch("import", path=path, account=account.strip(), conversation=conversation.strip())

    def bind_selected(self):
        row = self.attachment_table.currentRow()
        if 0 <= row < len(self.attachments):
            self.bind_attachment(self.attachments[row]["id"])

    def choose_file_root(self):
        path = QFileDialog.getExistingDirectory(self, "选择微信文件存储目录", self.file_root.text())
        if not path:
            return False
        root = Path(path).resolve()
        if root.parent == root or str(root).startswith(("\\\\", "//")):
            QMessageBox.warning(self, "目录无效", "请选择具体的本地微信文件目录，不能选择磁盘根目录或网络目录。")
            return False
        self.file_root.setText(str(root))
        self.host.db.set_setting("wechat_file_root", str(root))
        return True

    def clear_file_root(self):
        self.file_root.clear()
        self.host.db.set_setting("wechat_file_root", "")

    def bind_attachment(self, attachment_id, action=None):
        path, _ = QFileDialog.getOpenFileName(self, "选择实际附件（无需预先建立索引）", self.file_root.text())
        if path:
            self.launch("bind", attachment=attachment_id, path=path, action=action)

    def open_attachment(self):
        row = self.attachment_table.currentRow()
        if 0 <= row < len(self.attachments):
            self.locate_attachment(self.attachments[row]["id"], "open")

    def index_selected_attachments(self):
        rows = self.attachment_table.selectionModel().selectedRows()
        ids = [self.attachments[row.row()]["id"] for row in rows if 0 <= row.row() < len(self.attachments)]
        self.index_attachment_contents(ids)

    def index_attachment_contents(self, ids):
        if not ids:
            QMessageBox.information(self, "未选择附件", "请先选择需要建立正文索引的附件。")
            return
        if self.launch("attachment_index", attachments=list(ids)):
            self.host.index_result_button.setEnabled(False)
            self.host.header_status.setText("正在索引附件正文")
            self.cancel_button.setText("取消")
            self.cancel_button.setToolTip("当前文件解析或向量批次完成后取消；保留已提交的索引")

    def update_semantic_index(self):
        if self.host._model_task_running():
            self.status.setText("请等待当前搜索或索引任务完成。")
            return
        from .semantic_maintenance import maintenance_targets
        sources, paths = maintenance_targets(self.host.db)
        if not sources:
            QMessageBox.information(self, "尚无微信资料", "请先导入或同步需要检索的微信资料。")
            return
        answer = QMessageBox.question(self, "更新本地语义索引",
            f"将更新全部已导入的 {len(sources)} 个微信会话及 {len(paths)} 个已索引附件正文。\n\n"
            "使用本地缓存语义模型；不连接 Alpha，不下载附件，不修改原文件。已有附件正文会重新检查，"
            "旧版长文本会重新分块。可取消，已完成部分保留。\n\n是否继续？")
        if answer == QMessageBox.StandardButton.Yes and self.launch("semantic_update"):
            self.host.header_status.setText("正在更新微信语义索引")
            self.cancel_button.setText("取消")
            self.cancel_button.setToolTip("当前解析或向量批次结束后取消；保留已完成部分")

    def finish_semantic_update(self, result):
        self.host.header_status.setText("本地模式")
        self.cancel_button.setText("暂停")
        self.cancel_button.setToolTip("")
        summary = result.get("file_summary", {})
        partial = bool(result.get("error") or result.get("issues"))
        title = "已取消，完成部分已保留" if result.get("cancelled") else "部分完成" if partial else "本地语义索引已更新"
        self.status.setText(result.get("error") or (
            f"{title} · 会话 {result.get('sources_done', 0)}/{result.get('sources_total', 0)}"
            f" · 更新正文 {summary.get('files_indexed', 0)} · 更新片段 {result.get('updated_units', 0)}"
            f" · {result['seconds']:.2f} 秒"))
        self.status.setToolTip("\n".join(result.get("issues", [])))
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0 if partial or result.get("cancelled") else 1)
        # Also recheck after a partial task whose last step raised unexpectedly.
        self.host.search_index_dirty.emit()
        self.host.refresh_stats()
        self.refresh()
        self.host.show_selected_result()

    def finish_content_index(self, result):
        self.host.header_status.setText("本地模式")
        self.cancel_button.setText("暂停")
        self.cancel_button.setToolTip("")
        summary = result.get("summary", {})
        counts = (f"可检索 {result.get('ready', 0)} · 无正文 {result.get('empty', 0)}"
                  f" · 待处理 {result.get('issues', 0)} · 更新 {summary.get('files_indexed', 0)} 个文件")
        prefix = "已取消，完成部分已保留" if result.get("cancelled") else "附件正文索引完成"
        self.status.setText(result.get("error") or f"{prefix} · {counts} · {result['seconds']:.2f} 秒")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0 if result.get("error") or result.get("cancelled") else 1)
        self.host.search_index_dirty.emit()
        self.host.refresh_stats()
        self.refresh()
        self.host.show_selected_result()
        details = [f"{row['name'] or row['attachment']}：{row['detail']}" for row in result.get("outcomes", [])
                   if row["state"] in {"failed", "unavailable", "incomplete"}]
        if result.get("error") or details:
            box = QMessageBox(self)
            box.setWindowTitle("附件正文索引")
            box.setTextFormat(Qt.TextFormat.PlainText)
            box.setText(result.get("error") or "部分附件未完成，已成功的索引会保留。")
            if details:
                box.setDetailedText("\n".join(details))
            box.exec()

    def locate_attachment(self, attachment_id, action):
        if not self.launch("locate", attachment=attachment_id, action=action):
            QMessageBox.information(self, "任务进行中", "请等待当前搜索或索引任务完成后再打开附件。")
            return
        for button in self.host.file_actions:
            button.setEnabled(False)
        self.host.header_status.setText("正在定位微信附件")

    def finish_attachment(self, result):
        self.host.header_status.setText("本地模式")
        self.host.show_selected_result()
        if result.get("cancelled"):
            self.status.setText("附件定位已取消")
            return
        if result.get("error"):
            self.status.setText(result["error"])
            box = QMessageBox(self)
            box.setWindowTitle("附件暂时无法打开")
            box.setTextFormat(Qt.TextFormat.PlainText)
            box.setText(result["error"])
            root = box.addButton("设置微信文件目录", QMessageBox.ButtonRole.ActionRole)
            bind = box.addButton("关联已保存文件", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() == root and self.choose_file_root():
                self.locate_attachment(result["attachment"], result.get("action"))
            elif box.clickedButton() == bind:
                self.bind_attachment(result["attachment"], result.get("action"))
            return
        if result.get("candidates"):
            path, ok = QInputDialog.getItem(self, "确认本地附件",
                "以下为同名候选，尚未证明是同一份文件。请确认实际附件：", result["candidates"], 0, False)
            if ok:
                self.launch("bind", attachment=result["attachment"], path=path, action=result.get("action"))
            return
        self.status.setText(f"附件已定位 · {result['seconds']:.2f} 秒")
        self.host.search_index_dirty.emit()
        self.refresh_attachments()
        self.host.show_selected_result()
        if result.get("action"):
            self.host.perform_file_action(result["path"], result["action"])

    def remove_source(self):
        row = self.source_table.currentRow()
        if 0 <= row < len(self.sources) and QMessageBox.question(self, "移除会话索引", "删除该会话在 LocalMind 中的消息、向量与同步检查点？微信和本地文件不会改变。") == QMessageBox.StandardButton.Yes:
            self.auto.setChecked(False)
            self.launch("remove", source=self.sources[row]["id"])

    @Slot(object)
    def finished(self, result):
        self.busy = False
        self.host._update_result_controls()
        self.elapsed_timer.stop()
        self.elapsed.setText(f"{result['seconds']:.2f} 秒")
        for widget in self.controls:
            widget.setEnabled(True)
        self.keyword.setEnabled(self.single_session.isChecked())
        self.sessions.setEnabled(self.single_session.isChecked())
        self.cancel_button.setEnabled(False)
        if result["kind"] == "semantic_update":
            self.finish_semantic_update(result)
            self.host._queue_file_events([])
            return
        if result["kind"] == "attachment_index":
            self.finish_content_index(result)
            self.host._queue_file_events([])
            return
        if result["kind"] in {"locate", "bind"}:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0 if result.get("error") else 1)
            self.finish_attachment(result)
            self.host._queue_file_events([])
            return
        if result["kind"] == "scope_metadata":
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0 if result.get("error") else 1)
            if self.scope_dialog and result.get("connection_key") == digest(self.url.text() + "\0" + self.token.text()):
                self.scope_dialog.receive_metadata(result)
            self.status.setText(result.get("error", "联系人元数据读取结束；尚未建立索引"))
            self.host._queue_file_events([])
            return
        if result["kind"] != "bulk" or not result.get("job"):
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0 if result.get("error") else 1)
        if result["kind"] in {"sessions", "scope", "bulk"}:
            self.accounts.blockSignals(True)
            self.accounts.clear()
            self.accounts.addItem("当前账号（自动）", None)
            for account in result.get("accounts", []):
                label = account["name"] + ("" if account["ready"] else "（未就绪）")
                self.accounts.addItem(label, account["id"])
            index = self.accounts.findData(result.get("account"))
            self.accounts.setCurrentIndex(max(0, index))
            self.accounts.blockSignals(False)
            if self.active_scope and self.active_scope["account_id"] != self.accounts.currentData():
                self.active_scope = None
                self.confirmed_target = None
                self.scope_label.setText("选择会话")
        if result.get("error"):
            self.auto.setChecked(False)
            self.status.setText(result["error"] + f"（{result['seconds']:.2f} 秒）")
            if result["kind"] in {"sessions", "scope", "bulk"}:
                self.set_connection_status("任务已暂停" if result.get("cancelled") else "连接 / 任务异常",
                                           "idle" if result.get("cancelled") else "error")
        elif result["kind"] in {"sessions", "scope"}:
            self.session_cache = result["sessions"]
            self.filter_sessions()
            self.host.db.set_setting("weflow_url", self.url.text().strip())
            protocol = "Alpha" if result.get("protocol") == "alpha" else "WeFlow"
            self.set_connection_status(f"{protocol} 已连接", "ready")
            client = WeFlowClient(self.url.text(), account=self.accounts.currentData())
            job = load_batch(self.host.db, client)
            if job:
                self.show_batch_progress({**batch_summary(job), "sessions": job["sessions"]})
                if job["state"] != "completed" and not self.single_session.isChecked():
                    self.sync_button.setText("继续建立索引")
            self.status.setText(f"已连接 {protocol}，读取 {len(result['sessions'])} 个会话 · {result['seconds']:.2f} 秒")
        elif result["kind"] == "bulk":
            self.session_cache = result["job"]["sessions"]
            self.filter_sessions()
            self.set_connection_status("Alpha 已连接" if result.get("protocol") == "alpha" else "WeFlow 已连接", "ready")
            self.host.db.set_setting("weflow_url", self.url.text().strip())
            if result["state"] == "completed":
                self.confirmed_target = self.target()
                self.sync_button.setText("更新所选索引")
                self.status.setText(f"索引完成 · {result['succeeded_sessions']} 个会话已处理 · 更新 {result['updated_units']} 块")
                if result.get("new_sessions"):
                    self.status.setText(self.status.text() + f" · 新发现 {result['new_sessions']} 个会话，未纳入索引")
            else:
                self.auto.setChecked(False)
                self.sync_button.setText("继续建立索引")
                self.sync_button.setToolTip("沿用任务截止时间，跳过已完成会话，继续待处理和失败会话；取消增量同步可重新检查全部会话")
                self.status.setText("已暂停，完成部分已保存" if result["state"] == "paused" else f"部分会话失败或未完成 · 失败 {result['failed_sessions']} 个 · 部分完成 {result.get('partial_sessions', 0)} 个 · 待重试消息 {result.get('failed_messages', 0)} 条")
                if result["state"] == "partial" and not result["failed_sessions"] and result.get("partial_sessions"):
                    self.status.setText(f"正常消息已索引 · {result['partial_sessions']} 个会话部分完成 · {result['failed_messages']} 条消息待重试")
        else:
            prefix = f"部分完成 · 待重试 {result['failed_messages']} 条" if result.get("failed_messages") else "完成"
            self.status.setText(f"{prefix} · 读取 {result.get('received', 0)} 条 · 更新 {result.get('updated_units', 0)} 块 · 复用 {result.get('reused_units', 0)} 块 · {result['seconds']:.2f} 秒 · {result.get('model', '')}")
            if result["kind"] == "sync":
                self.confirmed_target = None if result.get("failed_messages") else self.target()
                if result.get("failed_messages"):
                    self.auto.setChecked(False)
        if result.get("changed") or result["kind"] not in {"sessions", "scope"}:
            self.host.search_index_dirty.emit()
            self.host.refresh_stats()
        self.refresh()
        if self.auto.isChecked():
            self.timer.start()
        self.host._queue_file_events([])
        if result["kind"] == "scope" and not result.get("error"):
            if result.get("connection_key") == digest(self.url.text() + "\0" + self.token.text()):
                self.choose_scope()
            else:
                self.status.setText("连接配置已改变，请重新选择索引范围。")

    def refresh(self):
        self.sources = ChatStore(self.host.db).sources()
        attachments = self.host.db.conn.execute("SELECT COUNT(*) FROM chat_attachments").fetchone()[0]
        self.overview.setText(f"{len(self.sources)} 个会话 · {sum(row['messages'] for row in self.sources)} 条消息 · {attachments} 个附件")
        self.data_tabs.setTabText(0, f"附件 · {attachments}")
        self.data_tabs.setTabText(1, f"已索引会话 · {len(self.sources)}")
        if not self.busy and not self.host.db.conn.execute("SELECT 1 FROM settings WHERE key LIKE 'weflow_batch:%' LIMIT 1").fetchone():
            self.job_rows = []
            self.task_table.setRowCount(0)
            self.batch_counts.setText("待开始")
            self.snapshot_time.clear()
        self.source_table.setRowCount(len(self.sources))
        for i, source in enumerate(self.sources):
            for j, key in enumerate(["account", "conversation", "messages", "kind"]):
                value = {"weflow_api": "本机 API", "import": "离线导入"}.get(source[key], source[key]) if key == "kind" else source[key]
                self.source_table.setItem(i, j, QTableWidgetItem(str(value)))
        self.refresh_attachments()

    def refresh_attachments(self):
        row = self.source_table.currentRow()
        source_id = self.sources[row]["id"] if 0 <= row < len(self.sources) else None
        self.attachments = ChatStore(self.host.db).attachments(source_id)
        self.attachment_table.setRowCount(len(self.attachments))
        states = {"manual": "人工确认", "path": "路径匹配", "hash": "哈希匹配", "missing": "未找到", "candidate": "同名待确认", "ambiguous": "多个候选"}
        for i, att in enumerate(self.attachments):
            state = states.get(att["link_state"], att["link_state"]) if att["file_id"] else states.get(att["link_state"], "未找到")
            if not att["file_id"] and att["link_state"] in {"hash", "path", "manual"}:
                state = "本地文件已移除"
            if not att["file_id"] and att["local_path"]:
                state = "已定位（打开时复核）"
            sender = f"转发者: {att['sender']}" if att["origin"] else att["sender"]
            indexed = content_status(att, include_embeddings=self.host.enable_embedding.isChecked(),
                                     model_name=self.host.db.get_setting("embedding_model"))
            for j, value in enumerate([att["name"], att["conversation"], sender, state, indexed, att["local_path"] or "未定位"]):
                item = QTableWidgetItem(str(value))
                detail = att["content_error"] if j == 4 else ""
                item.setToolTip(str(value) + ("\n" + att["origin"] if att["origin"] and j in {0, 2} else "")
                                + ("\n" + detail if detail else ""))
                self.attachment_table.setItem(i, j, item)

    def shutdown(self):
        self.timer.stop()
        self.elapsed_timer.stop()
        self.token.clear()
        self.thread.quit()
        return self.thread.wait(5000)
