"""Opt-in local acceptance UI. Source index read-only; working index in memory."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import QIODevice, QSaveFile, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QStyle, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
)

from localmind.config import default_db_path
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.file_freshness import inspect_file
from localmind.models import SearchQuery
from localmind.reranker import RerankerService
from localmind.searcher import SearchService


def write_report(path, report):
    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    target = QSaveFile(str(path))
    if not target.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError("report_open_failed")
    if target.write(payload) != len(payload):
        target.cancelWriting()
        raise OSError("report_write_failed")
    if not target.commit():
        raise OSError("report_commit_failed")


def readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def candidates(path):
    with closing(readonly(path)) as conn:
        rows = conn.execute("""SELECT DISTINCT f.* FROM files f
            JOIN chat_attachments a ON a.file_id=f.id
            JOIN chat_messages m ON m.id=a.message_id
            JOIN chat_sources s ON s.id=m.source_id
            WHERE s.platform='wechat' ORDER BY f.name,f.id LIMIT 500""").fetchall()
    return [dict(row) for row in rows if inspect_file(row) == "metadata_matches"]


def _rows(conn, table, column, ids):
    rows = []
    ids = sorted(set(ids))
    for start in range(0, len(ids), 400):
        batch = ids[start:start + 400]
        rows.extend(conn.execute(
            f"SELECT * FROM {table} WHERE {column} IN ({','.join('?' for _ in batch)})",
            batch,
        ).fetchall())
        if len(rows) > 50000:
            raise ValueError("snapshot_limit")
    return rows


def snapshot(source, target, file_ids):
    """Copy only linked WeChat candidates, their context units and stored vectors."""
    with closing(readonly(source)) as conn:
        conn.execute("BEGIN")
        atts = [row for row in _rows(conn, "chat_attachments", "file_id", file_ids)]
        messages = _rows(conn, "chat_messages", "id", [r["message_id"] for r in atts])
        sources = _rows(conn, "chat_sources", "id", [r["source_id"] for r in messages])
        wechat_sources = {r["id"] for r in sources if r["platform"] == "wechat"}
        messages = [r for r in messages if r["source_id"] in wechat_sources]
        message_ids = {r["id"] for r in messages}
        atts = [r for r in atts if r["message_id"] in message_ids]
        units = _rows(conn, "chat_units", "attachment_id", [r["id"] for r in atts])
        all_file_ids = set(file_ids) | {r["owner_file_id"] for r in units}
        files = _rows(conn, "files", "id", all_file_ids)
        chunks = _rows(conn, "chunks", "file_id", file_ids)
        chunk_map = {r["id"]: r for r in chunks}
        chunk_map.update({r["id"]: r for r in _rows(conn, "chunks", "id", [r["chunk_id"] for r in units])})
        chunks = list(chunk_map.values())
        vectors = _rows(conn, "vectors", "chunk_id", chunk_map)
        if sum(len(r["chunk_text"].encode("utf-8")) for r in chunks) > 64 * 1024 * 1024:
            raise ValueError("snapshot_text_limit")
        for table, rows in (
            ("files", files), ("chat_sources", [r for r in sources if r["id"] in wechat_sources]),
            ("chat_messages", messages), ("chat_attachments", atts),
            ("chunks", chunks), ("vectors", vectors), ("chat_units", units),
        ):
            if not rows:
                continue
            available = {r[1] for r in target.conn.execute(f"PRAGMA table_info({table})")}
            columns = [c for c in rows[0].keys() if c in available]
            target.conn.executemany(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                [tuple(row[c] for c in columns) for row in rows],
            )
        by_id = {r["id"]: r for r in files}
        target.conn.executemany(
            "INSERT INTO chunk_fts(rowid,file_name,file_path,chunk_text,source_type) VALUES(?,?,?,?,?)",
            [(r["id"], by_id[r["file_id"]]["name"], by_id[r["file_id"]]["path"],
              r["chunk_text"], r["source_type"]) for r in chunks],
        )
        # Do not copy API credentials, watched folders, checkpoints or other settings.
        model = conn.execute("SELECT value FROM settings WHERE key='embedding_model'").fetchone()
        cache = conn.execute("SELECT value FROM settings WHERE key='model_cache_dir'").fetchone()
        target.conn.commit()
        if model:
            target.set_setting("embedding_model", model[0])
        return {
            "candidate_files": len(file_ids), "attachments": len(atts),
            "body_chunks": sum(r["source_type"] != "chat" for r in chunks),
            "context_chunks": len(units), "vectors": len(vectors),
        }, cache[0] if cache else None


class SearchWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, path, file_ids, expected, query, evidence="all"):
        super().__init__()
        self.path, self.file_ids, self.expected, self.query = path, file_ids, expected, query
        self.evidence = evidence

    def run(self):
        db = None
        try:
            before = time.perf_counter()
            db = LocalMindDB(":memory:")
            counts, cache = snapshot(self.path, db, self.file_ids)
            snapshot_seconds = time.perf_counter() - before
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            embedding = EmbeddingService(cache_dir=cache, local_files_only=True, device="cpu")
            before = time.perf_counter()
            actual_model = embedding.active_model_name
            real_model = actual_model == embedding.model_name
            model_seconds = time.perf_counter() - before
            service = SearchService(db, embedding_service=embedding,
                                    reranker_service=RerankerService(enabled=False))
            modes, shown = {}, []
            for mode in ("keyword", "semantic", "hybrid"):
                if not real_model and mode != "keyword":
                    modes[mode] = {"status": "cached_model_unavailable"}
                    continue
                service.retrieval_mode = mode
                before = time.perf_counter()
                results = service.search(SearchQuery(self.query, scope="chat_files", limit=20, evidence=self.evidence))
                rank = next((i for i, r in enumerate(results, 1) if r.file_id == self.expected), None)
                modes[mode] = {
                    "status": "ok", "rank": rank, "top5_hit": rank is not None and rank <= 5,
                    "result_count": len(results), "milliseconds": round((time.perf_counter() - before) * 1000, 2),
                    "semantic_model_matches": actual_model == service.indexed_embedding_model,
                    "timings_ms": service.last_diagnostics.timings_ms,
                }
                if mode == "hybrid" or not real_model:
                    shown = results[:10]
            self.ready.emit({
                "metrics": {**counts, "evidence": self.evidence, "search_limit": 20,
                            "snapshot_seconds": round(snapshot_seconds, 3),
                            "model_load_seconds": round(model_seconds, 3), "real_cached_model": real_model,
                            "model_name": embedding.model_name, "modes": modes},
                "results": shown,
            })
        except Exception as exc:
            self.failed.emit(type(exc).__name__)
        finally:
            if db is not None:
                db.close()
            self.query = ""


class AcceptanceDialog(QDialog):
    def __init__(self, db_path, report_path, *, evidence="content"):
        super().__init__()
        self.db_path, self.report_path = db_path, Path(report_path).resolve()
        self.rows = candidates(db_path)
        self.worker = None
        self.active_trial = None
        self.report = {
            "run_id": uuid.uuid4().hex[:12], "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
            "phase": "waiting_for_query",
            "window_exposed": None,
            "source_index_read_only": True, "working_index_in_memory": True,
            "credentials_used": False, "network_downloads_allowed": False,
            "contains_names_paths_queries_or_chat_content": False,
            "candidate_files": len(self.rows), "trials": [],
            "search_limit": 20, "device": "cpu", "reranker_enabled": False,
            "candidate_scope": "existing_metadata_matched_linked_wechat_files",
            "not_a_representative_accuracy_benchmark": True,
        }
        self.setWindowTitle("LocalMind 微信文件检索验收")
        self.resize(960, 670)
        layout = QVBoxLayout(self)
        note = QLabel("原索引只读 · 临时内存索引 · 无需 Token\n"
                      "选择目标文件并输入你真实会使用的模糊描述。报告仅保存排名、耗时和确认结果。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.receipt = QLabel()
        self.receipt.setWordWrap(True)
        self.receipt.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.receipt)
        layout.addWidget(QLabel("目标文件"))
        self.files = QComboBox()
        for row in self.rows:
            label = row["name"] + ("  [正文解析失败]" if row["parse_status"] == "failed" else "")
            self.files.addItem(label)
        layout.addWidget(self.files)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)
        self.files.currentIndexChanged.connect(self.update_target)
        self.update_target()
        self.kind = QComboBox()
        self.kind.addItems(["仅文件正文", "仅聊天线索", "全部线索"])
        self.kind.setCurrentIndex(("content", "chat", "all").index(evidence))
        self.kind.setToolTip("与主界面的线索筛选一致；本轮使用 CPU、关闭精排、最多返回 20 项")
        layout.addWidget(self.kind)
        self.query = QTextEdit()
        self.query.setPlaceholderText("输入模糊描述，尽量不要直接使用完整文件名")
        self.query.setMaximumHeight(100)
        layout.addWidget(self.query)
        actions = QHBoxLayout()
        self.run_button = QPushButton("开始验证")
        self.run_button.setEnabled(bool(self.rows))
        self.run_button.clicked.connect(self.run_query)
        actions.addWidget(self.run_button)
        self.open_button = QPushButton("打开目标文件并确认")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_target)
        actions.addWidget(self.open_button)
        self.save_button = QPushButton()
        self.save_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_button.setToolTip("重新保存脱敏验收报告")
        self.save_button.setAccessibleName("保存验收报告")
        self.save_button.setFixedSize(32, 32)
        self.save_button.clicked.connect(self.save_report)
        actions.addWidget(self.save_button)
        close = QPushButton("完成并关闭")
        close.clicked.connect(self.close)
        actions.addWidget(close)
        layout.addLayout(actions)
        self.status = QLabel(f"候选池：{len(self.rows)} 个元数据未变化的真实文件")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["排名", "文件", "匹配来源", "分数"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(2, 270)
        self.table.setColumnWidth(3, 70)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.save_report()

    def update_target(self):
        if self.rows:
            self.path_label.setText(self.rows[self.files.currentIndex()]["path"])
        if hasattr(self, "open_button"):
            self.open_button.setEnabled(False)
        self.active_trial = None

    def run_query(self):
        query = self.query.toPlainText().strip()
        if not query or len(query) > 2000:
            QMessageBox.information(self, "查询", "请输入 1 至 2000 字的模糊描述。")
            return
        row = self.rows[self.files.currentIndex()]
        self.active_trial = None
        self.open_button.setEnabled(False)
        for widget in (self.files, self.kind, self.query, self.run_button):
            widget.setEnabled(False)
        self.status.setText("正在准备内存索引并加载本地模型，随后比较三种检索模式……")
        self.report["phase"] = "searching"
        self.save_report()
        self.worker = SearchWorker(self.db_path, [r["id"] for r in self.rows], row["id"], query,
                                   evidence=("content", "chat", "all")[self.kind.currentIndex()])
        self.worker.ready.connect(self.on_result)
        self.worker.failed.connect(self.on_failure)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_result(self, payload):
        metrics = payload["metrics"]
        metrics["query_kind"] = ("file_content", "chat_context", "combined")[self.kind.currentIndex()]
        metrics["open_dispatch_succeeded"] = False
        metrics["user_confirmed_open"] = None
        self.report["trials"].append(metrics)
        self.active_trial = metrics
        self.report["phase"] = "results_ready"
        self.save_report()
        parts = []
        for mode, label in (("keyword", "关键词"), ("semantic", "语义"), ("hybrid", "混合")):
            result = metrics["modes"][mode]
            parts.append(f"{label}：第 {result['rank']} 名 / {result['milliseconds']:.1f} ms"
                         if result.get("rank") else f"{label}：未命中或未启用")
        shown_mode = "混合" if metrics.get("real_cached_model", True) else "关键词"
        self.status.setText("；".join(parts) + f"\n{shown_mode}检索前 10 项")
        self.table.setRowCount(len(payload["results"]))
        expected = self.rows[self.files.currentIndex()]["id"]
        for i, result in enumerate(payload["results"]):
            values = [str(i + 1), ("[目标] " if result.file_id == expected else "") + result.name,
                      result.match_reason, f"{result.score:.3f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(i, col, item)
        self.open_button.setEnabled(True)

    def on_failure(self, error_type):
        self.report["trials"].append({"failure_type": error_type})
        self.report["phase"] = "search_failed"
        self.save_report()
        self.status.setText(f"本轮未完成：{error_type}。未修改原索引。")

    def on_finished(self):
        for widget in (self.files, self.kind, self.query, self.run_button):
            widget.setEnabled(True)

    def open_target(self):
        if self.active_trial is None:
            return
        row = self.rows[self.files.currentIndex()]
        path = Path(row["path"])
        if path.suffix.lower() not in {".txt", ".md", ".csv", ".docx", ".xlsx", ".pptx", ".pdf", ".png", ".jpg", ".jpeg"}:
            QMessageBox.information(self, "未打开", "验收窗口只打开常见文档和图片，不执行程序或脚本。")
            return
        if inspect_file(row) != "metadata_matches":
            QMessageBox.information(self, "未打开", "文件已变化或无法核对，请关闭窗口后重新验收。")
            return
        try:
            os.startfile(str(path))
        except OSError as exc:
            self.active_trial["open_error_type"] = type(exc).__name__
            self.save_report()
            QMessageBox.information(self, "未打开", "系统未能打开该文件，请检查文件关联。")
            return
        self.active_trial["open_dispatch_succeeded"] = True
        answer = QMessageBox.question(
            self, "打开验收", "请查看刚打开的文档。\n是否确实打开了你选择的目标文件？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self.active_trial["user_confirmed_open"] = answer == QMessageBox.StandardButton.Yes
        self.save_report()

    def save_report(self):
        try:
            write_report(self.report_path, self.report)
        except OSError:
            self.receipt.setText(f"验收 {self.report['run_id']} · 报告保存失败，结果仍在当前窗口")
            self.receipt.setStyleSheet("color: #b42318;")
            print("retrieval_acceptance: report_save_failed", flush=True)
            return False
        self.receipt.setText(f"验收 {self.report['run_id']} · 报告已保存 · {len(self.report['trials'])} 次记录")
        self.receipt.setStyleSheet("color: #087f72;")
        print("retrieval_acceptance: redacted_report_updated", flush=True)
        return True

    def ensure_visible(self):
        if self.report["phase"] == "closed":
            return
        if not self.windowHandle() or not self.windowHandle().isExposed():
            # A hidden Windows STARTUPINFO can suppress the first native ShowWindow.
            self.hide()
            self.showNormal()
            self.raise_()
            self.activateWindow()
        QTimer.singleShot(250, self.record_visibility)

    def record_visibility(self):
        if self.report["phase"] == "closed":
            return
        exposed = bool(self.windowHandle() and self.windowHandle().isExposed() and self.isVisible())
        self.report["window_exposed"] = exposed
        self.save_report()
        print("retrieval_acceptance: window_exposed" if exposed else "retrieval_acceptance: window_not_exposed", flush=True)

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.status.setText("检索正在结束，请稍后再关闭。")
            event.ignore()
            return
        previous = self.report["phase"]
        self.report["temporary_index_closed"] = True
        self.report["phase"] = "closed"
        if not self.save_report() and QMessageBox.question(
            self, "报告尚未保存", "报告保存失败，关闭后未保存的结果将丢失。\n仍然关闭吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            self.report["phase"] = previous
            event.ignore()
            return
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=default_db_path())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence", choices=("content", "chat", "all"), default="content")
    args = parser.parse_args()
    args.report = args.report.resolve()
    if args.report.exists():
        print('{"failure_type":"report_already_exists"}', flush=True)
        raise SystemExit(1)
    app = QApplication.instance() or QApplication([])
    try:
        dialog = AcceptanceDialog(args.db, args.report, evidence=args.evidence)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        QTimer.singleShot(100, dialog.ensure_visible)
        print("retrieval_acceptance: window_ready", flush=True)
        app.exec()
    except Exception as exc:
        print(json.dumps({"failure_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    print("retrieval_acceptance: stopped", flush=True)


if __name__ == "__main__":
    main()
