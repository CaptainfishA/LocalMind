import importlib.util
import json
import os
import sqlite3
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.models import SearchQuery
from test_chat_content_index import prepare, run
from test_weflow_v04 import searcher


spec = importlib.util.spec_from_file_location(
    "retrieval_live", Path(__file__).parents[1] / "benchmarks" / "verify_retrieval_live.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_memory_snapshot_preserves_candidates_and_never_copies_secrets(tmp_path):
    source = LocalMindDB(tmp_path / "original.db")
    target = LocalMindDB(":memory:")
    try:
        ids, files = prepare(source, tmp_path, [("a.txt", "蓝色印章验收"), ("b.txt", "备用电源维护")])
        assert run(source, tmp_path, ids)["ready"] == 2
        source.set_setting("weflow_token", "private-not-copied")
        rows = probe.candidates(source.db_path)
        assert len(rows) == 2
        before = source.conn.total_changes
        counts, cache = probe.snapshot(source.db_path, target, [r["id"] for r in rows])
        assert counts["candidate_files"] == 2 and counts["body_chunks"] == 2
        assert counts["context_chunks"] >= 2 and counts["vectors"] >= 2
        assert cache is None and target.get_setting("weflow_token") is None
        assert not target.conn.execute("PRAGMA foreign_key_check").fetchall()
        result = searcher(target, mode="keyword").search(SearchQuery("蓝色印章", scope="chat_files"))
        assert result[0].path == str(files["a.txt"])
        assert result[0].file_state == "metadata_matches"
        assert source.conn.total_changes == before
        conn = probe.readonly(source.db_path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM files")
        finally:
            conn.close()
    finally:
        source.close()
        target.close()


def test_validation_ui_open_requires_click_and_report_is_redacted(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    source = LocalMindDB(tmp_path / "original.db")
    window = None
    try:
        ids, files = prepare(source, tmp_path, [("私密文件名称.txt", "私密文件正文验收")])
        assert run(source, tmp_path, ids)["ready"] == 1
        results = searcher(source, mode="keyword").search(SearchQuery("验收", scope="chat_files"))
        output = tmp_path / "redacted.json"
        window = probe.AcceptanceDialog(source.db_path, output)
        initial = json.loads(output.read_text(encoding="utf-8"))
        assert initial["phase"] == "waiting_for_query" and not initial["trials"]
        assert "报告已保存" in window.receipt.text()
        window.query.setPlainText("这里是用户私密查询")
        opened = []
        monkeypatch.setattr(os, "startfile", lambda path: opened.append(path))
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        window.on_result({
            "results": results,
            "metrics": {"modes": {mode: {"rank": 1, "milliseconds": 2.0}
                                  for mode in ("keyword", "semantic", "hybrid")}},
        })
        assert not opened and window.open_button.isEnabled()
        window.show()
        app.processEvents()
        screenshots = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
        if screenshots:
            Path(screenshots).mkdir(parents=True, exist_ok=True)
            window.grab().save(str(Path(screenshots) / "retrieval-acceptance-synthetic.png"))
        window.open_button.click()
        assert opened == [str(files["私密文件名称.txt"])]
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["trials"][0]["user_confirmed_open"] is True
        text = output.read_text(encoding="utf-8")
        assert "私密" not in text and str(tmp_path) not in text
        assert report["phase"] == "results_ready"
        window.update_target()
        assert not window.open_button.isEnabled()
    finally:
        if window:
            window.close()
        source.close()
        app.processEvents()


def test_worker_marks_hash_fallback_as_non_semantic(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = LocalMindDB(tmp_path / "original.db")
    try:
        ids, _ = prepare(source, tmp_path, [("sample.txt", "验收正文")])
        assert run(source, tmp_path, ids)["ready"] == 1
        rows = probe.candidates(source.db_path)
        monkeypatch.setattr(probe, "EmbeddingService", lambda **kwargs: EmbeddingService(force_hash=True))
        worker = probe.SearchWorker(source.db_path, [rows[0]["id"]], rows[0]["id"], "验收")
        ready, errors = [], []
        worker.ready.connect(ready.append)
        worker.failed.connect(errors.append)
        worker.run()
        assert not errors and len(ready) == 1
        metrics = ready[0]["metrics"]
        assert not metrics["real_cached_model"]
        assert metrics["modes"]["keyword"]["rank"] == 1
        assert metrics["modes"]["semantic"]["status"] == "cached_model_unavailable"
        assert metrics["modes"]["hybrid"]["status"] == "cached_model_unavailable"
        assert worker.query == ""
    finally:
        source.close()
        app.processEvents()


def test_live_worker_passes_evidence_and_production_limit(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = LocalMindDB(tmp_path / "original.db")
    try:
        ids, _ = prepare(source, tmp_path, [("sample.txt", "只在文件里的验证码")])
        assert run(source, tmp_path, ids)["ready"] == 1
        rows = probe.candidates(source.db_path)
        monkeypatch.setattr(probe, "EmbeddingService", lambda **kwargs: EmbeddingService(force_hash=True))
        worker = probe.SearchWorker(source.db_path, [rows[0]["id"]], rows[0]["id"], "验证码", evidence="chat")
        ready, errors = [], []
        worker.ready.connect(ready.append)
        worker.failed.connect(errors.append)
        worker.run()
        assert not errors and len(ready) == 1
        assert not ready[0]["results"]
        assert ready[0]["metrics"]["evidence"] == "chat"
        assert ready[0]["metrics"]["search_limit"] == 20
    finally:
        source.close()
        app.processEvents()


def test_report_path_does_not_follow_working_directory_changes(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = LocalMindDB(tmp_path / "original.db")
    monkeypatch.chdir(tmp_path)
    dialog = probe.AcceptanceDialog(source.db_path, Path("receipt.json"))
    other = tmp_path / "other"
    other.mkdir()
    try:
        monkeypatch.chdir(other)
        dialog.on_failure("SyntheticFailure")
        report = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert report["phase"] == "search_failed" and len(report["trials"]) == 1
        assert not (other / "receipt.json").exists()
    finally:
        dialog.close()
        source.close()
        app.processEvents()


def test_failed_save_keeps_results_visible_and_retry_persists(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = LocalMindDB(tmp_path / "original.db")
    ids, _ = prepare(source, tmp_path, [("sample.txt", "验收样本")])
    assert run(source, tmp_path, ids)["ready"] == 1
    path = tmp_path / "receipt.json"
    dialog = probe.AcceptanceDialog(source.db_path, path)
    original_writer = probe.write_report
    original_bytes = path.read_bytes()
    def fail(*args):
        raise OSError("private filesystem details")
    try:
        monkeypatch.setattr(probe, "write_report", fail)
        dialog.on_result({"results": [], "metrics": {"modes": {
            mode: {"rank": 2, "milliseconds": 12.0} for mode in ("keyword", "semantic", "hybrid")
        }}})
        assert "第 2 名" in dialog.status.text() and "保存失败" in dialog.receipt.text()
        assert "private" not in dialog.receipt.text()
        assert len(dialog.report["trials"]) == 1 and path.read_bytes() == original_bytes
        dialog.show()
        app.processEvents()
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
        assert not dialog.close()
        assert dialog.report["phase"] == "results_ready"
        monkeypatch.setattr(probe, "write_report", original_writer)
        dialog.save_button.click()
        assert json.loads(path.read_text(encoding="utf-8"))["trials"][0]["modes"]["hybrid"]["rank"] == 2
        assert "报告已保存" in dialog.receipt.text()
    finally:
        monkeypatch.setattr(probe, "write_report", original_writer)
        dialog.close()
        source.close()
        app.processEvents()


def test_atomic_report_short_write_does_not_corrupt_previous_receipt(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    probe.write_report(path, {"trials": []})
    original = path.read_bytes()
    factory = probe.QSaveFile
    class ShortWrite:
        def __init__(self, name):
            self.target = factory(name)
        def __getattr__(self, name):
            return getattr(self.target, name)
        def write(self, payload):
            return self.target.write(payload[:2])
    monkeypatch.setattr(probe, "QSaveFile", ShortWrite)
    with pytest.raises(OSError, match="report_write_failed"):
        probe.write_report(path, {"trials": [1, 2]})
    assert path.read_bytes() == original


def test_existing_report_is_not_overwritten_at_launch(tmp_path, monkeypatch, capsys):
    path = tmp_path / "receipt.json"
    probe.write_report(path, {"previous_result": True})
    original = path.read_bytes()
    monkeypatch.setattr("sys.argv", ["verify", "--report", str(path)])
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 1 and "report_already_exists" in capsys.readouterr().out
    assert path.read_bytes() == original


def test_visibility_check_and_no_reopening_after_close(tmp_path):
    from PySide6.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    source = LocalMindDB(tmp_path / "original.db")
    path = tmp_path / "receipt.json"
    dialog = probe.AcceptanceDialog(source.db_path, path)
    try:
        dialog.ensure_visible()
        QTest.qWait(350)
        assert dialog.isVisible()
        assert json.loads(path.read_text(encoding="utf-8"))["window_exposed"] is True
        dialog.close()
        assert not dialog.isVisible()
        before = path.read_bytes()
        dialog.ensure_visible()
        dialog.record_visibility()
        assert not dialog.isVisible() and path.read_bytes() == before
    finally:
        dialog.close()
        source.close()
        app.processEvents()
