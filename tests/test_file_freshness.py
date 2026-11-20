import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from localmind import file_freshness
from localmind.db import LocalMindDB
from test_weflow_v04 import index_file, searcher


@pytest.fixture
def indexed(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    path = tmp_path / "sample.txt"
    record = index_file(db, path, "独有验收验证码")
    yield db, path, record
    db.close()


def test_search_marks_modified_deleted_and_reindexed_file(indexed):
    db, path, record = indexed
    service = searcher(db, mode="keyword")
    original = service.search("独有验收验证码")[0]
    assert original.file_state == "metadata_matches"
    assert service.last_diagnostics.timings_ms["file_check"] >= 0
    path.write_text("文件已经变化，独有验收验证码", encoding="utf-8")
    changed = service.search("独有验收验证码")[0]
    assert changed.file_state == "changed" and changed.score == original.score
    index_file(db, path, "重新索引独有验收验证码")
    assert service.search("独有验收验证码")[0].file_state == "metadata_matches"
    path.unlink()
    assert service.search("独有验收验证码")[0].file_state == "missing"


def test_failed_parse_and_snapshot_changed(indexed):
    db, path, record = indexed
    service = searcher(db, mode="keyword")
    result = service.search("独有验收验证码")[0]
    db.update_file_status(record["id"], parse_status="failed", error="synthetic failure")
    assert service.search("独有验收验证码")[0].file_state == "index_failed"
    db.conn.execute("UPDATE files SET modified_at=? WHERE id=?", ("changed", record["id"]))
    db.conn.commit()
    file_freshness.annotate_results(db, [result])
    assert result.file_state == "snapshot_changed"
    db.delete_file_by_path(path)
    file_freshness.annotate_results(db, [result])
    assert result.file_state == "snapshot_changed"


def test_budget_dedup_and_virtual_results(indexed, monkeypatch):
    db, path, record = indexed
    result = searcher(db, mode="keyword").search("独有验收验证码")[0]
    inspect = Mock(return_value="changed")
    monkeypatch.setattr(file_freshness, "inspect_file", inspect)
    duplicate = replace(result)
    virtual = replace(result, result_kind="message", file_state="unchecked")
    assert file_freshness.annotate_results(db, [result, duplicate, virtual], max_files=1) == 1
    assert result.file_state == duplicate.file_state == "changed"
    assert virtual.file_state == "unchecked" and inspect.call_count == 1
    assert file_freshness.annotate_results(db, [result], clock=iter([0, 1]).__next__) == 0
    assert result.file_state == "unchecked"


@pytest.mark.parametrize("value", ["relative.txt", r"\\server\share\a.txt", "//server/a.txt",
                                   "https://host/a.txt", "localmind-chat://test", "\x00"])
def test_unsupported_paths_never_stat(value, monkeypatch):
    monkeypatch.setattr(Path, "lstat", Mock(side_effect=AssertionError("must not stat")))
    assert file_freshness.inspect_file({"path": value}) == "unsupported"


def test_mapped_network_drive_never_stat(indexed, monkeypatch):
    db, path, record = indexed
    monkeypatch.setattr(file_freshness, "_is_remote_drive", lambda path: True)
    monkeypatch.setattr(Path, "lstat", Mock(side_effect=AssertionError("must not stat")))
    assert file_freshness.inspect_file(record) == "unsupported"


def test_reparse_parent_is_not_followed(indexed, monkeypatch):
    db, path, record = indexed
    def lstat(value):
        assert value != path, "must not descend into reparse parent"
        return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
    monkeypatch.setattr(Path, "lstat", lstat)
    assert file_freshness.inspect_file(record) == "unsupported"


def test_permission_error_is_a_status(indexed, monkeypatch):
    db, path, record = indexed
    monkeypatch.setattr(Path, "lstat", Mock(side_effect=PermissionError))
    assert file_freshness.inspect_file(record) == "unreadable"
