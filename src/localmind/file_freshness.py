"""Bounded metadata checks for search snapshots; never reads document content."""
from __future__ import annotations

import os
import stat
import time
from datetime import datetime
from pathlib import Path


STATE_LABELS = {
    "unchecked": "尚未核对",
    "metadata_matches": "文件元数据未变化",
    "changed": "文件已变化，索引待核对",
    "missing": "文件已移除或移动",
    "unreadable": "文件暂不可读取",
    "unsupported": "路径未核对",
    "index_failed": "正文解析失败，索引待核对",
    "snapshot_changed": "索引已更新，结果需重新检索",
}
WARNING_STATES = {"changed", "missing", "unreadable", "index_failed", "snapshot_changed"}


def _is_remote_drive(path):
    if os.name != "nt":
        return False
    import ctypes
    return ctypes.windll.kernel32.GetDriveTypeW(path.anchor) == 4


def inspect_file(record, *, directories=None):
    """Metadata equality is not a content hash check. Reparse points are skipped."""
    value = record["path"]
    if (not isinstance(value, str) or value.startswith(("\\\\", "//")) or "://" in value
            or "\x00" in value or (os.name == "nt" and ":" in value[2:])):
        return "unsupported"
    path = Path(value)
    if not path.is_absolute() or len(path.parts) > 64 or _is_remote_drive(path):
        return "unsupported"
    directories = {} if directories is None else directories
    try:
        # Check each parent before descending, so an SMB junction is not followed.
        for parent in reversed(path.parents):
            key = os.path.normcase(str(parent))
            info = directories.get(key)
            if info is None:
                info = parent.lstat()
                directories[key] = info
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                return "unsupported"
            if not stat.S_ISDIR(info.st_mode):
                return "missing"
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return "unsupported"
        if not stat.S_ISREG(info.st_mode):
            return "missing"
        modified = datetime.fromtimestamp(info.st_mtime).isoformat(timespec="microseconds")
        return "metadata_matches" if info.st_size == record["size"] and modified == record["modified_at"] else "changed"
    except (FileNotFoundError, NotADirectoryError):
        return "missing"
    except (OSError, ValueError, OverflowError):
        return "unreadable"


def annotate_results(db, results, *, max_files=100, budget_seconds=0.1, clock=time.monotonic):
    """Check only returned files. The budget gates new checks, not an in-flight stat."""
    deadline = clock() + budget_seconds
    checked, directories = {}, {}
    for result in results:
        if result.result_kind != "file":
            continue
        key = (result.file_id, result.path, result.modified_at)
        if key in checked:
            result.file_state = checked[key]
            continue
        if len(checked) >= max_files or clock() >= deadline:
            result.file_state = "unchecked"
            continue
        record = db.get_file(result.file_id)
        if record is None:
            state = "snapshot_changed"
        elif record["path"] != result.path or record["modified_at"] != result.modified_at:
            state = "snapshot_changed"
        else:
            state = inspect_file(record, directories=directories)
            if state == "metadata_matches" and record["parse_status"] == "failed":
                state = "index_failed"
        checked[key] = state
        result.file_state = state
    return len(checked)
