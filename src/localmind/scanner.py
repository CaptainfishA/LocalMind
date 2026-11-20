from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from .config import IMAGE_EXTENSIONS, MAX_FILE_SIZE_BYTES, SCAN_FILE_LIMIT, SUPPORTED_EXTENSIONS
from .models import FileRecord


def _iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value).isoformat(timespec="microseconds")


def hash_file(path: Path, *, limit_bytes: int = MAX_FILE_SIZE_BYTES) -> str | None:
    try:
        if path.stat().st_size > limit_bytes:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def build_file_record(path: str | Path, *, include_hash: bool = False) -> FileRecord | None:
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None
    try:
        stat = file_path.stat()
    except OSError:
        return None
    needs_ocr = ext in IMAGE_EXTENSIONS
    return FileRecord(
        path=str(file_path),
        name=file_path.name,
        ext=ext.lstrip("."),
        size=int(stat.st_size),
        created_at=_iso_from_timestamp(stat.st_ctime),
        modified_at=_iso_from_timestamp(stat.st_mtime),
        file_hash=hash_file(file_path) if include_hash else None,
        parse_status="pending",
        ocr_status="pending" if needs_ocr else "not_required",
        needs_ocr=needs_ocr,
    )


def scan_folder(
    root: str | Path,
    *,
    limit: int | None = SCAN_FILE_LIMIT,
    on_error: Callable[[OSError], None] | None = None,
) -> Iterator[FileRecord]:
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Folder does not exist: {root_path}")

    seen = 0
    for dirpath, dirnames, filenames in os.walk(root_path, onerror=on_error):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in filenames:
            if limit is not None and seen >= limit:
                return
            path = Path(dirpath) / filename
            record = build_file_record(path)
            if record is None:
                if path.suffix.lower() in SUPPORTED_EXTENSIONS and on_error is not None:
                    on_error(OSError(f"Unable to read file metadata: {path}"))
                continue
            seen += 1
            yield record
