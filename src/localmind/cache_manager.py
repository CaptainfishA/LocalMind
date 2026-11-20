from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import default_model_cache_dir

CACHE_MARKER = ".localmind-model-cache"
CUSTOM_CACHE_FOLDER = "LocalMindModels"


@dataclass(frozen=True, slots=True)
class CacheStats:
    path: Path
    files: int
    bytes_total: int


def resolve_model_cache_dir(configured: str | Path | None = None) -> Path:
    path = Path(configured).expanduser() if configured else default_model_cache_dir()
    return path.resolve()


def prepare_model_cache_dir(path: str | Path | None = None) -> Path:
    cache_dir = resolve_model_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / CACHE_MARKER
    if not marker.exists():
        marker.write_text("LocalMind managed model cache\n", encoding="ascii")
    return cache_dir


def prepare_custom_cache_dir(parent: str | Path) -> Path:
    selected = Path(parent).expanduser().resolve()
    cache_dir = (
        selected
        if selected.name == CUSTOM_CACHE_FOLDER and (selected / CACHE_MARKER).is_file()
        else selected / CUSTOM_CACHE_FOLDER
    )
    return prepare_model_cache_dir(cache_dir)


def model_cache_stats(path: str | Path | None = None) -> CacheStats:
    cache_dir = resolve_model_cache_dir(path)
    if not cache_dir.exists():
        return CacheStats(path=cache_dir, files=0, bytes_total=0)
    files = 0
    bytes_total = 0
    for item in cache_dir.rglob("*"):
        try:
            if item.is_file() and item.name != CACHE_MARKER:
                files += 1
                bytes_total += item.stat().st_size
        except OSError:
            continue
    return CacheStats(path=cache_dir, files=files, bytes_total=bytes_total)


def clear_model_cache(path: str | Path | None = None) -> CacheStats:
    cache_dir = resolve_model_cache_dir(path)
    before = model_cache_stats(cache_dir)
    if not cache_dir.exists():
        return before
    if cache_dir != default_model_cache_dir().resolve() and not (cache_dir / CACHE_MARKER).is_file():
        raise ValueError("拒绝清理未由 LocalMind 管理的目录")
    if cache_dir.parent == cache_dir or cache_dir == Path.home().resolve():
        raise ValueError("拒绝清理过于宽泛的目录")

    for item in cache_dir.iterdir():
        if item.name == CACHE_MARKER:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    prepare_model_cache_dir(cache_dir)
    return before


def format_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in {"B", "KB"} else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"
