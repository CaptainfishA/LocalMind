from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "LocalMind"
DB_FILENAME = "localmind.db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 30
EMBEDDING_BATCH_SIZE = 64
INDEX_FILE_BATCH_SIZE = 32
INDEX_CHUNK_BATCH_SIZE = 256
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
PDF_OCR_PAGE_LIMIT = 10
# V0.2 no longer truncates a selected folder after an arbitrary number of files.
# The UI still recommends indexing user-selected folders instead of a whole drive.
SCAN_FILE_LIMIT: int | None = None

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".docx",
    ".xlsx",
    ".pptx",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv"}


def default_data_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME
    return Path.home() / ".localmind"


def default_db_path() -> Path:
    return default_data_dir() / DB_FILENAME


def default_model_cache_dir() -> Path:
    return default_data_dir() / "models"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
