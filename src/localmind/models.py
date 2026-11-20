from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class FileRecord:
    path: str
    name: str
    ext: str
    size: int
    created_at: str
    modified_at: str
    file_hash: str | None
    scan_status: str = "scanned"
    parse_status: str = "pending"
    ocr_status: str = "not_required"
    needs_ocr: bool = False
    error: str | None = None
    id: int | None = None


@dataclass(slots=True)
class TextChunk:
    chunk_text: str
    source_type: str
    page_no: int | None = None
    sheet_name: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    ocr_confidence: float | None = None
    file_id: int | None = None
    id: int | None = None
    created_at: str = ""


@dataclass(slots=True)
class OcrResult:
    text: str
    status: str
    page_no: int | None = None
    confidence: float | None = None
    error: str | None = None
    created_at: str = ""


@dataclass(slots=True)
class SearchQuery:
    text: str
    file_type: str | None = None
    path_contains: str | None = None
    modified_after: str | None = None
    modified_before: str | None = None
    limit: int = 20
    scope: str = "all"
    chat_platform: str | None = None
    conversation: str | None = None
    sender: str | None = None
    evidence: str = "all"
    infer_dates: bool = True


@dataclass(frozen=True, slots=True)
class ChatSourceReference:
    """Identity of the message anchoring an indexed chat context, not its author."""

    message_id: int
    attachment_id: int | None
    platform: str
    account: str
    source_key: str
    external_id: str
    conversation: str
    sender: str
    sent_at: str
    attachment_name: str = ""
    origin: str = ""

    @property
    def identity(self):
        return (self.platform, self.account, self.source_key, self.external_id,
                self.message_id, self.attachment_id, self.attachment_name, self.origin)


@dataclass(slots=True)
class SearchEvidence:
    chunk_id: int
    source_type: str
    snippet: str
    matched_terms: tuple[str, ...]
    page_no: int | None = None
    sheet_name: str | None = None
    query_clause: str = ""
    chat_source: ChatSourceReference | None = None
    refined: bool = False


@dataclass(slots=True)
class SearchResult:
    file_id: int
    chunk_id: int | None
    name: str
    path: str
    ext: str
    modified_at: str
    score: float
    keyword_score: float
    semantic_score: float
    fused_score: float
    reranker_score: float | None
    snippet: str
    source_type: str
    match_reason: str = ""
    page_no: int | None = None
    sheet_name: str | None = None
    is_ocr: bool = False
    ocr_confidence: float | None = None
    result_kind: str = "file"
    chat_context: str = ""
    chat_message_id: int | None = None
    attachment_id: int | None = None
    link_state: str = ""
    file_state: str = "unchecked"
    source_query: SearchQuery | None = None
    supporting_evidence: tuple[SearchEvidence, ...] = ()
    chat_source: ChatSourceReference | None = None
    identifier_query: str = ""
    identifier_matched: bool = False
    preview_anchor: str = ""


@dataclass(slots=True)
class IndexStats:
    files_total: int
    chunks_total: int
    parsed_files: int
    ocr_done: int
    ocr_pending: int
    vectors_total: int
    watched_folders: int
    db_size_bytes: int
    last_scan_at: str | None
