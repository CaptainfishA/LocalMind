from __future__ import annotations

from dataclasses import replace

from .models import TextChunk, utc_now_iso

TEXT_INDEX_VERSION = 1
CHUNK_MAX_CHARS = 384


def split_text(text: str, *, min_size: int = 160, max_size: int = CHUNK_MAX_CHARS, overlap: int = 64) -> list[tuple[str, int, int]]:
    if not 0 < min_size <= max_size or not 0 <= overlap < min_size:
        raise ValueError("Invalid chunk size or overlap")
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []
    if len(normalized) <= max_size:
        return [(normalized, 0, len(normalized))]

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_size, len(normalized))
        window = normalized[start:end]
        split_at = max(window.rfind("\n"), window.rfind("。"), window.rfind("；"), window.rfind(";"))
        if split_at >= min_size:
            end = start + split_at + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_text_block(
    text: str,
    *,
    source_type: str,
    page_no: int | None = None,
    sheet_name: str | None = None,
    ocr_confidence: float | None = None,
) -> list[TextChunk]:
    created_at = utc_now_iso()
    return [
        TextChunk(
            chunk_text=chunk,
            source_type=source_type,
            page_no=page_no,
            sheet_name=sheet_name,
            start_offset=start,
            end_offset=end,
            ocr_confidence=ocr_confidence,
            created_at=created_at,
        )
        for chunk, start, end in split_text(text)
    ]


def rechunk(chunks: list[TextChunk]) -> list[TextChunk]:
    output = []
    for chunk in chunks:
        for text, start, end in split_text(chunk.chunk_text):
            output.append(replace(chunk, chunk_text=text, id=None, file_id=None,
                start_offset=chunk.start_offset + start if chunk.start_offset is not None else None,
                end_offset=chunk.start_offset + end if chunk.start_offset is not None else None))
    return output
