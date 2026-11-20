"""Bounded, read-only lookup of missing content/context in recalled files."""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CompanionRequest:
    file_id: int
    source: str
    terms: tuple[str, ...]


@dataclass(slots=True)
class CompanionRecall:
    chunk_targets: dict[int, int] = field(default_factory=dict)
    requested_files: int = 0
    checked_files: int = 0
    file_limited: bool = False
    time_limited: bool = False


def recall_companions(db, requests: list[CompanionRequest], *, restricted=False,
                      max_files=32, per_file=2, budget_ms=50.0) -> CompanionRecall:
    if type(max_files) is not int or not 1 <= max_files <= 64:
        raise ValueError("Invalid companion file limit")
    if type(per_file) is not int or not 1 <= per_file <= 4:
        raise ValueError("Invalid companion chunk limit")
    if not math.isfinite(budget_ms) or not 0 <= budget_ms <= 1000:
        raise ValueError("Invalid companion time budget")
    if any(type(r.file_id) is not int or r.file_id <= 0 or r.source not in {"content", "chat"} for r in requests):
        raise ValueError("Invalid companion request")
    if len({r.file_id for r in requests}) != len(requests):
        raise ValueError("Duplicate companion file")
    report = CompanionRecall(requested_files=len(requests), file_limited=len(requests) > max_files)
    if not requests:
        return report
    deadline = time.perf_counter() + budget_ms / 1000

    def expired():
        if time.perf_counter() >= deadline:
            report.time_limited = True
        return report.time_limited

    # This connection belongs to the serial search worker. Always remove the
    # handler, including after an interrupted SELECT, before any other DB work.
    db.conn.set_progress_handler(expired, 1000)
    try:
        for request in requests[:max_files]:
            if expired():
                break
            terms = list(dict.fromkeys(term.strip().lower() for term in request.terms if term.strip()))[:16]
            if not terms:
                report.checked_files += 1
                continue
            scores = []
            for term in terms:
                column = "lower(c.chunk_text)" if term.lower() != term.upper() else "c.chunk_text"
                scores.append(f"(instr({column},?)>0)")
            if request.source == "content":
                source = "chunks c"
                condition = "c.file_id=? AND c.source_type!='chat'"
            else:
                source = """chat_attachments a JOIN files f ON f.id=a.file_id
                    JOIN chat_units u ON u.message_id=a.message_id AND u.attachment_id=a.id
                    JOIN chunks c ON c.id=u.chunk_id"""
                condition = """a.file_id=? AND c.source_type='chat'
                    AND (a.linked_hash IS NULL OR a.linked_hash=f.file_hash)"""
            scope = "AND EXISTS (SELECT 1 FROM search_scope WHERE search_scope.id=c.id)" if restricted else ""
            try:
                rows = db.conn.execute(f"""
                    WITH scored AS MATERIALIZED (
                        SELECT c.id AS chunk_id, {' + '.join(scores)} AS score
                        FROM {source} WHERE {condition} {scope}
                    )
                    SELECT chunk_id FROM scored WHERE score>0 ORDER BY score DESC,chunk_id LIMIT ?
                    """, (*terms, request.file_id, per_file)).fetchall()
            except sqlite3.OperationalError as exc:
                if report.time_limited and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                    break
                raise
            report.checked_files += 1
            report.chunk_targets.update((int(row["chunk_id"]), request.file_id) for row in rows)
    finally:
        db.conn.set_progress_handler(None, 0)
    return report
