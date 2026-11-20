"""Shared, parameterized filters applied before bounded keyword/vector recall."""
from __future__ import annotations

from datetime import datetime


def epoch(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def interval_predicates(column, query):
    clauses, params = [], []
    for value, operator in ((query.modified_after, ">="), (query.modified_before, "<")):
        if value:
            moment = epoch(value)
            if moment is None:
                raise ValueError("日期筛选无效，请使用 ISO 日期时间")
            clauses.append(f"localmind_epoch({column}) {operator} ?")
            params.append(moment)
    return clauses, params


def chat_predicates(query, *, include_dates=True):
    clauses, params = [], []
    for value, column in ((query.chat_platform, "s.platform"),
                          (query.conversation, "s.conversation"), (query.sender, "m.sender")):
        if value:
            clauses.append(f"instr(lower({column}),lower(?))>0")
            params.append(value)
    if include_dates:
        dates, moments = interval_predicates("m.sent_at", query)
        clauses.extend(dates)
        params.extend(moments)
    return clauses, params


def file_predicates(query, *, ext="f.ext", path="f.path"):
    clauses, params = [], []
    if query.file_type:
        clauses.append(f"lower({ext})=?")
        params.append(query.file_type.lower().lstrip("."))
    if query.path_contains:
        clauses.append(f"instr(lower({path}),lower(?))>0")
        params.append(query.path_contains)
    return clauses, params


def where(clauses):
    return " AND " + " AND ".join(clauses) if clauses else ""
