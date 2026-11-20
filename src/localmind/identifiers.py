"""Conservative single-code lookup; never a general Boolean query parser."""

import re
from functools import lru_cache


_LEFT = r"(?<![A-Za-z0-9])(?<![A-Za-z0-9][_-])"
_RIGHT = r"(?![A-Za-z0-9]|[-_][A-Za-z0-9]|\.\d)"
_TOKEN = re.compile(_LEFT + r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*" + _RIGHT)
_LOGIC = re.compile(r"不|没|排除|除外|除了|或|要么|\b(?:not|except|or|without)\b", re.I)


def query_identifier(text: str) -> str:
    if _LOGIC.search(text) or re.search(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}", text):
        return ""
    codes = []
    for match in _TOKEN.finditer(text):
        token = match.group()
        if (4 <= len(token) <= 64 and re.search(r"[A-Za-z]", token)
                and len(re.findall(r"\d", token)) >= 2
                and not re.match(r"\.\d", text[match.end():])):
            codes.append(token.lower())
    unique = list(dict.fromkeys(codes))
    return unique[0] if len(unique) == 1 else ""


@lru_cache(maxsize=128)
def identifier_pattern(code: str):
    return re.compile(_LEFT + re.escape(code) + _RIGHT, re.I)


def identifier_position(text: str, code: str) -> int:
    match = identifier_pattern(code).search(text) if code else None
    return match.start() if match else -1


def identifier_content(text: str, source_type: str) -> str:
    if source_type != "chat" or not text.startswith("平台: "):
        return text
    lines = text.splitlines(keepends=True)
    while lines and lines[0].startswith(("平台: ", "会话: ", "发送者: ", "消息时间: ")):
        lines.pop(0)
    return "".join(lines)


def indexed_identifier(text: str, source_type: str, code: str) -> bool:
    return identifier_position(identifier_content(text, source_type), code) >= 0


def row_has_identifier(row, code: str) -> bool:
    return bool(code) and (indexed_identifier(str(row["chunk_text"]), row["source_type"], code)
                           or (row["source_type"] != "chat" and identifier_position(str(row["name"]), code) >= 0))
