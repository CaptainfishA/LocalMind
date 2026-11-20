"""Bounded excerpt alignment against current file content, not a new search index."""

import re

MAX_LOCATE_CHARS = 1_000_000
MAX_ANCHOR_CHARS = 512


def find_anchor(text: str, anchor: str):
    needle = re.sub(r"\s+", "", anchor)
    if not needle:
        return None
    compact = re.sub(r"\s+", "", text)
    start = compact.find(needle)
    if start < 0:
        return None
    end, cursor, first = start + len(needle), 0, None
    for index, char in enumerate(text):
        if char.isspace():
            continue
        if cursor == start:
            first = index
        cursor += 1
        if cursor == end:
            return first, index + 1
    return None


def text_window(text, anchor, *, limit=24000):
    match = find_anchor(text, anchor) if anchor else None
    start = max(0, match[0] - 800) if match else 0
    return text[start:start + limit], start, match


def table_window(rows, anchor="", *, numbered=False):
    """Keep real row/column coordinates, including empty cells, during alignment."""
    lines, cells, cursor = [], [], 0
    for row_number, values in enumerate(rows, 1):
        content = [(column, value.strip()) for column, value in enumerate(values) if value.strip()]
        prefix = f"R{row_number}: " if numbered and content else ""
        line = prefix + " | ".join(value for _, value in content)
        position = cursor + len(prefix)
        for column, value in content:
            cells.append((row_number - 1, column, position, position + len(value)))
            position += len(value) + 3
        lines.append(line)
        cursor += len(line) + 1
    text = "\n".join(lines)
    match = find_anchor(text, anchor) if anchor else None
    matched = [cell for cell in cells if match and cell[2] < match[1] and cell[3] > match[0]]
    row_start = max(0, matched[0][0] - 2) if matched else 0
    column_start = max(0, matched[0][1] - 1) if matched else 0
    visible = []
    highlights = []
    for i, row in enumerate(rows[row_start:row_start + 60], row_start):
        values = []
        for j, value in enumerate(row[column_start:column_start + 12], column_start):
            cell = next((entry for entry in matched if entry[:2] == (i, j)), None)
            if cell:
                offset = max(0, match[0] - cell[2] - 40)
                value = value.strip()
                values.append(("…" if offset else "") + value[offset:offset + 300])
                highlights.append([i - row_start, j - column_start])
            else:
                values.append(value[:200])
        visible.append(values)
    return dict(rows=visible, row_start=row_start + 1, column_start=column_start + 1,
                focus_cells=highlights, focus_status="matched" if matched else "not_found" if anchor else "")
