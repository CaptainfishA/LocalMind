"""Readable excerpts; never synthesize claims or reinterpret document markup."""

import re
from itertools import groupby

from PySide6.QtGui import QColor, QFont, QTextBlockFormat, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

from ..ui_formatting import display_time
from ..chunker import CHUNK_MAX_CHARS


class ReadableText(QPlainTextEdit):
    def __init__(self, placeholder=""):
        super().__init__()
        self.setObjectName("readableText")
        self.setReadOnly(True)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(40)

    def setPlainText(self, text):  # noqa: N802
        super().setPlainText(text)
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.SelectionType.Document)
        block = QTextBlockFormat()
        block.setLineHeight(145, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        block.setBottomMargin(8)
        cursor.mergeBlockFormat(block)
        item = self.document().firstBlock()
        while item.isValid():
            if item.text().startswith(("文件正文", "聊天线索")):
                heading = QTextCursor(item)
                heading.select(QTextCursor.SelectionType.BlockUnderCursor)
                format = QTextCharFormat()
                format.setFontWeight(QFont.Weight.DemiBold)
                format.setForeground(QColor("#237b63"))
                heading.mergeCharFormat(format)
            item = item.next()


def readable(text):
    lines = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if len(line) > 110:
            paragraphs, current = [], ""
            for sentence in re.split(r"(?<=[。！？])\s*", line):
                current += sentence
                if len(current) >= 70:
                    paragraphs.append(current)
                    current = ""
            if current:
                paragraphs.append(current)
            line = "\n".join(paragraphs)
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def excerpt(text, limit, terms=()):
    if len(text) <= limit:
        return text, False
    positions = [text.casefold().find(term.casefold()) for term in terms if term]
    positions = [p for p in positions if p >= 0]
    start = max(0, min(positions) - 100) if positions else 0
    return ("…" if start else "") + text[start:start + limit] + "…", True


def fold_repeated_lines(text):
    """Collapse only exact adjacent repetition; the expanded view keeps the source."""
    lines, folded = [], False
    for line, group in groupby(text.splitlines()):
        count = sum(1 for _ in group)
        if count >= 3 and len(line.strip()) >= 16:
            lines.extend((line, f"…（相同内容共 {count} 行）"))
            folded = True
        else:
            lines.extend([line] * count)
    return "\n".join(lines), folded


def chat_excerpt(text, name=""):
    if not text.startswith("平台: "):
        return text
    lines = text.splitlines()
    if len(lines) == 1:
        # Search snippets can have their newlines normalized before reaching the UI.
        _, separator, body = text.partition("消息: ")
        if not separator:
            return text
        prefix = "[文件] " + name
        if name and body.startswith(prefix):
            body = body[len(prefix):].lstrip()
        return body.replace("上下文（引用 / 相邻消息，非附件内容）:", "相邻 / 引用消息（不属于文件正文）\n", 1)
    while lines and lines[0].startswith(("平台: ", "会话: ", "发送者: ", "消息时间: ", "附件: ")):
        lines.pop(0)
    if lines and lines[0] == "消息: [文件] " + name:
        lines.pop(0)
    elif lines and lines[0].startswith("消息: "):
        lines[0] = lines[0][len("消息: "):]
    if lines and lines[0].startswith("上下文（引用 / 相邻消息，非附件内容）:"):
        lines[0] = "相邻 / 引用消息（不属于文件正文）"
    return "\n".join(lines)


def evidence_sources(result):
    sources, seen = [], set()
    for ref in [result.chat_source, *(part.chat_source for part in result.supporting_evidence
                                     if part.source_type == "chat")]:
        if ref and ref.identity not in seen:
            sources.append(ref)
            seen.add(ref.identity)
    return sources


def source_title(ref):
    return " · ".join(" ".join(str(value).split()) for value in (
        ref.conversation or "未命名会话", ref.sender or "未识别发送者", display_time(ref.sent_at),
    ))


def evidence_source_label(ref, expanded):
    if ref is None:
        return ""
    conversation = " ".join(ref.conversation.split()) or "未命名会话"
    if not expanded and len(conversation) > 32:
        conversation = conversation[:32] + "…"
    label = " · " + conversation
    if expanded:
        label += "\n关联消息：" + source_title(ref)
        if ref.origin:
            label += "\n" + ref.origin
    return label


def evidence_text(result, expanded=False):
    sections, shortened = [], False
    evidence = result.supporting_evidence
    if not evidence:
        raw = chat_excerpt(result.snippet, result.name) if result.source_type == "chat" else result.snippet
        if not expanded:
            raw, shortened = fold_repeated_lines(raw)
        text, clipped = excerpt(raw, len(raw) if expanded else 480)
        shortened |= clipped
        location = (f" · 第 {result.page_no} 页" if result.page_no else "")
        location += f" · {result.sheet_name}" if result.sheet_name else ""
        heading = "聊天线索" if result.source_type == "chat" else "文件正文"
        if result.source_type == "chat":
            heading += evidence_source_label(result.chat_source, expanded)
            shortened |= bool(result.chat_source) and not expanded
        if result.source_type.startswith("ocr"):
            heading += " · OCR"
        sections.append(heading + location + "\n" + readable(text))
    else:
        seen = set()
        refined_ids = {part.chunk_id for part in evidence if part.refined}
        for part in sorted(evidence, key=lambda part: part.source_type == "chat"):
            if not expanded and not part.refined and part.chunk_id in refined_ids:
                shortened = True
                continue
            raw = chat_excerpt(part.snippet, result.name) if part.source_type == "chat" else part.snippet
            key = (part.source_type, raw, part.query_clause,
                   part.chat_source.identity if part.chat_source else None)
            if key in seen:
                continue
            seen.add(key)
            heading = "聊天线索" if part.source_type == "chat" else "文件正文"
            if part.refined:
                heading += " · 匹配片段"
            elif part.chunk_id in refined_ids:
                heading += " · 上下文"
            if part.source_type == "chat":
                heading += evidence_source_label(part.chat_source, expanded)
                shortened |= bool(part.chat_source) and not expanded
            if part.source_type.startswith("ocr"):
                heading += " · OCR"
            heading += f" · 第 {part.page_no} 页" if part.page_no else ""
            heading += f" · {part.sheet_name}" if part.sheet_name else ""
            if not expanded:
                raw, folded = fold_repeated_lines(raw)
                shortened |= folded
            text, clipped = excerpt(raw, len(raw) if expanded else CHUNK_MAX_CHARS, part.matched_terms)
            shortened |= clipped
            description = "\n对应描述：" + part.query_clause if expanded and part.query_clause else ""
            shortened |= bool(part.query_clause) and not expanded
            sections.append(heading + description + ("\n\n" if expanded else "\n") + readable(text))
            if not expanded and len(sections) == 3 and len(evidence) > 3:
                shortened = True
                break
    return "\n\n".join(sections), shortened


def context_text(context, attachment=None, expanded=False, *, include_header=True):
    if not context:
        return "", False
    if expanded:
        return readable(context), False
    header, separator, body = context.partition("\n\n")
    if not separator:
        return readable(context), False
    if attachment:
        lines = [attachment["conversation"],
                 f"{attachment.get('sender') or '未识别发送者'}  ·  {display_time(attachment.get('sent_at'))}"]
        if attachment.get("origin"):
            lines.append(attachment["origin"])
    else:
        lines = [line for line in header.splitlines()
                 if not line.startswith(("采集时间:", "消息 ID:", "关联:"))]
    text, shortened = excerpt(body, 650)
    if not include_header:
        return readable(text), True
    return "\n".join(lines) + "\n\n" + readable(text), shortened or header != "\n".join(lines)
