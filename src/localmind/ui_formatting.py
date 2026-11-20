"""Plain-data presentation shared by local result views."""
from datetime import datetime
from html import escape


def display_time(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M")
    except (AttributeError, ValueError, OverflowError, OSError):
        return value or "未知"


def source_label(value):
    value = str(value or "")
    return {
        "chat": "聊天线索", "ocr_image": "图片 OCR", "ocr_pdf": "扫描 PDF",
        "txt": "文本正文", "md": "文本正文", "text": "文本正文",
        "csv": "表格内容", "xlsx": "表格内容", "docx": "文档正文",
        "pptx": "幻灯片", "pdf": "PDF 正文",
    }.get(value, "OCR 文本" if value.startswith("ocr") else "其他来源")


def plain_tooltip(value):
    # Qt auto-detects rich text in tooltips. Only our wrapper may contain markup.
    return "<qt>" + escape(str(value)).replace("\n", "<br>") + "</qt>"


def result_source_label(result):
    sources = {part.source_type for part in result.supporting_evidence} or {result.source_type}
    if len(sources) == 1:
        return source_label(next(iter(sources)))
    return "正文与聊天" if "chat" in sources else "文件正文"
