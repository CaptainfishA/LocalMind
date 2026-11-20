"""Bounded, read-only file previews. Run in a disposable process, never in Qt's UI thread."""

from __future__ import annotations

import base64
import csv
import codecs
from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import sys
import zipfile
from itertools import islice

from .file_freshness import _is_remote_drive
from .preview_focus import MAX_ANCHOR_CHARS, MAX_LOCATE_CHARS, table_window, text_window
from .pdf_focus import page_focus

MAX_BYTES = 50 * 1024 * 1024
MAX_TEXT = 24000
IMAGE_EDGE = 1600
SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif",
             ".txt", ".md", ".csv", ".docx", ".xlsx", ".pptx"}


class PreviewError(ValueError):
    pass


def local_path(value):
    if (not isinstance(value, str) or "\x00" in value or "://" in value
            or value.startswith(("\\\\", "//")) or not Path(value).is_absolute()
            or (os.name == "nt" and (not PureWindowsPath(value).is_absolute() or ":" in value[2:]))):
        raise PreviewError("仅预览已定位的本地文件")
    path = Path(value)
    if len(path.parts) > 64 or _is_remote_drive(path):
        raise PreviewError("不自动预览网络路径")
    # Inspect parents before descending, without following junctions into remote shares.
    for entry in (*reversed(path.parents), path):
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PreviewError("不自动预览链接或重解析路径，可使用系统程序打开")
    if not stat.S_ISREG(info.st_mode):
        raise PreviewError("该路径不是普通文件")
    if info.st_size > MAX_BYTES:
        raise PreviewError("文件超过 50 MB 预览上限，可使用系统程序打开")
    if path.suffix.lower() not in SUPPORTED:
        raise PreviewError("此格式暂不支持内嵌预览，可使用系统程序打开")
    return path, info


def check_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 5000 or sum(item.file_size for item in entries) > 80 * 1024 * 1024:
            raise PreviewError("文档解压内容过大，已停止预览")
        if any(item.file_size > 24 * 1024 * 1024 for item in entries):
            raise PreviewError("文档包含过大的内部资源，已停止预览")


def _text_pages(text, page, caption, note="", anchor=""):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, offset, match = text_window(text, anchor, limit=MAX_TEXT)
    count = max(1, (len(text) + 3499) // 3500)
    page = max(0, min(page, count - 1))
    highlights = []
    if match:
        start, end = max(0, match[0] - offset - page * 3500), min(3500, match[1] - offset - page * 3500)
        if start < end:
            highlights.append([start, end])
    if anchor:
        note += (f" · 已定位索引片段，节选起点约第 {offset + 1} 字符" if match else
                 " · 当前读取范围未找到索引片段，显示开头")
    return dict(kind="text", text=text[page * 3500:(page + 1) * 3500], page=page,
                pages=count, caption=caption, note=note, unit="段", focus_spans=highlights,
                focus_status="matched" if match else "not_found" if anchor else "")


def _read_text(path, limit, file_size):
    with path.open("rb") as stream:
        raw = stream.read(limit)
    encodings = ("utf-16",) if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            # A byte limit may split a valid multi-byte character; it is not an encoding failure.
            text = codecs.getincrementaldecoder(encoding)().decode(raw, final=file_size <= len(raw))
            return text, len(raw)
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace"), len(raw)


def _table_rows(iterator, *, focus=False):
    rows, length = [], 0
    for row in islice(iterator, 2000 if focus else 60):
        values = [str(value)[:1000 if focus else 200] if value is not None else ""
                  for value in islice(row, 64 if focus else 12)]
        length += sum(map(len, values)) + len(values) * 3 + 24
        if length > MAX_LOCATE_CHARS:
            break
        rows.append(values)
    return rows


def preview_file(value, page=0, sheet="", expected_hash="", anchor=""):
    if not isinstance(anchor, str) or len(anchor) > MAX_ANCHOR_CHARS:
        raise PreviewError("预览定位片段无效或过长")
    path, before = local_path(value)
    if expected_hash:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_hash:
            raise PreviewError("附件内容与已确认文件不一致，请重新定位")
    page = max(0, int(page))
    ext = path.suffix.lower()
    if ext in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}:
        import pymupdf

        if ext != ".pdf":
            # Decode only bounded raster input; never follow document links or play animation.
            from PIL import Image, ImageOps
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as source:
                    if source.width * source.height > 24_000_000:
                        raise PreviewError("图片像素过大，已停止预览")
                    source.seek(0)
                    source.thumbnail((IMAGE_EDGE, IMAGE_EDGE))
                    pixels = ImageOps.exif_transpose(source).convert("RGB")
                    stream = io.BytesIO()
                    pixels.save(stream, format="PNG")
                    payload = stream.getvalue()
            data = dict(kind="image", image=base64.b64encode(payload).decode("ascii"), page=0,
                        pages=1, caption="图片预览", note="" if ext != ".gif" else "显示首帧", unit="页")
        else:
            with pymupdf.open(path) as document:
                if document.needs_pass:
                    raise PreviewError("PDF 已加密，请使用系统程序打开")
                if not document.page_count:
                    raise PreviewError("PDF 没有可预览页面")
                page = min(page, document.page_count - 1)
                current = document.load_page(page)
                scale = min(2.0, IMAGE_EDGE / max(current.rect.width, current.rect.height, 1))
                pix = current.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                data = dict(kind="image", image=base64.b64encode(pix.tobytes("png")).decode("ascii"),
                            page=page, pages=document.page_count, caption="PDF 原页", unit="页",
                            **page_focus(current, pix, scale, anchor))
    elif ext in {".txt", ".md", ".csv"}:
        text, read_bytes = _read_text(path, (2 * 1024 * 1024 if anchor else 256 * 1024), before.st_size)
        if anchor:
            text = text[:MAX_LOCATE_CHARS]
        if ext == ".csv":
            rows = _table_rows(csv.reader(io.StringIO(text)), focus=bool(anchor))
            data = dict(kind="table", **table_window(rows, anchor), page=0, pages=1, caption="CSV 内容",
                        note="每次显示最多 60 行、12 列" + (" · 定位读取前 2 MB、2000 行、64 列" if anchor else ""), unit="页")
        else:
            note = ("定位读取前 2 MB、1000000 字符" if anchor else "文本分段预览"
                    + (" · 仅显示前 24000 字符" if len(text) > MAX_TEXT or before.st_size > read_bytes else ""))
            data = _text_pages(text, page, "文本内容", note, anchor)
    else:
        check_archive(path)
        if ext == ".docx":
            from docx import Document
            from docx.table import Table
            document = Document(path)
            def parts():
                for block in document.iter_inner_content():
                    if isinstance(block, Table):
                        for row in islice(block.rows, 2000 if anchor else 60):
                            values = (cell.text.strip()[:1000 if anchor else 300]
                                      for cell in islice(row.cells, 64 if anchor else 12))
                            yield " | ".join(value for value in values if value)
                    else:
                        yield block.text
            lines, length = [], 0
            budget = MAX_LOCATE_CHARS if anchor else MAX_TEXT
            for value in parts():
                lines.append(value[:max(0, budget - length)])
                length += len(value) + 2
                if length >= budget:
                    break
            data = _text_pages("\n\n".join(lines), page, "Word 内容",
                               "文字与表格节选，非原始排版" + (" · 定位读取前 1000000 字符" if anchor else
                               " · 已达 24000 字符上限" if length >= MAX_TEXT else ""), anchor)
        elif ext == ".pptx":
            from pptx import Presentation
            document = Presentation(path)
            if not len(document.slides):
                raise PreviewError("演示文稿没有幻灯片")
            page = min(page, len(document.slides) - 1)
            lines = []
            for shape in document.slides[page].shapes:
                if shape.has_text_frame:
                    lines.append(shape.text[:MAX_TEXT])
                elif shape.has_table:
                    lines.extend("  |  ".join(cell.text[:300] for cell in row.cells) for row in shape.table.rows)
                if sum(map(len, lines)) >= MAX_TEXT:
                    break
            data = dict(kind="text", text="\n\n".join(lines)[:MAX_TEXT], page=page,
                        pages=len(document.slides), caption="幻灯片文字", note="非原始排版；图片需打开文件查看", unit="页")
        else:
            from openpyxl import load_workbook
            with path.open("rb") as stream, closing(load_workbook(
                    stream, read_only=True, data_only=True, keep_links=False)) as document:
                names = document.sheetnames
                page = names.index(sheet) if sheet in names else min(page, len(names) - 1)
                current = document[names[page]]
                row_limit, col_limit = (2000, 64) if anchor else (60, 12)
                with closing(current.iter_rows(min_row=1, max_row=min(current.max_row or row_limit, row_limit),
                                               max_col=min(current.max_column or col_limit, col_limit), values_only=True)) as iterator:
                    rows = _table_rows(iterator, focus=bool(anchor))
                data = dict(kind="table", **table_window(rows, anchor, numbered=True), page=page,
                            pages=len(names), caption="工作表 · " + names[page],
                            note="每次显示最多 60 行、12 列 · 公式仅显示已保存的值"
                            + (" · 定位读取前 2000 行、64 列" if anchor else ""), unit="表")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PreviewError("文件在预览期间发生变化，请刷新预览")
    return {**data, "path": str(path), "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def main():
    try:
        request = json.loads(sys.stdin.read(16384))
        data = preview_file(request["path"], request.get("page", 0), request.get("sheet", ""),
                            request.get("expected_hash", ""), request.get("anchor", ""))
    except FileNotFoundError:
        data = {"error": "文件已移动或删除，请重新定位"}
    except PermissionError:
        data = {"error": "文件暂不可读，请检查权限或占用情况"}
    except PreviewError as error:
        data = {"error": str(error)}
    except Exception:
        data = {"error": "文件损坏或格式暂不可预览，可使用系统程序打开"}
    print(json.dumps(data, ensure_ascii=True))


if __name__ == "__main__":
    main()
