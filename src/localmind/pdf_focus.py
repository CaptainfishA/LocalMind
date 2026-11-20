"""Read-only alignment of indexed body text to one rendered PDF page."""

import math

from .preview_focus import find_anchor

MAX_PAGE_CHARS = 100_000
MAX_REGIONS = 128


def page_focus(page, pixmap, scale, anchor):
    if not anchor:
        return dict(focus_status="", focus_polygons=[], note="")
    import pymupdf

    def unavailable(status, note):
        return dict(focus_status=status, focus_polygons=[], note=note)

    try:
        textpage = page.get_textpage(flags=pymupdf.TEXTFLAGS_TEXT)
        text = textpage.extractText()
        if not text.strip():
            return unavailable("no_text", "本页无可定位文字，仅显示原页")
        if len(text) > MAX_PAGE_CHARS:
            return unavailable("limited", "本页文字超过 100000 字符，已跳过定位")
        match = find_anchor(text, anchor)
        if not match:
            return unavailable("not_found", "本页未找到索引片段，显示原页")
        # Search the complete matched text, including its current line breaks.
        # Never highlight independent keywords from an otherwise missing passage.
        quads = page.search_for(text[slice(*match)], textpage=textpage, quads=True)
        polygons = []
        for quad in quads:
            rotated = quad * page.rotation_matrix
            points = [rotated.ul, rotated.ur, rotated.lr, rotated.ll]
            values = [[(p.x * scale - pixmap.x) / pixmap.width,
                       (p.y * scale - pixmap.y) / pixmap.height] for p in points]
            if not all(math.isfinite(value) for point in values for value in point):
                continue
            xs, ys = zip(*values)
            if max(xs) <= 0 or min(xs) >= 1 or max(ys) <= 0 or min(ys) >= 1:
                continue
            polygons.append(values)
            if len(polygons) == MAX_REGIONS:
                break
        if not polygons:
            return unavailable("not_found", "本页未找到可显示的索引片段，显示原页")
        note = "已标记本页索引片段"
        if len(quads) > MAX_REGIONS:
            note += f" · 仅显示前 {MAX_REGIONS} 个区域"
        return dict(focus_status="matched", focus_polygons=polygons, note=note)
    except Exception:
        # Optional alignment must not prevent an otherwise readable page preview.
        return unavailable("unavailable", "本页暂无法定位，显示原页")
