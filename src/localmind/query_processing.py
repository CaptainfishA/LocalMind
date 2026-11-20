from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from .identifiers import query_identifier


_INTENT_PHRASES = (
    "帮我查找",
    "帮我搜索",
    "帮我找到",
    "帮我找",
    "我想查找",
    "我想搜索",
    "我想找",
    "电脑里面",
    "电脑里",
    "文件里面",
    "文件中",
    "文件里",
    "包含",
    "有关",
    "关于",
    "相关的",
    "提到",
    "写了",
    "查找",
    "搜索",
    "找到",
    "那一份",
    "那份",
    "一份",
    "文件",
)

_STOP_WORDS = {
    "的",
    "了",
    "和",
    "或",
    "与",
    "中",
    "里",
    "有",
    "一个",
    "一些",
    "这个",
    "那个",
    "请",
    "一下",
    "修改过",
    "发来的",
}

_TIME_WORDS = {"上个月", "本月", "这个月", "去年", "今年"}

_SEARCH_REQUEST = re.compile(
    r"(?:(?:请|麻烦)(?:你)?\s*)?(?:(?:帮我|替我|给我|我想|想要)\s*)?"
    r"(?:查找|寻找|搜索|找|搜)(?:一下|一找|一搜)?(?:吧)?"
)
_CONTENT_PREFIX = re.compile(
    r"^(?:(?:文件|文档|资料)(?:的)?(?:内容|正文)|正文|内容)"
    r"(?:中|里)?\s*(?:是|为|讲的是|写的是|说明(?!书|文)|介绍(?!信)|描述(?!符)|提到|[:：])"
)
_CHAT_PREFIX = re.compile(
    r"^(?:聊天(?:记录)?|聊天消息|聊天线索|群聊(?:记录)?|群消息)"
    r"(?:中|里)?\s*(?:说|提到|写|约定|要求|确认|讨论|[:：])"
)

_EXTENSION_ALIASES = {
    "pdf": {"pdf"},
    "ppt": {"pptx"},
    "pptx": {"pptx"},
    "演示文稿": {"pptx"},
    "word": {"docx"},
    "docx": {"docx"},
    "文档": {"docx"},
    "excel": {"xlsx", "csv"},
    "xlsx": {"xlsx"},
    "表格": {"xlsx", "csv"},
    "csv": {"csv"},
    "截图": {"png", "jpg", "jpeg"},
    "图片": {"png", "jpg", "jpeg"},
    "照片": {"png", "jpg", "jpeg"},
}


@dataclass(slots=True)
class ProcessedQuery:
    original: str
    normalized: str
    semantic_text: str
    terms: list[str]
    desired_extensions: set[str]
    modified_after: str | None = None
    modified_before: str | None = None
    identifier: str = ""

    @property
    def descriptive(self) -> bool:
        """Use semantics more strongly for sentences, while preserving literal lookups."""
        literal = (
            re.search(r"[a-z0-9]+[-_][a-z0-9]*\d[a-z0-9_-]*", self.normalized, re.I)
            or re.search(r"[a-z]+\d{2,}", self.normalized, re.I)
            or re.search(r"\.(?:txt|md|csv|docx|xlsx|pptx|pdf|png|jpe?g)(?![a-z0-9])", self.normalized)
            or re.search(r"\d{6,}", self.normalized)
            or "\\" in self.original or "/" in self.original
        )
        return not literal and len(re.findall(r"[一-鿿]", self.original)) >= 10 and len(self.terms) >= 3


def process_query(text: str, *, now: datetime | None = None, infer_dates: bool = True) -> ProcessedQuery:
    original = unicodedata.normalize("NFKC", text).strip()
    normalized = _normalize_spaces(original.lower())
    desired_extensions = _infer_extensions(normalized)
    modified_after, modified_before = (
        _infer_date_range(original, now or datetime.now()) if infer_dates else (None, None))

    lexical_text = normalized
    for phrase in _INTENT_PHRASES:
        lexical_text = lexical_text.replace(phrase, " ")
    lexical_text = _normalize_spaces(lexical_text)
    identifier = query_identifier(original)
    token_text = re.sub(re.escape(identifier), " ", lexical_text, flags=re.I) if identifier else lexical_text
    terms = ([identifier] if identifier else []) + _tokenize(token_text)
    terms = [
        term
        for term in terms
        if term not in _STOP_WORDS and _is_useful_term(term)
        and not ((modified_after or modified_before) and term in _TIME_WORDS)
    ]

    compact = "".join(lexical_text.split())
    if not identifier and 2 <= len(compact) <= 18 and compact not in terms:
        terms.insert(0, compact)

    return ProcessedQuery(
        original=original,
        normalized=normalized,
        semantic_text=original,
        terms=_dedupe(terms)[:16],
        desired_extensions=desired_extensions,
        modified_after=modified_after,
        modified_before=modified_before,
        identifier=identifier,
    )


def semantic_clauses(query: ProcessedQuery) -> tuple[str, ...]:
    """Keep explicit, conjunctive two-part descriptions."""
    if not query.descriptive or len(query.original) > 384:
        return ()
    if re.search(r"不要|排除|不包含|不包括|不含|并非|而非|不是|除了|或|要么|\b(?:or|not|except)\b",
                 query.original, re.I):
        return ()
    spans = _description_spans(query.original)
    while spans and _SEARCH_REQUEST.fullmatch(query.original[slice(*spans[0])].strip()):
        spans.pop(0)
    parts = [query.original[start:end].strip() for start, end in spans]
    if len(parts) > 2:
        sources = semantic_clause_sources(tuple(parts))
        starts = [index for index, source in enumerate(sources) if source != "all"]
        # Only explicit source headings can own internal punctuation. Preserve every
        # character after the opening request; never drop an unlabeled requirement.
        if len(starts) != 2 or starts[0] != 0:
            return ()
        boundary = starts[1]
        parts = [query.original[spans[0][0]:spans[boundary - 1][1]].strip(),
                 query.original[spans[boundary][0]:spans[-1][1]].strip()]
    if len(parts) != 2 or any(not 6 <= len(part) <= 192 for part in parts):
        return ()
    if any(len(re.findall(r"[一-鿿]", part)) < 4 for part in parts):
        return ()
    if any(len(_dedupe([term for term in _tokenize(part)
                       if _is_useful_term(term) and term not in _STOP_WORDS])) < 2 for part in parts):
        return ()
    for source, body in (_source_description(part) for part in parts):
        if source != "all" and not any(_is_useful_term(term) and term not in _STOP_WORDS for term in _tokenize(body)):
            return ()
    return tuple(parts) if parts[0] != parts[1] else ()


def semantic_clause_sources(clauses: tuple[str, ...]) -> tuple[str, ...]:
    """Honor explicit source prefixes only; leave ordinary descriptions unrestricted."""
    return tuple(_source_description(clause)[0] for clause in clauses)


def semantic_clause_texts(clauses: tuple[str, ...]) -> tuple[str, ...]:
    """Encode the description, while retaining the original clause for routing and display."""
    return tuple(_source_description(clause)[1] for clause in clauses)


def single_source_clauses(query: ProcessedQuery, source: str) -> tuple[str, ...]:
    """Require explicit conjunction before decomposing a source-filtered query."""
    if source not in {"content", "chat"}:
        raise ValueError("Invalid clause source")
    clauses = semantic_clauses(query)
    sources = semantic_clause_sources(clauses)
    if not clauses or any(value not in {"all", source} for value in sources):
        return ()
    start = query.original.find(clauses[0]) + len(clauses[0])
    end = query.original.find(clauses[1], start)
    boundary = query.original[start:end]
    explicit = (all(value == source for value in sources) or ";" in boundary
                or re.match(r"^(?:而且|并且|同时|还要|还需|以及|另外)", clauses[1]))
    return clauses if explicit else ()


def _source_description(clause: str) -> tuple[str, str]:
    request = _SEARCH_REQUEST.match(clause)
    prefix = clause[request.end():].lstrip() if request else clause
    for source, pattern in (("content", _CONTENT_PREFIX), ("chat", _CHAT_PREFIX)):
        match = pattern.match(prefix)
        if match:
            return source, prefix[match.end():].strip()
    return "all", clause


def _description_spans(text: str) -> list[tuple[int, int]]:
    """Split only outside balanced quotes, retaining offsets into the original query."""
    quotes = {"“": "”", "‘": "’", "「": "」", "『": "』", "《": "》", '"': '"', "'": "'"}
    closing = set(quotes.values())
    stack, spans, start = [], [], 0
    for index, char in enumerate(text):
        if char == "'" and re.fullmatch(r"[A-Za-z0-9]'[A-Za-z0-9]", text[max(0, index - 1):index + 2]):
            continue
        if stack and char == stack[-1]:
            stack.pop()
        elif char in quotes:
            stack.append(quotes[char])
        elif char in closing:
            return []
        elif not stack and char in ",，;；。\n":
            if text[start:index].strip():
                spans.append((start, index))
            start = index + 1
    if stack:
        return []
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z一-鿿._+-]+", " ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    try:
        import jieba

        return [item.strip().lower() for item in jieba.cut_for_search(text) if item.strip()]
    except Exception:
        return _fallback_tokens(text)


def _fallback_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for part in re.findall(r"[a-zA-Z0-9._+-]+|[一-鿿]+", text.lower()):
        if re.fullmatch(r"[一-鿿]+", part) and len(part) > 3:
            tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
            tokens.extend(part[index : index + 3] for index in range(len(part) - 2))
        tokens.append(part)
    return tokens


def _is_useful_term(term: str) -> bool:
    if not term or term.isspace():
        return False
    if re.fullmatch(r"[一-鿿]", term):
        return False
    return len(term) >= 2


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _infer_extensions(text: str) -> set[str]:
    output: set[str] = set()
    compact = "".join(text.split())
    for alias, extensions in _EXTENSION_ALIASES.items():
        if alias in compact:
            output.update(extensions)
    return output


def _infer_date_range(text: str, now: datetime) -> tuple[str | None, str | None]:
    mentions = list(re.finditer(r"上个月|这个月|本月|去年|今年", text))
    if len(mentions) != 1:
        return None, None
    mention = mentions[0]
    # These need richer temporal/boolean parsing; do not turn them into a hard interval.
    if re.search(r"不|没|未|而非|排除|除了|除外|或|要么|无论|"
                 r"之前|以前|之后|以后|以来|至今|截至|截止", text):
        return None, None
    if "\\" in text or "/" in text:
        return None, None
    quoted = r'"[^"]*"|「[^」]*」|『[^』]*』|“[^”]*”|‘[^’]*’|《[^》]*》|' + r"'[^']*'"
    if any(part.start() <= mention.start() < part.end() for part in re.finditer(quoted, text)):
        return None, None
    before, after = text[:mention.start()], text[mention.end():]
    if re.search(r"包含|提到|写了|写着|正文|内容|文件里|文件中|标题|文件名|名为|叫做", before):
        return None, None
    if re.match(r"\s*(?:起|开始|到|至|之前|以前|之后|以后|以来)", after):
        return None, None
    action = re.match(r"\s*(?:(?:的|刚刚|刚|曾经|曾|已经|才)\s*)*"
                      r"(?:发给|发来|发送|发过|发的|收到|接收|转发|分享|提供|上传|下载|保存|创建|修改|更新)", after)
    leading = re.fullmatch(r"\s*(?:(?:请|帮我|替我|麻烦|我想|想要|想|找一下|找|查找|找到|"
                           r"搜索|搜一下|搜|看一下|看看|一下)\s*)*", before)
    # A leading shorthand remains supported, but dates inside descriptions/names are content.
    if not action and (not leading or re.search(r"\.[a-zA-Z0-9]{1,10}(?![a-zA-Z0-9])", text)):
        return None, None
    text = mention.group()
    first_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if "上个月" in text:
        if first_this_month.month == 1:
            first_last_month = first_this_month.replace(year=first_this_month.year - 1, month=12)
        else:
            first_last_month = first_this_month.replace(month=first_this_month.month - 1)
        return first_last_month.isoformat(), first_this_month.isoformat()
    first_this_year = first_this_month.replace(month=1)
    if "去年" in text:
        return first_this_year.replace(year=now.year - 1).isoformat(), first_this_year.isoformat()
    if "今年" in text:
        return first_this_year.isoformat(), first_this_year.replace(year=now.year + 1).isoformat()
    if "本月" in text or "这个月" in text:
        next_month = (first_this_month.replace(year=now.year + 1, month=1) if now.month == 12
                      else first_this_month.replace(month=now.month + 1))
        return first_this_month.isoformat(), next_month.isoformat()
    return None, None
