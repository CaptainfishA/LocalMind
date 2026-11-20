"""Bounded, query-time body embeddings, staged until the search is accepted."""

from collections import OrderedDict
from dataclasses import dataclass, field
import re
import time

import numpy as np

from .embeddings import normalize_vectors


@dataclass(frozen=True, slots=True)
class BodyFragment:
    text: str
    start: int
    end: int


@dataclass(slots=True)
class RefinementPlan:
    identity: tuple
    selected: int = 0
    encoded: int = 0
    cache_hits: int = 0
    complete: bool = True
    matches: dict[int, tuple[BodyFragment, float]] = field(default_factory=dict)
    pending: dict[str, np.ndarray] = field(default_factory=dict)
    used: tuple[str, ...] = ()


def body_fragments(text: str) -> tuple[BodyFragment, ...]:
    """Only literal substrings; offsets stay relative to the snapshot chunk."""
    if len(text) > 4096:
        return ()
    output, seen = [], {text.strip()}
    for line in re.finditer(r"[^\r\n]+", text):
        units = [(line.group(), line.start())]
        if len(line.group()) > 384:
            units = [(match.group(), line.start() + match.start())
                     for match in re.finditer(r"[^。！？!?；;]+[。！？!?；;]?", line.group())]
        for value, offset in units:
            start = offset + len(value) - len(value.lstrip())
            value = value.strip()
            if 24 <= len(value) <= 384 and value not in seen:
                output.append(BodyFragment(value, start, start + len(value)))
                seen.add(value)
                if len(output) == 16:
                    return tuple(output)
    return tuple(output)


class ParagraphRefiner:
    MAX_CACHE = 1024
    BUDGET_SECONDS = 0.200

    def __init__(self):
        self.cache = OrderedDict()
        self.identity = None

    def clear(self):
        self.cache.clear()
        self.identity = None

    def prepare(self, items, query_vector, *, identity, encoder, terms=(), device="cpu", check=lambda: None):
        started = time.perf_counter()
        check()
        if identity != self.identity:
            self.clear()
            self.identity = identity
        limit, batch_size = (24, 8) if device == "cuda" else (8, 4)
        groups, counts = OrderedDict(), {}
        for item in sorted(items, key=lambda value: (-value["preliminary"], value["chunk_id"])):
            check()
            row = item["row"]
            if row["source_type"] == "chat":
                continue
            fid = int(row["file_id"])
            if (fid not in counts and len(counts) >= 24) or counts.get(fid, 0) >= 2:
                continue
            counts[fid] = counts.get(fid, 0) + 1
            for fragment in body_fragments(row["chunk_text"]):
                groups.setdefault(fid, []).append((int(item["chunk_id"]), fragment))
        for group in groups.values():
            group.sort(key=lambda entry: (-sum(term.lower() in entry[1].text.lower() for term in terms if len(term) >= 2),
                                          len(entry[1].text), entry[0], entry[1].start))
        # Round-robin across files; cache warmth must not change which texts are selected.
        texts = OrderedDict()
        for depth in range(max(map(len, groups.values()), default=0)):
            for group in groups.values():
                if depth < len(group):
                    texts.setdefault(group[depth][1].text, None)
                if len(texts) == limit:
                    break
            if len(texts) == limit:
                break
        plan = RefinementPlan(identity, selected=len(texts), used=tuple(texts))
        if not texts:
            return plan
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim != 1 or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("Invalid refinement query vector")
        query = normalize_vectors(query)[0]
        missing = [text for text in texts if text not in self.cache]
        plan.cache_hits = len(texts) - len(missing)
        for start in range(0, len(missing), batch_size):
            check()
            if time.perf_counter() - started >= self.BUDGET_SECONDS:
                plan.complete = False
                return plan
            current = missing[start:start + batch_size]
            vectors = np.asarray(encoder(current), dtype="float32")
            check()
            if (vectors.shape != (len(current), len(query)) or not np.isfinite(vectors).all()
                    or np.any(np.linalg.norm(vectors, axis=1) == 0)):
                raise ValueError("Invalid refinement document vectors")
            for text, vector in zip(current, normalize_vectors(vectors)):
                value = vector.copy()
                value.flags.writeable = False
                plan.pending[text] = value
            plan.encoded += len(current)
        scores = {text: max(0.0, min(1.0, float(
            (plan.pending[text] if text in plan.pending else self.cache[text]) @ query))) for text in texts}
        for group in groups.values():
            check()
            for cid, fragment in group:
                if fragment.text in scores:
                    score = scores[fragment.text]
                    previous = plan.matches.get(cid)
                    if previous is None or score > previous[1]:
                        plan.matches[cid] = fragment, score
        return plan

    def commit(self, plan):
        if plan is None or plan.identity != self.identity:
            return
        self.cache.update(plan.pending)
        for text in plan.used:
            if text in self.cache:
                self.cache.move_to_end(text)
        while len(self.cache) > self.MAX_CACHE:
            self.cache.popitem(last=False)
