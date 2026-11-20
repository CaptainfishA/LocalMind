from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def query_answerability(query: dict) -> str:
    value = query.get("answerability", "answerable")
    if value not in ("answerable", "no_answer"):
        raise ValueError("answerability must be answerable or no_answer")
    return value


def validate_no_answer(qid: str, judgments: dict[str, int], doc_ids: set[str]) -> None:
    # Missing positives are not evidence that a query has no answer.
    if not doc_ids or set(judgments) != doc_ids or any(grade != 0 for grade in judgments.values()):
        raise ValueError(f"No-answer query {qid} requires explicit zero judgments for the entire corpus")


def new_directory(path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise ValueError(f"Output must be a new directory: {target}")
    target.mkdir(parents=True)
    return target


def finish_dataset(root: Path, corpus: list[dict], queries: list[dict], qrels: list[dict], metadata: dict) -> None:
    for doc in corpus:
        doc["sha256"] = file_hash(root / doc["path"])
    write_jsonl(root / "corpus.jsonl", corpus)
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(root / "qrels.jsonl", qrels)
    write_json(root / "manifest.json", {
        "schema_version": 1, **metadata,
        "documents": len(corpus), "queries": len(queries),
        "checksums": {name: file_hash(root / name) for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")},
    })


def load_dataset(root: str | Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    base = Path(root).resolve()
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")
    for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl"):
        if file_hash(base / name) != manifest["checksums"][name]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    corpus, queries, qrels = [read_jsonl(base / name) for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")]
    doc_ids, query_ids, paths = set(), set(), set()
    no_answer_ids = set()
    for doc in corpus:
        path = (base / doc["path"]).resolve()
        if not path.is_relative_to(base / "files") or path in paths or doc["file_id"] in doc_ids:
            raise ValueError("Duplicate document ID/path or file outside dataset/files")
        if file_hash(path) != doc["sha256"]:
            raise ValueError(f"Document changed: {doc['path']}")
        doc_ids.add(doc["file_id"])
        paths.add(path)
    actual_paths = {p.resolve() for p in (base / "files").rglob("*") if p.is_file()}
    if paths != actual_paths:
        raise ValueError("Unlisted or missing files in dataset/files")
    for query in queries:
        if query["query_id"] in query_ids or not query["text"].strip() or query["split"] not in {"dev", "test"}:
            raise ValueError("Invalid or duplicate query")
        query_ids.add(query["query_id"])
        if query_answerability(query) == "no_answer":
            no_answer_ids.add(query["query_id"])
        filters = query.get("filters", {})
        allowed = {"file_type", "path_contains", "modified_after", "modified_before"}
        if not isinstance(filters, dict) or set(filters) - allowed or any(not isinstance(v, str) for v in filters.values()):
            raise ValueError("Invalid query filters")
    pairs, positive_queries, judgments = set(), set(), {}
    for rel in qrels:
        pair = rel["query_id"], rel["file_id"]
        if (pair in pairs or pair[0] not in query_ids or pair[1] not in doc_ids
                or type(rel["relevance"]) is not int or not 0 <= rel["relevance"] <= 3):
            raise ValueError("Invalid qrel or missing reference")
        pairs.add(pair)
        judgments.setdefault(pair[0], {})[pair[1]] = rel["relevance"]
        if rel["relevance"] > 0:
            positive_queries.add(pair[0])
    for qid in no_answer_ids:
        validate_no_answer(qid, judgments.get(qid, {}), doc_ids)
    if not query_ids or positive_queries != query_ids - no_answer_ids:
        raise ValueError("Each answerable query must have at least one judged positive")
    if manifest.get("documents") != len(corpus) or manifest.get("queries") != len(queries):
        raise ValueError("Dataset counts do not match manifest")
    return manifest, corpus, queries, qrels


def build_text_subset(
    documents: list[dict], queries: list[dict], qrels: list[dict], output: str | Path,
    *, seed: int = 20261013, query_count: int = 100, corpus_size: int = 5000,
    hard_per_query: int = 10, source: dict | None = None,
) -> Path:
    """Input: id/text documents and queries; query_id/file_id/relevance judgments."""
    if query_count < 2 or corpus_size < 0 or hard_per_query < 0:
        raise ValueError("Need >=2 queries, nonnegative corpus size and hard candidate count")
    docs = {str(d["id"]): str(d["text"]) for d in documents}
    query_map = {str(q["id"]): str(q["text"]) for q in queries}
    no_answer_ids = {str(q["id"]) for q in queries if query_answerability(q) == "no_answer"}
    if len(docs) != len(documents) or len(query_map) != len(queries):
        raise ValueError("Duplicate source IDs")
    rel_map: dict[str, dict[str, int]] = {}
    for rel in qrels:
        qid, did = str(rel["query_id"]), str(rel["file_id"])
        grade = rel["relevance"]
        if qid not in query_map or did not in docs or type(grade) is not int or not 0 <= grade <= 3:
            raise ValueError("Invalid source qrel")
        rel_map.setdefault(qid, {})[did] = max(grade, rel_map.get(qid, {}).get(did, 0))
    for qid in no_answer_ids:
        validate_no_answer(qid, rel_map.get(qid, {}), set(docs))
    eligible = sorted({q for q, rel in rel_map.items() if any(v > 0 for v in rel.values())} | no_answer_ids)
    if query_count > len(eligible):
        raise ValueError(f"Only {len(eligible)} labeled queries available")
    rng = random.Random(seed)
    selected = rng.sample(eligible, query_count)
    positives = {d for q in selected for d, grade in rel_map[q].items() if grade > 0}
    if corpus_size and corpus_size < len(positives):
        raise ValueError(f"Corpus limit would drop positives; require at least {len(positives)} documents")
    limit = min(corpus_size or len(docs), len(docs))
    retained = set(positives)
    mined = 0
    if hard_per_query and len(retained) < limit:
        # FTS mining does not relabel unjudged candidates as known negatives.
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE VIRTUAL TABLE candidates USING fts5(doc_id UNINDEXED, text, tokenize='trigram')")
            connection.executemany("INSERT INTO candidates VALUES (?, ?)", sorted(docs.items()))
            for qid in selected:
                compact = re.sub(r"\s+", "", query_map[qid])
                terms = list(dict.fromkeys(compact[i:i+3] for i in range(max(0, len(compact)-2))))[:24]
                expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
                if not expression:
                    continue
                rows = connection.execute(
                    "SELECT doc_id FROM candidates WHERE candidates MATCH ? ORDER BY bm25(candidates), doc_id LIMIT ?",
                    (expression, hard_per_query + len(rel_map[qid])),
                )
                added = 0
                for (did,) in rows:
                    if did not in retained and len(retained) < limit and added < hard_per_query:
                        retained.add(did)
                        added += 1
                        mined += 1
        finally:
            connection.close()
    remaining = sorted(set(docs) - retained)
    retained.update(rng.sample(remaining, limit - len(retained)))

    # Connected positive documents and duplicate query texts stay in one split.
    parents = {q: q for q in selected}

    def find(q: str) -> str:
        while parents[q] != q:
            parents[q] = parents[parents[q]]
            q = parents[q]
        return q

    owners: dict[tuple[str, str], str] = {}
    for q in selected:
        keys = [("query", re.sub(r"\s+", "", query_map[q]).casefold())]
        keys += [("positive", d) for d, grade in rel_map[q].items() if grade > 0]
        for key in keys:
            if key in owners:
                parents[find(q)] = find(owners[key])
            owners[key] = q
    groups: dict[str, list[str]] = {}
    for q in selected:
        groups.setdefault(find(q), []).append(q)
    if len(groups) < 2:
        raise ValueError("All queries share positives; cannot create independent dev/test groups")
    dev_ids: set[str] = set()
    for group in list(groups.values())[:-1]:
        if len(dev_ids) < max(1, query_count // 3):
            dev_ids.update(group)
    root = new_directory(output)
    (root / "files").mkdir()
    corpus = []
    for did in sorted(retained):
        name = hashlib.sha256(did.encode()).hexdigest() + ".txt"
        path = Path("files") / name
        (root / path).write_text(docs[did], encoding="utf-8")
        corpus.append({"file_id": did, "path": path.as_posix()})
    prepared_queries = [{"query_id": q, "text": query_map[q], "split": "dev" if q in dev_ids else "test",
                         "intent": ["public_retrieval"],
                         **({"answerability": "no_answer"} if q in no_answer_ids else {})} for q in selected]
    prepared_rels = [{"query_id": q, "file_id": d, "relevance": grade} for q in selected
                     for d, grade in sorted(rel_map[q].items()) if d in retained]
    finish_dataset(root, corpus, prepared_queries, prepared_rels, {
        "name": "public-text-subset", "kind": "public_subset", "seed": seed,
        "source": source or {}, "hard_candidate_method": "FTS5 character trigram BM25",
        "mined_documents": mined, "source_documents": len(docs),
        "label_policy": "Preserve source judgments. Unjudged is not confirmed irrelevant.",
        "comparability": "Sampled corpus; not directly comparable to the official full-corpus leaderboard.",
    })
    return root
