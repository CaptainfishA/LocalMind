"""Same-target semantic coverage for explicit two-part descriptions."""
from .chats import CHAT_PREFIX
from .query_processing import semantic_clause_sources


def file_clause_evidence(items, scores, clauses, *, whole_scores=None, target_keys=None):
    sources = semantic_clause_sources(clauses)
    groups = {}
    for item in items:
        row, cid = item["row"], int(item["chunk_id"])
        if cid not in scores:
            continue
        if row["source_type"] == "chat" and not row["path"].startswith(CHAT_PREFIX):
            refs = row.get("chat_refs", [])
            if not refs or refs[0].get("target_id") != row["file_id"]:
                continue
            ref = refs[0]
            if ref.get("linked_hash") and ref["linked_hash"] != ref.get("target_hash"):
                continue
        target = int(row["file_id"])
        if target_keys is not None:
            expected = f"file:{target}"
            if row["source_type"] == "chat" and row["path"].startswith(CHAT_PREFIX):
                refs = row.get("chat_refs", [])
                ref = refs[0] if len(refs) == 1 else {}
                mid, aid = ref.get("message_id"), ref.get("attachment_id")
                if (ref.get("chunk_id") != cid or ref.get("owner_file_id") != target
                        or type(mid) is not int or mid <= 0
                        or (aid is not None and (type(aid) is not int or aid <= 0))):
                    continue
                expected = f"chat:{mid}:{aid or 0}"
            target = target_keys.get(cid)
            if target != expected:
                continue
        groups.setdefault(target, []).append(item)
    output = {}
    for fid, group in groups.items():
        pools = [[item for item in group if source == "all"
                  or (item["row"]["source_type"] == "chat") == (source == "chat")]
                 for source in sources]
        if any(not pool for pool in pools):
            output[fid] = 0.0, ()
            continue
        winners = [max(pool, key=lambda item: (scores[item["chunk_id"]][i], -int(item["chunk_id"])))
                   for i, pool in enumerate(pools)]
        a, b = [max(0.0, min(1.0, scores[item["chunk_id"]][i])) for i, item in enumerate(winners)]
        coverage = 2 * a * b / (a + b) if a + b else 0.0
        parts = {}
        if a > 0 and b > 0:
            for item, clause in zip(winners, clauses):
                cid = int(item["chunk_id"])
                if cid in parts:
                    parts[cid]["query_clause"] += "\n" + clause
                else:
                    parts[cid] = {"chunk_id": cid, "row": item["row"], "query_clause": clause}
        if whole_scores is not None and parts:
            # A source-filtered query needs both condition coverage and whole-query
            # coherence. Keep the actual whole-query contributor in its explanation.
            whole = max(group, key=lambda item: (whole_scores.get(item["chunk_id"], 0.0), -int(item["chunk_id"])))
            cid = int(whole["chunk_id"])
            value = max(0.0, min(1.0, whole_scores.get(cid, 0.0)))
            coverage = (coverage + value) / 2
            if value > 0:
                parts.setdefault(cid, {"chunk_id": cid, "row": whole["row"]})
        output[fid] = coverage, tuple(parts.values())
    return output


def coverage_ranks(coverage):
    """Rank the semantic unit actually scored: one target, with ties sharing rank."""
    ranks = {}
    previous, rank = None, 0
    for position, (fid, (score, _parts)) in enumerate(
            sorted(coverage.items(), key=lambda entry: (-entry[1][0], entry[0])), 1):
        if score <= 0:
            continue
        if score != previous:
            rank, previous = position, score
        ranks[fid] = rank
    return ranks


def evidence_parts(item):
    """Include both lexical and semantic contributors without repeating a chunk."""
    parts = {}
    for part in (*item.get("joint_evidence", ()), *item.get("semantic_evidence", ())):
        cid = part["chunk_id"]
        if cid not in parts:
            parts[cid] = dict(part)
        elif part.get("query_clause"):
            parts[cid]["query_clause"] = part["query_clause"]
    detail = item.get("detail_evidence")
    if detail:
        parts.setdefault(item["chunk_id"], {"chunk_id": item["chunk_id"], "row": item["row"]})
        return (detail, *parts.values())
    return tuple(parts.values())


def evidence_key(item):
    key = tuple(sorted(part["chunk_id"] for part in evidence_parts(item)))
    detail = item.get("detail_evidence")
    return (*key, ("detail", detail["row"]["chunk_text"])) if detail else key
