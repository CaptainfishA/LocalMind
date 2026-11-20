from __future__ import annotations

import math


def ranking_metrics(ranked_ids: list[str], relevance: dict[str, int]) -> dict[str, float]:
    """Positive grades are relevant; unjudged results count as zero for ranking."""
    ranked = list(dict.fromkeys(ranked_ids))
    positives = {key for key, grade in relevance.items() if grade > 0}
    if not positives:
        raise ValueError("Ranking metrics require at least one judged relevant document")
    hits5 = len(set(ranked[:5]) & positives)
    reciprocal = next((1.0 / rank for rank, key in enumerate(ranked[:10], 1) if key in positives), 0.0)

    def dcg(grades: list[int]) -> float:
        return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))

    ideal = dcg(sorted(relevance.values(), reverse=True)[:10])
    return {
        "hit_at_1": float(bool(ranked) and ranked[0] in positives),
        "hit_at_5": float(hits5 > 0),
        "recall_at_5": hits5 / len(positives),
        "mrr_at_10": reciprocal,
        "ndcg_at_10": dcg([relevance.get(key, 0) for key in ranked[:10]]) / ideal,
        "judged_at_10": sum(key in relevance for key in ranked[:10]) / max(1, len(ranked[:10])),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}


def no_answer_metrics(ranked_ids: list[str]) -> dict[str, float]:
    """Candidate return behavior on explicitly judged no-answer queries, not confidence."""
    count = len(set(ranked_ids))
    return {
        "candidate_return_rate": float(count > 0),
        "empty_result_rate": float(count == 0),
        "mean_candidate_count": float(count),
    }
