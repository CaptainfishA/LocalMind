from __future__ import annotations

import json
from pathlib import Path


def compare_reports(before: str | Path, after: str | Path) -> dict:
    left = json.loads(Path(before).read_text(encoding="utf-8"))
    right = json.loads(Path(after).read_text(encoding="utf-8"))
    if (left["dataset"]["checksums"] != right["dataset"]["checksums"]
            or left["split"] != right["split"] or left["query_count"] != right["query_count"]):
        raise ValueError("Reports must use the same frozen dataset and query split")
    left_count = left.get("ranking_query_count", left["query_count"])
    right_count = right.get("ranking_query_count", right["query_count"])
    left_no_answer = left.get("no_answer", {"query_count": 0, "metrics": {}})
    right_no_answer = right.get("no_answer", {"query_count": 0, "metrics": {}})
    if (left_count != right_count or left_no_answer["query_count"] != right_no_answer["query_count"]
            or left_count + left_no_answer["query_count"] != left["query_count"]):
        raise ValueError("Reports must use the same answerability denominators")

    def deltas(before, after):
        return {key: {"before": value, "after": after[key], "delta": after[key] - value}
                for key, value in before.items() if key in after}

    return {
        "before": str(before), "after": str(after), "split": left["split"],
        "query_count": left["query_count"], "delta_definition": "after minus before",
        "ranking_query_count": left_count,
        "metrics": deltas(left["metrics"], right["metrics"]),
        "no_answer": {"query_count": left_no_answer["query_count"],
                      "metrics": deltas(left_no_answer["metrics"], right_no_answer["metrics"])},
        "performance": deltas(left["performance"], right["performance"]),
    }
