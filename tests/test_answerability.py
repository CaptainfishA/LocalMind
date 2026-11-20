import json
from pathlib import Path

import pytest

from localmind.evaluation.compare import compare_reports
from localmind.evaluation.dataset import (
    build_text_subset, finish_dataset, load_dataset, read_jsonl, write_json, write_jsonl, file_hash,
)
from localmind.evaluation.fixtures import build_answerability_fixtures
from localmind.evaluation.metrics import no_answer_metrics, ranking_metrics
from localmind.evaluation.runner import run_evaluation


def mutate(root, name, rows):
    write_jsonl(root / name, rows)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["checksums"][name] = file_hash(root / name)
    if name == "queries.jsonl":
        manifest["queries"] = len(rows)
    write_json(root / "manifest.json", manifest)


def test_no_answer_metrics_do_not_claim_ranking_success():
    assert no_answer_metrics([]) == {"candidate_return_rate": 0, "empty_result_rate": 1, "mean_candidate_count": 0}
    assert no_answer_metrics(["a", "a", "b"]) == {
        "candidate_return_rate": 1, "empty_result_rate": 0, "mean_candidate_count": 2}
    with pytest.raises(ValueError, match="at least one"):
        ranking_metrics([], {"a": 0})


def test_answerability_fixture_is_frozen_reproducible_and_completely_judged(tmp_path):
    a = build_answerability_fixtures(tmp_path / "a")
    b = build_answerability_fixtures(tmp_path / "b")
    manifest, docs, queries, rels = load_dataset(a)
    assert manifest == load_dataset(b)[0]
    assert len(docs) == 4 and len(queries) == 14 and len(rels) == 56
    assert sum(q.get("answerability") == "no_answer" for q in queries) == 6
    for split in ("dev", "test"):
        group = [q for q in queries if q["split"] == split]
        assert len(group) == 7 and sum(q.get("answerability") == "no_answer" for q in group) == 3
    assert all("缺失" not in q["text"] for q in queries)
    with pytest.raises(ValueError, match="new directory"):
        build_answerability_fixtures(a)


@pytest.mark.parametrize("corruption", ["missing_judgment", "positive_judgment", "unlabeled", "invalid_label"])
def test_missing_or_conflicting_labels_cannot_become_no_answer(tmp_path, corruption):
    root = build_answerability_fixtures(tmp_path / "data")
    _, _, queries, rels = load_dataset(root)
    query = next(q for q in queries if q.get("answerability") == "no_answer")
    if corruption == "missing_judgment":
        rels.remove(next(r for r in rels if r["query_id"] == query["query_id"]))
        mutate(root, "qrels.jsonl", rels)
    elif corruption == "positive_judgment":
        next(r for r in rels if r["query_id"] == query["query_id"])["relevance"] = 1
        mutate(root, "qrels.jsonl", rels)
    else:
        if corruption == "unlabeled":
            query.pop("answerability")
        else:
            query["answerability"] = "unknown"
        mutate(root, "queries.jsonl", queries)
    with pytest.raises(ValueError):
        load_dataset(root)


def source_rows():
    documents = [{"id": str(i), "text": f"模拟文件 {i} 的资料"} for i in range(5)]
    queries = [{"id": "yes", "text": "模拟文件的资料"},
               {"id": "none", "text": "火星探测器资料", "answerability": "no_answer"},
               {"id": "none-copy", "text": "火星探测器资料", "answerability": "no_answer"}]
    rels = [{"query_id": "yes", "file_id": "0", "relevance": 3}]
    rels += [{"query_id": q, "file_id": d["id"], "relevance": 0}
             for q in ("none", "none-copy") for d in documents]
    return documents, queries, rels


def test_import_retains_explicit_no_answer_and_keeps_duplicate_queries_in_one_split(tmp_path):
    docs, queries, rels = source_rows()
    root = build_text_subset(docs, queries, rels, tmp_path / "subset", query_count=3, corpus_size=3)
    _, corpus, selected, judgments = load_dataset(root)
    assert len(corpus) == 3
    rows = {q["query_id"]: q for q in selected}
    assert rows["none"]["answerability"] == rows["none-copy"]["answerability"] == "no_answer"
    assert rows["none"]["split"] == rows["none-copy"]["split"]
    assert len([r for r in judgments if r["query_id"] == "none"]) == 3
    assert {q["split"] for q in selected} == {"dev", "test"}


@pytest.mark.parametrize("change", ["missing", "positive", "float", "boolean", "label"])
def test_import_rejects_incomplete_negative_source_and_noninteger_grades(tmp_path, change):
    docs, queries, rels = source_rows()
    if change == "missing":
        rels.pop()
    elif change == "positive":
        rels[-1]["relevance"] = 1
    elif change == "float":
        rels[-1]["relevance"] = .5
    elif change == "boolean":
        rels[-1]["relevance"] = False
    else:
        queries[-1]["answerability"] = None
    with pytest.raises(ValueError):
        build_text_subset(docs, queries, rels, tmp_path / "bad", query_count=3, corpus_size=3)
    assert not (tmp_path / "bad").exists()


def test_mixed_evaluation_has_separate_denominators_and_review_rows(tmp_path):
    root = build_answerability_fixtures(tmp_path / "data")
    output = tmp_path / "run"
    report = run_evaluation(root, output, mode="keyword")
    rows = read_jsonl(output / "queries.jsonl")
    positive = [r for r in rows if r["answerability"] == "answerable"]
    negative = [r for r in rows if r["answerability"] == "no_answer"]
    assert report["query_count"] == 7 and report["ranking_query_count"] == len(positive) == 4
    assert report["no_answer"]["query_count"] == len(negative) == 3
    assert all(r["metrics"] == {} for r in negative)
    assert all(r["no_answer_metrics"] == {} for r in positive)
    assert report["metrics"]["hit_at_1"] == sum(r["metrics"]["hit_at_1"] for r in positive) / 4
    assert report["no_answer"]["metrics"]["candidate_return_rate"] == sum(bool(r["ranked"]) for r in negative) / 3
    assert all(r["relevance"] == 0 for q in negative for r in q["ranked"])
    assert report["by_intent"]["missing_identifier"]["ranking_query_count"] == 0
    assert "hit_at_1" not in report["by_intent"]["missing_identifier"]
    assert report["by_intent"]["identifier"]["no_answer"]["query_count"] == 0
    failures = read_jsonl(output / "failures.jsonl")
    assert {r["query_id"] for r in failures} == {
        r["query_id"] for r in rows if r["metrics"].get("hit_at_1", 1) < 1 or r["no_answer_metrics"].get("candidate_return_rate", 0)}


def test_no_answer_only_evaluation_does_not_divide_by_zero_or_report_perfect_ranking(tmp_path):
    root = build_answerability_fixtures(tmp_path / "data")
    manifest, docs, queries, rels = load_dataset(root)
    queries = [q for q in queries if q.get("answerability") == "no_answer"]
    ids = {q["query_id"] for q in queries}
    finish_dataset(root, docs, queries, [r for r in rels if r["query_id"] in ids], {"name": manifest["name"]})
    report = run_evaluation(root, tmp_path / "run", mode="keyword")
    assert report["query_count"] == report["no_answer"]["query_count"] == 3
    assert report["ranking_query_count"] == 0 and report["metrics"] == {}
    assert 0 <= report["no_answer"]["metrics"]["candidate_return_rate"] <= 1


def test_comparison_keeps_no_answer_metrics_and_rejects_denominator_changes(tmp_path):
    before = {"dataset": {"checksums": {"queries": "same"}}, "split": "test", "query_count": 6,
              "ranking_query_count": 4, "metrics": {"hit_at_1": .75}, "performance": {"hot_p50_ms": 10},
              "no_answer": {"query_count": 2, "metrics": {"candidate_return_rate": 1.0}}}
    after = {**before, "no_answer": {"query_count": 2, "metrics": {"candidate_return_rate": .5}}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_json(a, before)
    write_json(b, after)
    comparison = compare_reports(a, b)
    assert comparison["ranking_query_count"] == 4
    assert comparison["no_answer"]["query_count"] == 2
    assert comparison["no_answer"]["metrics"]["candidate_return_rate"]["delta"] == -.5
    after["ranking_query_count"] = 6
    write_json(b, after)
    with pytest.raises(ValueError, match="denominators"):
        compare_reports(a, b)


def test_cli_build_and_validate_answerability_dataset(tmp_path, monkeypatch, capsys):
    import sys
    from localmind.evaluation.__main__ import main

    path = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", ["localmind-eval", "build-answerability", str(path)])
    main()
    assert str(path.resolve()) in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["localmind-eval", "validate", str(path)])
    main()
    assert json.loads(capsys.readouterr().out)["queries"] == 14
    monkeypatch.setattr(sys, "argv", ["localmind-eval", "run", str(path), "--mode", "keyword",
                                      "--output", str(tmp_path / "run")])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["ranking_query_count"] == 4
    assert result["no_answer"]["query_count"] == 3
    assert "candidate_return_rate" in result["no_answer"]["metrics"]


def test_import_can_build_no_answer_only_dataset(tmp_path):
    docs, queries, rels = source_rows()
    queries = [{**queries[1], "id": "a"},
               {**queries[1], "id": "b", "text": "another absent topic"}]
    rels = [{"query_id": q["id"], "file_id": d["id"], "relevance": 0} for q in queries for d in docs]
    root = build_text_subset(docs, queries, rels, tmp_path / "data", query_count=2, corpus_size=3)
    _, corpus, selected, judgments = load_dataset(root)
    assert len(corpus) == 3 and len(judgments) == 6
    assert {q["split"] for q in selected} == {"dev", "test"}
    assert all(q["answerability"] == "no_answer" for q in selected)


def test_comparison_accepts_legacy_positive_report_only_for_same_denominators(tmp_path):
    legacy = {"dataset": {"checksums": {"queries": "same"}}, "split": "test", "query_count": 4,
              "metrics": {"hit_at_1": .75}, "performance": {"hot_p50_ms": 10}}
    current = {**legacy, "ranking_query_count": 4, "no_answer": {"query_count": 0, "metrics": {}}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_json(a, legacy)
    write_json(b, current)
    result = compare_reports(a, b)
    assert result["ranking_query_count"] == 4 and result["no_answer"]["query_count"] == 0
    current.update(ranking_query_count=3, no_answer={"query_count": 1, "metrics": {}})
    write_json(b, current)
    with pytest.raises(ValueError, match="denominators"):
        compare_reports(a, b)
