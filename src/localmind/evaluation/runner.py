from __future__ import annotations

import platform
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .. import __version__
from ..db import LocalMindDB
from ..embeddings import EmbeddingService
from ..indexer import IndexingService
from ..models import SearchQuery
from ..ocr import OcrService
from ..reranker import RerankerService
from ..searcher import SearchService
from .dataset import file_hash, load_dataset, new_directory, query_answerability, write_json, write_jsonl
from .metrics import mean_metrics, no_answer_metrics, ranking_metrics


def run_evaluation(dataset: str | Path, output: str | Path, *, split: str = "test",
                   mode: str = "hybrid", device: str = "auto", rerank: bool = False,
                   reranker_runtime: str = "auto", rerank_top_k: int = 30,
                   force_hash: bool = False, cache_dir: str | None = None,
                   local_files_only: bool = True, enable_ocr: bool = False) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("Select dev or test")
    base = Path(dataset).resolve()
    manifest, corpus, all_queries, qrels = load_dataset(base)
    queries = [q for q in all_queries if q["split"] == split]
    if not queries:
        raise ValueError(f"No queries in split {split}")
    output_path = Path(output).resolve()
    if output_path.is_relative_to(base):
        raise ValueError("Evaluation output must be outside the frozen dataset")
    root = new_directory(output_path)
    embedding = EmbeddingService(device=device, force_hash=force_hash, cache_dir=cache_dir,
                                 local_files_only=local_files_only)
    reranker = RerankerService(enabled=rerank, cache_dir=cache_dir, runtime=reranker_runtime,
                              local_files_only=local_files_only)
    db = LocalMindDB(root / "index.sqlite3")
    rows = []
    semantic = mode != "keyword"
    try:
        if semantic and not force_hash and embedding.active_model_name != embedding.model_name:
            raise RuntimeError(f"Real-model evaluation cannot use hash fallback: {embedding.load_error}")
        started = time.perf_counter()
        indexer = IndexingService(db, embedding_service=embedding,
                                 ocr_service=OcrService(enabled=enable_ocr, cache_dir=cache_dir))
        summary = indexer.scan_and_index([base / "files"], include_embeddings=semantic)
        indexing_seconds = time.perf_counter() - started
        if summary.errors:
            raise RuntimeError(f"Indexing failed for {summary.errors} files; inspect index.sqlite3")
        paths = {str((base / d["path"]).resolve()).casefold(): d["file_id"] for d in corpus}
        db_ids = {int(r["id"]): paths[str(Path(r["path"]).resolve()).casefold()]
                  for r in db.conn.execute("SELECT id,path FROM files")}
        relevance = {q["query_id"]: {} for q in queries}
        for rel in qrels:
            if rel["query_id"] in relevance:
                relevance[rel["query_id"]][rel["file_id"]] = rel["relevance"]
        started = time.perf_counter()
        service = SearchService(db, embedding_service=embedding, reranker_service=reranker,
                                retrieval_mode=mode, rerank_top_k=rerank_top_k)
        vector_refresh_ms = (time.perf_counter() - started) * 1000
        # Warmup is reported separately; measured query timings are not called cold startup.
        first = queries[0]
        started = time.perf_counter()
        service.search(SearchQuery(text=first["text"], **first.get("filters", {}), limit=10))
        warmup_ms = (time.perf_counter() - started) * 1000
        for index, query in enumerate(queries, 1):
            started = time.perf_counter()
            results = service.search(SearchQuery(text=query["text"], **query.get("filters", {}), limit=10))
            latency = (time.perf_counter() - started) * 1000
            if rerank and reranker.load_error:
                raise RuntimeError(f"Reranker evaluation failed: {reranker.load_error}")
            ranked = [db_ids[result.file_id] for result in results]
            judgments = relevance[query["query_id"]]
            answerability = query_answerability(query)
            metrics = ranking_metrics(ranked, judgments) if answerability == "answerable" else {}
            positives = {key for key, grade in judgments.items() if grade > 0}
            channels = {name: [db_ids[key] for key in ids[:100]]
                        for name, ids in service.last_candidate_file_ids.items()}
            if positives:
                for channel, ids in channels.items():
                    metrics[f"{channel}_candidate_recall"] = len(positives & set(ids)) / len(positives)
                union = set(channels.get("keyword", [])) | set(channels.get("semantic", []))
                metrics["union_candidate_recall"] = len(positives & union) / len(positives)
            rows.append({"query_id": query["query_id"], "text": query["text"], "intent": query.get("intent", []),
                         "answerability": answerability,
                         "no_answer_metrics": no_answer_metrics(ranked) if not positives else {},
                         "metrics": metrics, "latency_ms": latency,
                         "timings_ms": service.last_diagnostics.timings_ms,
                         "ranked": [{"file_id": db_ids[r.file_id], "score": r.score,
                                     "relevance": judgments.get(db_ids[r.file_id]),
                                     "path": str(Path(r.path).relative_to(base)), "reason": r.match_reason}
                                    for r in results]})
            if index % 10 == 0:
                print(f"Evaluated {index}/{len(queries)}", flush=True)
        grouped = {}
        for row in rows:
            for intent in row["intent"]:
                grouped.setdefault(intent, []).append(row)

        def aggregate(group):
            positive = [r["metrics"] for r in group if r["answerability"] == "answerable"]
            negative = [r["no_answer_metrics"] for r in group if r["answerability"] == "no_answer"]
            return {"ranking_query_count": len(positive), "metrics": mean_metrics(positive),
                    "no_answer": {"query_count": len(negative), "metrics": mean_metrics(negative)}}

        by_intent = {}
        for key, group in grouped.items():
            values = aggregate(group)
            by_intent[key] = {"queries": len(group), **values.pop("metrics"), **values}
        packages = {}
        for name in ["torch", "sentence-transformers", "faiss-cpu", "numpy", "PySide6"]:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        report = {
            "version": __version__, "dataset": manifest, "split": split, "query_count": len(rows),
            "configuration": {"mode": mode, "hash_embedding": force_hash, "device_requested": device,
                              "device_active": embedding.device if semantic else "disabled",
                              "runtime_note": embedding.runtime_note,
                              "embedding_model": embedding.active_model_name if semantic else "disabled",
                              "reranker": reranker.model_name if rerank else "disabled",
                              "reranker_runtime": reranker.runtime.name if rerank else "disabled",
                              "rerank_top_k": rerank_top_k, "local_files_only": local_files_only},
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
            "source_sha256": {str(path.relative_to(Path(__file__).parent.parent)): file_hash(path)
                              for path in sorted(Path(__file__).parent.parent.rglob("*.py"))},
            **aggregate(rows),
            "by_intent": by_intent,
            "performance": {"indexing_seconds": indexing_seconds, "vector_refresh_ms": vector_refresh_ms,
                            "warmup_search_ms": warmup_ms, "embedding_load_seconds": embedding.load_seconds,
                            "hot_p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)),
                            "hot_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95))},
            "index": asdict(summary),
            "metric_notes": ["Hit@5 means any positive in top 5; Recall@5 divides by all judged positives.",
                             "Unjudged documents count as zero for ranking; judged_at_10 reports annotation coverage.",
                             "Candidate recall examines up to 100 deduplicated files per recorded channel after filters; it does not measure exhaustive corpus recall.",
                             "Hash and synthetic regression scores are not evidence of real semantic accuracy."],
        }
        report["metric_notes"].extend([
            "Ranking and candidate recall average answerable queries only; ranking_query_count is their denominator.",
            "No-answer queries require explicit zero judgments for every document in the frozen corpus.",
            "No-answer metrics describe whether search returned candidates, not whether it asserted a factual answer.",
            "No score threshold or automatic abstention is applied; an empty result is not proof of calibrated uncertainty.",
        ])
        write_jsonl(root / "queries.jsonl", rows)
        write_jsonl(root / "failures.jsonl", (r for r in rows if (
            r["metrics"].get("hit_at_1", 1) < 1 or r["no_answer_metrics"].get("candidate_return_rate", 0) > 0)))
        write_json(root / "report.json", report)
        return report
    except Exception as exc:
        write_json(root / "error.json", {"error": str(exc), "completed_queries": len(rows)})
        raise
    finally:
        db.close()
