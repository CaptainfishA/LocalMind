from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_text_subset, load_dataset, read_jsonl
from .fixtures import build_answerability_fixtures, build_fixtures
from .public_data import build_t2
from .runner import run_evaluation
from .compare import compare_reports


def main() -> None:
    parser = argparse.ArgumentParser(prog="localmind-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    fixtures = commands.add_parser("build-fixtures", help="Create generated Office regression files")
    fixtures.add_argument("output")
    fixtures.add_argument("--projects", type=int, default=12)
    fixtures.add_argument("--seed", type=int, default=20261013)
    answerability = commands.add_parser("build-answerability", help="Create closed-world no-answer regression files")
    answerability.add_argument("output")
    for name in ("build-t2", "import-jsonl"):
        prepare = commands.add_parser(name)
        prepare.add_argument("output")
        prepare.add_argument("--queries", type=int, default=100)
        prepare.add_argument("--documents", type=int, default=5000, help="0 retains the full corpus")
        prepare.add_argument("--hard-per-query", type=int, default=10)
        prepare.add_argument("--seed", type=int, default=20261013)
        if name == "build-t2":
            prepare.add_argument("--cache", default="benchmarks/raw")
        else:
            prepare.add_argument("--source", required=True, help="Directory with corpus/queries/qrels.jsonl")
    validate = commands.add_parser("validate")
    validate.add_argument("dataset")
    compare = commands.add_parser("compare")
    compare.add_argument("before")
    compare.add_argument("after")
    run = commands.add_parser("run")
    run.add_argument("dataset")
    run.add_argument("--output", required=True)
    run.add_argument("--split", choices=["dev", "test"], default="test")
    run.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid")
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--rerank", action="store_true")
    run.add_argument("--reranker-runtime", choices=["auto", "cuda", "onnx", "cpu"], default="auto")
    run.add_argument("--rerank-top-k", type=int, default=30)
    run.add_argument("--hash", action="store_true", help="Functional smoke only, not real model quality")
    run.add_argument("--cache-dir")
    run.add_argument("--allow-download", action="store_true")
    run.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    if args.command == "build-fixtures":
        print(build_fixtures(args.output, seed=args.seed, projects=args.projects))
    elif args.command == "build-answerability":
        print(build_answerability_fixtures(args.output))
    elif args.command in {"build-t2", "import-jsonl"}:
        options = dict(seed=args.seed, query_count=args.queries, corpus_size=args.documents,
                       hard_per_query=args.hard_per_query)
        if args.command == "build-t2":
            print(build_t2(args.output, args.cache, **options))
        else:
            source = Path(args.source)
            print(build_text_subset(read_jsonl(source / "corpus.jsonl"), read_jsonl(source / "queries.jsonl"),
                                    read_jsonl(source / "qrels.jsonl"), args.output, **options))
    elif args.command == "validate":
        manifest, *_ = load_dataset(args.dataset)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_reports(args.before, args.after), ensure_ascii=False, indent=2))
    else:
        report = run_evaluation(args.dataset, args.output, split=args.split, mode=args.mode, device=args.device,
                                rerank=args.rerank, reranker_runtime=args.reranker_runtime,
                                rerank_top_k=args.rerank_top_k, force_hash=args.hash, cache_dir=args.cache_dir,
                                local_files_only=not args.allow_download, enable_ocr=args.ocr)
        print(json.dumps({"metrics": report["metrics"], "ranking_query_count": report["ranking_query_count"],
                          "no_answer": report["no_answer"], "performance": report["performance"]}, indent=2))


if __name__ == "__main__":
    main()
