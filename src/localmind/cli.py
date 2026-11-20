from __future__ import annotations

import argparse

from .db import LocalMindDB
from .embeddings import EmbeddingService
from .indexer import IndexingService
from .ocr import OcrService
from .models import SearchQuery
from .reranker import RerankerService
from .runtime import detect_runtime_capabilities
from .searcher import SearchService


def main() -> None:
    parser = argparse.ArgumentParser(prog="localmind-index")
    parser.add_argument("--db", help="Use an explicit index database")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Scan and index a folder")
    scan.add_argument("folder")
    scan.add_argument("--no-ocr", action="store_true")
    scan.add_argument("--no-embedding", action="store_true")
    search = sub.add_parser("search", help="Search indexed files")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--scope", choices=["all", "files", "chat_files", "messages"], default="all")
    search.add_argument("--conversation")
    search.add_argument("--sender")
    search.add_argument("--evidence", choices=["all", "content", "chat"], default="all")
    search.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid")
    search.add_argument("--after", help="Inclusive ISO date/time lower bound")
    search.add_argument("--before", help="Exclusive ISO date/time upper bound")
    search.add_argument("--no-date-inference", action="store_true", help="Treat time words as content, not date filters")
    rerank_group = search.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false")
    search.set_defaults(rerank=None)
    sub.add_parser("stats", help="Show index stats")
    sub.add_parser("clear", help="Clear local index")

    args = parser.parse_args()
    if args.command == "search" and args.evidence != "all" and args.scope != "chat_files":
        parser.error("--evidence requires --scope chat_files")
    if args.command == "search":
        from .search_filters import epoch
        if any(value and epoch(value) is None for value in (args.after, args.before)):
            parser.error("--after/--before require valid ISO dates")
    db = LocalMindDB(args.db)
    try:
        cache_dir = db.get_setting("model_cache_dir")
        if args.command == "scan":
            indexer = IndexingService(
                db,
                embedding_service=EmbeddingService(cache_dir=cache_dir, device=db.get_setting("embedding_device", "auto") or "auto"),
                ocr_service=OcrService(enabled=not args.no_ocr, cache_dir=cache_dir),
            )
            summary = indexer.scan_and_index(
                [args.folder],
                include_embeddings=not args.no_embedding,
                progress=print,
            )
            print(
                f"完成：发现 {summary.files_seen}，更新 {summary.files_indexed}，"
                f"跳过 {summary.files_skipped}，移除 {summary.files_removed}，错误 {summary.errors}"
            )
        elif args.command == "search":
            stored_enabled = db.get_setting("reranker_enabled")
            if args.rerank is not None:
                enable_reranker = args.rerank
            elif stored_enabled is not None:
                enable_reranker = stored_enabled == "1"
            else:
                enable_reranker = detect_runtime_capabilities().default_reranker_enabled
            service = SearchService(
                db,
                embedding_service=EmbeddingService(cache_dir=cache_dir, device=db.get_setting("embedding_device", "auto") or "auto"),
                reranker_service=RerankerService(
                    enabled=enable_reranker,
                    cache_dir=cache_dir,
                    runtime=db.get_setting("reranker_runtime", "auto") or "auto",
                ),
                retrieval_mode=args.mode,
            )
            query = SearchQuery(args.query, limit=args.limit, scope=args.scope, conversation=args.conversation,
                                sender=args.sender, evidence=args.evidence,
                                modified_after=args.after, modified_before=args.before,
                                infer_dates=not args.no_date_inference)
            for index, result in enumerate(service.search(query), start=1):
                print(f"{index}. [排序分 {result.score:.3f}] {result.name}（{result.match_reason}）")
                print(f"   {result.path if result.result_kind == 'file' else '未定位本地文件 / 聊天记录'}")
                print(f"   {result.snippet}")
                if result.chat_context:
                    print(result.chat_context)
        elif args.command == "stats":
            print(db.stats())
        elif args.command == "clear":
            db.clear_index()
            print("本地索引已清空")
    finally:
        db.close()
