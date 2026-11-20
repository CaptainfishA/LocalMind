from __future__ import annotations

from pathlib import Path
import os

from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.indexer import IndexingService
from localmind.models import SearchQuery
from localmind.monitor import FileEvent, apply_file_events
from localmind.ocr import OcrService
from localmind.reranker import RerankerService
from localmind.searcher import SearchService


def make_searcher(db: LocalMindDB) -> SearchService:
    return SearchService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        reranker_service=RerankerService(enabled=False),
    )


class StubReranker:
    enabled = True
    model_name = "test-reranker"
    load_error = None

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [0.98 if "保证金" in passage else 0.05 for passage in passages]


def test_text_scan_index_and_keyword_search(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "报价单.txt").write_text("A 公司报价单，包含付款信息和年度预算。", encoding="utf-8")
    (docs / "市场分析.md").write_text("日本市场分析 PPT 初稿，讨论渠道策略。", encoding="utf-8")

    db = LocalMindDB(tmp_path / "localmind.db")
    indexer = IndexingService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        ocr_service=OcrService(enabled=False),
    )
    summary = indexer.scan_and_index([docs])

    assert summary.files_seen == 2
    assert summary.files_indexed == 2
    assert db.stats().chunks_total >= 2

    searcher = make_searcher(db)
    results = searcher.search(SearchQuery(text="付款信息", limit=5))
    assert results
    assert results[0].name == "报价单.txt"


def test_semantic_hash_fallback_returns_related_file(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "合同条款.txt").write_text("合同包含保证金条款、付款节点和违约责任。", encoding="utf-8")
    (docs / "旅行计划.txt").write_text("周末旅行计划，包含酒店和路线。", encoding="utf-8")

    db = LocalMindDB(tmp_path / "localmind.db")
    indexer = IndexingService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        ocr_service=OcrService(enabled=False),
    )
    indexer.scan_and_index([docs])

    searcher = make_searcher(db)
    results = searcher.search("有保证金的 PDF 或合同", limit=5)
    assert results
    assert results[0].name == "合同条款.txt"


def test_clear_index_removes_search_results(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "预算.csv").write_text("项目,金额\n年度预算,10000\n", encoding="utf-8")

    db = LocalMindDB(tmp_path / "localmind.db")
    indexer = IndexingService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        ocr_service=OcrService(enabled=False),
    )
    indexer.scan_and_index([docs])
    db.clear_index()

    searcher = make_searcher(db)
    assert searcher.search("年度预算") == []


def test_natural_chinese_query_uses_meaningful_keyword_terms(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "客户资料.txt").write_text("付款信息包括银行账号和结算日期。", encoding="utf-8")
    (docs / "会议安排.txt").write_text("下周召开团队例会。", encoding="utf-8")

    db = LocalMindDB(tmp_path / "localmind.db")
    IndexingService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        ocr_service=OcrService(enabled=False),
    ).scan_and_index([docs])

    results = make_searcher(db).search("帮我找包含付款信息的文件", limit=5)
    assert results
    assert results[0].name == "客户资料.txt"
    assert results[0].keyword_score > 0
    assert "关键词" in results[0].match_reason


def test_reranker_score_is_used_and_explained(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "目标合同.txt").write_text("合同约定履约保证金和付款节点。", encoding="utf-8")
    (docs / "普通合同.txt").write_text("合同记录了联系人和会议时间。", encoding="utf-8")

    db = LocalMindDB(tmp_path / "localmind.db")
    IndexingService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        ocr_service=OcrService(enabled=False),
    ).scan_and_index([docs])
    searcher = SearchService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        reranker_service=StubReranker(),  # type: ignore[arg-type]
    )

    results = searcher.search("有保证金条款的合同")
    assert results[0].name == "目标合同.txt"
    assert results[0].reranker_score == 0.98
    assert "AI 精排" in results[0].match_reason


def test_rescan_updates_changed_file_and_removes_deleted_file(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    changed = docs / "合同.txt"
    deleted = docs / "旧报价.txt"
    changed.write_text("旧内容：保证金为一万元。", encoding="utf-8")
    deleted.write_text("即将删除的蓝色报价单。", encoding="utf-8")

    db = LocalMindDB(tmp_path / "localmind.db")
    indexer = IndexingService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        ocr_service=OcrService(enabled=False),
    )
    indexer.scan_and_index([docs])

    changed.write_text("新内容：付款节点调整为验收后十天。", encoding="utf-8")
    timestamp = changed.stat().st_mtime + 2
    os.utime(changed, (timestamp, timestamp))
    deleted.unlink()
    summary = indexer.scan_and_index([docs])

    searcher = make_searcher(db)
    assert all(result.name != "旧报价.txt" for result in searcher.search("蓝色报价单"))
    results = searcher.search("付款节点")
    assert results and results[0].name == "合同.txt"
    assert summary.files_removed == 1
    assert summary.files_indexed == 1


def test_file_monitor_move_event_replaces_old_path(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    old_path = docs / "临时报价.txt"
    new_path = docs / "客户最终报价.txt"
    old_path.write_text("A 公司最终报价为八万元。", encoding="utf-8")

    db_path = tmp_path / "localmind.db"
    db = LocalMindDB(db_path)
    IndexingService(
        db,
        embedding_service=EmbeddingService(force_hash=True),
        ocr_service=OcrService(enabled=False),
    ).scan_and_index([docs])

    old_path.rename(new_path)
    summary = apply_file_events(
        db_path,
        [FileEvent("moved", str(old_path), str(new_path))],
        enable_ocr=False,
        enable_embeddings=False,
    )

    assert summary.removed == 1
    assert summary.indexed == 1
    assert db.get_file_by_path(str(old_path)) is None
    assert db.get_file_by_path(str(new_path)) is not None
    results = make_searcher(db).search("最终报价")
    assert results and results[0].name == "客户最终报价.txt"
