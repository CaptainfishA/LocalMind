from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from sqlite3 import Row
from typing import Callable

from .config import RERANK_TOP_K
from .chats import CHAT_PREFIX, ChatStore, chat_source_reference, context_label
from .db import LocalMindDB
from .embeddings import EmbeddingService, HashEmbeddingBackend
from .models import SearchEvidence, SearchQuery, SearchResult
from .query_processing import (
    ProcessedQuery, process_query, semantic_clauses, semantic_clause_sources,
    semantic_clause_texts, single_source_clauses,
)
from .reranker import RerankerService
from .vector_index import LocalVectorIndex
from .file_freshness import annotate_results
from .companion_recall import CompanionRecall, CompanionRequest, recall_companions
from .clause_semantics import coverage_ranks, evidence_key, evidence_parts, file_clause_evidence
from .identifiers import identifier_content, identifier_position, row_has_identifier
from .preview_focus import MAX_ANCHOR_CHARS
from .paragraph_refinement import ParagraphRefiner


# These can still retrieve files, but cannot independently establish a second
# source of evidence just because they occur in a context header or request.
_JOINT_GENERIC_TERMS = {
    "资料", "材料", "内容", "文件", "有关", "关于", "需要", "使用", "要用", "发给",
    "提供", "用于", "相关", "这份", "那份", "一份", "消息", "附件", "记录",
    "平台", "会话", "发送者", "消息时间", "微信", "未知",
    "同事", "人员", "朋友", "用户", "群聊", "群组",
}


@dataclass(slots=True)
class SearchDiagnostics:
    query_terms: list[str]
    keyword_candidates: int
    semantic_candidates: int
    total_candidates: int
    indexed_embedding_model: str | None
    active_embedding_model: str | None
    semantic_status: str
    reranker_status: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieval_mode: str = "hybrid"
    ranking_profile: str = "balanced"
    scope: str = "all"
    evidence: str = "all"
    companion_candidates: int = 0
    companion_checked_files: int = 0
    companion_requested_files: int = 0
    companion_status: str = ""
    index_revision: int | None = None
    vector_revision: int | None = None
    index_current: bool = True
    snapshot_hold_ms: float = 0.0
    caller_transaction: bool = False
    clause_count: int = 0
    clause_status: str = ""
    modified_after: str | None = None
    modified_before: str | None = None
    identifier: str = ""
    identifier_matches: int = 0
    refinement_status: str = ""
    refinement_selected: int = 0
    refinement_encoded: int = 0
    refinement_cache_hits: int = 0
    refinement_improved: int = 0


@dataclass(slots=True)
class _PreparedSemantic:
    vector: object | None
    model: str | None
    status: str
    clauses: tuple[str, ...] = ()
    clause_vectors: object | None = None
    clause_error: str = ""


@dataclass(slots=True)
class _CandidateBatch:
    query: SearchQuery
    processed: ProcessedQuery
    items: list[dict]
    diagnostics: SearchDiagnostics
    joint_enabled: bool
    semantic_scores: dict[int, float] = field(default_factory=dict)


class SearchCancelled(Exception):
    """A cooperative stop between complete search stages."""


class SearchService:
    def __init__(
        self,
        db: LocalMindDB,
        *,
        embedding_service: EmbeddingService | None = None,
        vector_index: LocalVectorIndex | None = None,
        reranker_service: RerankerService | None = None,
        retrieval_mode: str = "hybrid",
        rerank_top_k: int | None = None,
        enable_clause_semantics: bool = True,
        enable_paragraph_refinement: bool = True,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self.reranker_service = reranker_service or RerankerService()
        if retrieval_mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")
        self.retrieval_mode = retrieval_mode
        self.rerank_top_k = rerank_top_k
        self.enable_clause_semantics = enable_clause_semantics
        self.enable_paragraph_refinement = enable_paragraph_refinement
        self._paragraph_refiner = ParagraphRefiner()
        self.should_cancel = should_cancel
        self.last_candidate_file_ids: dict[str, list[int]] = {}
        self.vector_index = vector_index or LocalVectorIndex()
        self.chat_store = ChatStore(db)
        self._scope_key = None
        self._allowed_ids: set[int] | None = None
        self._target_mode = "files"
        self._target_groups: dict[str, dict[int, str]] = {}
        self._source_ids: dict[str, set[int]] = {}
        self._index_revision: int | None = None
        self._vector_revision: int | None = None
        self._links_revision: int | None = None
        self.indexed_embedding_model: str | None = None
        self.refresh_vector_index()
        self.last_diagnostics = SearchDiagnostics(
            query_terms=[],
            keyword_candidates=0,
            semantic_candidates=0,
            total_candidates=0,
            indexed_embedding_model=self.indexed_embedding_model,
            active_embedding_model=None,
            semantic_status="尚未搜索",
            reranker_status="尚未搜索",
        )

    def _check_cancelled(self):
        callback = getattr(self, "should_cancel", None)
        if callback is not None and callback():
            raise SearchCancelled("Search cancelled")

    def _prepare_links(self):
        self._check_cancelled()
        if self.db.conn.in_transaction:
            return
        revision, _ = self.db.index_revisions()
        if revision != self._links_revision:
            self.chat_store.resolve_links()
            # Do not claim to have resolved writes committed after this initial read.
            self._links_revision = revision

    def _refresh_snapshot_caches(self, revisions, *, force=False):
        self._check_cancelled()
        revision, vector_revision = revisions
        if revision != self._index_revision:
            self._paragraph_refiner.clear()
            self._scope_key = None
            self._target_groups.clear()
            self._source_ids.clear()
            self._index_revision = revision
        if force or vector_revision != self._vector_revision:
            self.indexed_embedding_model = self._resolve_indexed_model()
            ids, vectors = self.db.load_vectors(model_name=self.indexed_embedding_model)
            self._check_cancelled()
            self.vector_index.rebuild(ids, vectors)
            self._vector_revision = vector_revision
        self._check_cancelled()
        return len(self.vector_index.ids)

    def _forget_snapshot(self):
        self._paragraph_refiner.clear()
        self._index_revision = self._vector_revision = None
        self._scope_key = None
        self._target_groups.clear()
        self._source_ids.clear()

    def clear_refinement_cache(self):
        self._paragraph_refiner.clear()

    def refresh_vector_index(self, *, force=True) -> int:
        self._check_cancelled()
        caller_transaction = self.db.conn.in_transaction
        self._prepare_links()
        try:
            with self.db.search_snapshot() as revisions:
                return self._refresh_snapshot_caches(revisions, force=force)
        finally:
            if caller_transaction:
                self._forget_snapshot()

    def search(self, query: SearchQuery | str, *, limit: int | None = None) -> list[SearchResult]:
        self._check_cancelled()
        search_query = query if isinstance(query, SearchQuery) else SearchQuery(text=query)
        if limit is not None:
            search_query.limit = limit
        text = search_query.text.strip()
        if not text:
            return []

        query_started = time.perf_counter()
        explicit_dates = search_query.modified_after is not None or search_query.modified_before is not None
        processed = (process_query(text) if search_query.infer_dates
                     else process_query(text, infer_dates=False))
        self._check_cancelled()
        clauses = self._query_clauses(search_query, processed)
        if not explicit_dates:
            search_query = replace(
                search_query, modified_after=processed.modified_after, modified_before=processed.modified_before)
        query_ms = (time.perf_counter() - query_started) * 1000
        caller_transaction = self.db.conn.in_transaction
        sync_started = time.perf_counter()
        self._prepare_links()
        sync_ms = (time.perf_counter() - sync_started) * 1000
        hold_ms = embedding_ms = 0.0
        prepared = None
        prior_diagnostics = self.last_diagnostics
        prior_candidates = self.last_candidate_file_ids
        try:
            # At most two snapshots: test eligibility, then collect with a prepared vector.
            # Model loading/inference never extends a search-owned read transaction.
            while True:
                self._check_cancelled()
                snapshot_started = time.perf_counter()
                with self.db.search_snapshot() as revisions:
                    self._refresh_snapshot_caches(revisions)
                    scope_started = time.perf_counter()
                    sync_ms += (scope_started - snapshot_started) * 1000
                    self._set_scope(search_query)
                    self._check_cancelled()
                    query_ms += (time.perf_counter() - scope_started) * 1000
                    needs_embedding = (self.retrieval_mode != "keyword" and prepared is None
                                       and self.vector_index.has_vectors(self._allowed_ids))
                    if not needs_embedding:
                        batch = self._collect_candidates(search_query, processed, prepared)
                hold_ms += (time.perf_counter() - snapshot_started) * 1000
                if not needs_embedding:
                    break
                embedding_started = time.perf_counter()
                prepared = self._prepare_semantic(
                    processed.semantic_text, clauses if self.indexed_embedding_model != HashEmbeddingBackend.name else ())
                self._check_cancelled()
                embedding_ms = (time.perf_counter() - embedding_started) * 1000
            self._check_cancelled()
            refinement = self._refine_candidates(batch, prepared, caller_transaction=caller_transaction)
            self._check_cancelled()
            results = self._rank_candidates(batch)
            self._check_cancelled()
            self.last_diagnostics.index_revision, self.last_diagnostics.vector_revision = revisions
            self.last_diagnostics.timings_ms.update(query=query_ms, snapshot=sync_ms, embedding=embedding_ms)
            self.last_diagnostics.snapshot_hold_ms = hold_ms
            self.last_diagnostics.caller_transaction = caller_transaction
            self.last_diagnostics.index_current = self.db.index_revisions() == revisions
            self._check_cancelled()
            if self.last_diagnostics.index_current and not caller_transaction:
                self._paragraph_refiner.commit(refinement)
            return results
        except SearchCancelled:
            # TEMP scope writes may have rolled back; completed vector caches are
            # still valid and checked against the persisted version on the next call.
            self._scope_key = None
            self.last_diagnostics = prior_diagnostics
            self.last_candidate_file_ids = prior_candidates
            raise
        except BaseException:
            # A failed read rolls back its temporary scope table as well.
            self._forget_snapshot()
            raise
        finally:
            if caller_transaction:
                # The caller can roll back and later reuse the same revision numbers.
                self._forget_snapshot()

    def _query_clauses(self, query: SearchQuery, processed: ProcessedQuery) -> tuple[str, ...]:
        if not self.enable_clause_semantics or self.retrieval_mode != "hybrid":
            return ()
        if query.evidence == "all" and query.scope in {"all", "chat_files"}:
            return semantic_clauses(processed)
        if query.scope == "files" and query.evidence == "all":
            return single_source_clauses(processed, "content")
        if query.scope == "chat_files" and query.evidence in {"content", "chat"}:
            return single_source_clauses(processed, query.evidence)
        return ()

    def _clause_targets(self, items):
        # Pending attachment windows have separate storage files but one recall target.
        groups = self._target_groups.get(self._target_mode, {})
        return {item["chunk_id"]: groups.get(item["chunk_id"], f"file:{item['row']['file_id']}")
                for item in items}

    def _set_scope(self, search_query):
        self._target_mode = "messages" if search_query.scope == "messages" else "files"
        scope_key = (
            search_query.scope, search_query.chat_platform, search_query.conversation, search_query.sender,
            search_query.evidence, search_query.file_type, search_query.path_contains,
            search_query.modified_after, search_query.modified_before,
        )
        if scope_key != self._scope_key:
            allowed_ids = self.chat_store.scope_ids(search_query)
            if allowed_ids is not None:
                self.db.set_search_scope(allowed_ids)
            self._allowed_ids = allowed_ids
            self._scope_key = scope_key

    def _collect_candidates(self, search_query, processed, prepared):
        self._check_cancelled()
        started = time.perf_counter()
        candidate_limit = max(100, search_query.limit * 8)
        keyword_ranks = self._keyword_candidates(processed, candidate_limit) if self.retrieval_mode != "semantic" else {}
        keyword_at = time.perf_counter()
        self._check_cancelled()
        if self.retrieval_mode != "keyword":
            semantic_scores, semantic_ranks, active_model, semantic_status = self._semantic_candidates(
                processed.semantic_text, candidate_limit, prepared,
            )
        else:
            semantic_scores, semantic_ranks, active_model, semantic_status = {}, {}, None, "仅关键词"
        semantic_at = time.perf_counter()
        self._check_cancelled()
        descriptive = processed.descriptive and bool(semantic_scores) and active_model != HashEmbeddingBackend.name

        all_chunk_ids = set(keyword_ranks) | set(semantic_ranks)
        chunk_rows = self.chat_store.enrich(self.db.fetch_chunks_by_ids(all_chunk_ids), search_query)
        self._check_cancelled()
        self.last_candidate_file_ids = {
            name: _dedupe(int(chunk_rows[cid]["file_id"]) for cid in ranks
                          if cid in chunk_rows and self._passes_filters(chunk_rows[cid], search_query, processed))
            for name, ranks in [("keyword", keyword_ranks), ("semantic", semantic_ranks)]
        }
        eligible: list[dict[str, object]] = []
        for chunk_id, row in chunk_rows.items():
            if not self._passes_filters(row, search_query, processed):
                continue
            eligible.append(self._score_row(
                chunk_id, row, processed, descriptive=descriptive,
                keyword_rank=keyword_ranks.get(chunk_id), semantic_rank=semantic_ranks.get(chunk_id),
                semantic_score=semantic_scores.get(chunk_id, 0.0),
            ))

        companion_started = time.perf_counter()
        companion_report = CompanionRecall()
        initial_count = len(eligible)
        joint_enabled = (self.retrieval_mode != "semantic" and search_query.evidence == "all"
                         and search_query.scope in {"all", "chat_files"})
        if joint_enabled:
            companion_report = self._extend_joint_candidates(eligible, processed, search_query, descriptive=descriptive)
        self._check_cancelled()
        companion_ms = (time.perf_counter() - companion_started) * 1000
        clause_started = time.perf_counter()
        clause_count = 0
        clause_status = prepared.clause_error if prepared else ""
        if prepared and prepared.clause_vectors is not None and active_model == self.indexed_embedding_model:
            try:
                scores = self.vector_index.score_ids(prepared.clause_vectors, [item["chunk_id"] for item in eligible])
                self._check_cancelled()
                single_source = search_query.scope == "files" or search_query.evidence in {"content", "chat"}
                whole_scores = None
                if single_source:
                    whole_scores = dict(semantic_scores)
                    missing = scores.keys() - whole_scores.keys()
                    if missing:
                        additions = self.vector_index.score_ids(prepared.vector, missing)
                        whole_scores.update({cid: values[0] for cid, values in additions.items()})
                        self._check_cancelled()
                targets = self._clause_targets(eligible)
                coverage = file_clause_evidence(eligible, scores, prepared.clauses,
                                                whole_scores=whole_scores, target_keys=targets)
                semantic_target_ranks = coverage_ranks(coverage)
                keyword_target_ranks = {}
                for item in eligible:
                    target, rank = targets[item["chunk_id"]], item["keyword_rank"]
                    if rank is not None:
                        keyword_target_ranks[target] = min(rank, keyword_target_ranks.get(target, rank))
                updates = []
                for item in eligible:
                    target = targets[item["chunk_id"]]
                    value, parts = coverage.get(target, (0.0, ()))
                    fused = reciprocal_rank_fusion(keyword_target_ranks.get(target), semantic_target_ranks.get(target))
                    preliminary = self._base_score(
                        item["lexical"], value, fused, item["filename"], item["extension_bonus"],
                        descriptive=descriptive,
                    )
                    updates.append((item, value, parts, fused, preliminary))
                for item, value, parts, fused, preliminary in updates:
                    item["semantic"] = value
                    item["fused"] = fused
                    item["semantic_evidence"] = parts
                    if parts:
                        item["reasons"].insert(0, "分句语义覆盖")
                    item["preliminary"] = preliminary
                clause_count = len(prepared.clauses)
                source_names = {"all": ("正文" if search_query.scope == "files" or search_query.evidence == "content" else
                                        "聊天" if search_query.evidence == "chat" else "不限来源"),
                                "chat": "聊天", "content": "正文"}
                source_label = " / ".join(source_names[source] for source in semantic_clause_sources(prepared.clauses))
                scoring = "整句相似度与分句调和平均各占一半" if single_source else "调和平均覆盖分"
                clause_status = f"按同一目标合并两条描述（{source_label}），语义分为{scoring}，不是正确概率"
            except SearchCancelled:
                raise
            except Exception as exc:
                clause_status = f"分句语义不可用，保留整句排序：{exc}"
        clause_ms = (time.perf_counter() - clause_started) * 1000
        self._check_cancelled()
        diagnostics = SearchDiagnostics(
            query_terms=processed.terms,
            keyword_candidates=len(keyword_ranks),
            semantic_candidates=len(semantic_ranks),
            total_candidates=len(eligible),
            indexed_embedding_model=self.indexed_embedding_model,
            active_embedding_model=active_model,
            semantic_status=semantic_status,
            reranker_status="尚未精排",
            retrieval_mode=self.retrieval_mode,
            ranking_profile="descriptive" if descriptive else "balanced",
            scope=search_query.scope,
            evidence=search_query.evidence,
            identifier=processed.identifier,
            clause_count=clause_count,
            clause_status=clause_status,
            modified_after=search_query.modified_after,
            modified_before=search_query.modified_before,
            companion_candidates=len(eligible) - initial_count,
            companion_checked_files=companion_report.checked_files,
            companion_requested_files=companion_report.requested_files,
            companion_status=("互补检查未完成：达到时间预算" if companion_report.time_limited else
                              "互补检查未完成：达到文件上限" if companion_report.file_limited else
                              "互补检查完成" if companion_report.requested_files else ""),
            timings_ms={
                "keyword": (keyword_at - started) * 1000,
                "semantic": (semantic_at - keyword_at) * 1000,
                "companion": companion_ms,
                "clause": clause_ms,
                "fusion": (companion_started - semantic_at) * 1000,
            },
        )
        return _CandidateBatch(search_query, processed, eligible, diagnostics, joint_enabled, semantic_scores)

    def _refine_candidates(self, batch, prepared, *, caller_transaction=False):
        if (not self.enable_paragraph_refinement or self.retrieval_mode != "hybrid" or caller_transaction
                or batch.processed.identifier or not batch.processed.descriptive
                or batch.query.evidence == "chat" or batch.query.scope == "messages"
                or not prepared or prepared.vector is None or prepared.clauses
                or prepared.model != self.indexed_embedding_model or prepared.model == HashEmbeddingBackend.name):
            return None
        started = time.perf_counter()
        diagnostics = batch.diagnostics
        identity = prepared.model, id(getattr(self.embedding_service, "_backend", self.embedding_service))
        try:
            plan = self._paragraph_refiner.prepare(batch.items, prepared.vector, identity=identity,
                encoder=self.embedding_service.encode_documents, terms=batch.processed.terms,
                device=getattr(self.embedding_service, "device", "cpu"), check=self._check_cancelled)
            self._check_cancelled()
            current = (self.embedding_service.active_model_name,
                       id(getattr(self.embedding_service, "_backend", self.embedding_service)))
            if current != identity:
                self._paragraph_refiner.clear()
                raise ValueError("Refinement model changed")
            diagnostics.refinement_selected = plan.selected
            diagnostics.refinement_encoded = plan.encoded
            diagnostics.refinement_cache_hits = plan.cache_hits
            diagnostics.refinement_status = ("正文细化完成" if plan.complete else "正文细化达到预算，保留基础排序") if plan.selected else ""
            if not plan.complete:
                return plan
            updates = {item["chunk_id"]: plan.matches[item["chunk_id"]] for item in batch.items
                       if item["chunk_id"] in plan.matches and plan.matches[item["chunk_id"]][1] > item["semantic"] + 1e-6}
            if not updates:
                return plan
            scores = {**batch.semantic_scores, **{cid: value[1] for cid, value in updates.items()}}
            prior_order = {cid: n for n, cid in enumerate(batch.semantic_scores)}
            order = sorted(scores, key=lambda cid: (-scores[cid], prior_order.get(cid, len(prior_order)), cid))
            ranks = {cid: n for n, cid in enumerate(order, 1)}
            staged = []
            for item in batch.items:
                semantic = updates[item["chunk_id"]][1] if item["chunk_id"] in updates else item["semantic"]
                fused = reciprocal_rank_fusion(item["keyword_rank"], ranks.get(item["chunk_id"]))
                preliminary = self._base_score(item["lexical"], semantic, fused, item["filename"], item["extension_bonus"],
                                               descriptive=diagnostics.ranking_profile == "descriptive")
                changed = dict(semantic=semantic, fused=fused, preliminary=preliminary)
                if item["chunk_id"] in updates:
                    fragment, _ = updates[item["chunk_id"]]
                    changed["detail_evidence"] = dict(chunk_id=item["chunk_id"], refined=True,
                        row={**item["row"], "chunk_text": fragment.text})
                    changed["reasons"] = ["正文片段匹配", *item["reasons"]]
                staged.append((item, changed))
            self._check_cancelled()
            for item, changed in staged:
                item.update(changed)
            diagnostics.refinement_improved = len(updates)
            return plan
        except SearchCancelled:
            raise
        except Exception:
            diagnostics.refinement_status = "正文细化不可用，保留基础排序"
            return None
        finally:
            diagnostics.timings_ms["refinement"] = (time.perf_counter() - started) * 1000

    def _rank_candidates(self, batch):
        self._check_cancelled()
        search_query, processed, eligible = batch.query, batch.processed, batch.items
        diagnostics = batch.diagnostics
        fusion_started = time.perf_counter()
        descriptive = diagnostics.ranking_profile == "descriptive"
        if batch.joint_enabled:
            self._add_joint_evidence(eligible, processed, descriptive=descriptive)
        self._check_cancelled()
        prefer_identifier = self.retrieval_mode != "semantic" and bool(processed.identifier)
        eligible.sort(key=lambda item: (-int(prefer_identifier and item["identifier_matched"]),
                                        -float(item["preliminary"]), int(item["chunk_id"])))
        rerank_candidates: list[dict[str, object]] = []
        rerank_pairs = set()
        if self.reranker_service.enabled:
            rerank_limit = min(
                RERANK_TOP_K,
                int(getattr(self.reranker_service, "recommended_top_k", RERANK_TOP_K)),
            )
            if self.rerank_top_k is not None:
                rerank_limit = max(1, min(self.rerank_top_k, 100))
            counts: dict[int, int] = {}
            for item in eligible:
                fid = int(item["row"]["file_id"])
                pair_key = evidence_key(item)
                if pair_key and pair_key in rerank_pairs:
                    continue
                if counts.get(fid, 0) >= 2:
                    continue
                rerank_candidates.append(item)
                if pair_key:
                    rerank_pairs.add(pair_key)
                counts[fid] = counts.get(fid, 0) + 1
                if len(rerank_candidates) >= rerank_limit:
                    break
        fusion_at = time.perf_counter()
        reranker_scores = self._rerank(processed, rerank_candidates)
        self._check_cancelled()
        pair_scores = {
            evidence_key(item): reranker_scores[item["chunk_id"]]
            for item in rerank_candidates if evidence_parts(item) and item["chunk_id"] in reranker_scores
        }
        for item in eligible:
            key = evidence_key(item)
            if key in pair_scores:
                reranker_scores[item["chunk_id"]] = pair_scores[key]
        rerank_at = time.perf_counter()
        reranker_status = self._reranker_status(reranker_scores)

        chunk_results: list[SearchResult] = []
        for item in eligible:
            chunk_id = int(item["chunk_id"])
            row = item["row"]
            lexical_score = float(item["lexical"])
            filename_score = float(item["filename"])
            semantic_score = float(item["semantic"])
            fused_score = float(item["fused"])
            reranker_score = reranker_scores.get(chunk_id)
            extension_bonus = float(item["extension_bonus"])
            if reranker_scores:
                final_score = (
                    (reranker_score or 0.0) * 0.62
                    + lexical_score * 0.16
                    + semantic_score * 0.10
                    + fused_score * 0.07
                    + filename_score * 0.05
                    + extension_bonus
                )
            else:
                final_score = float(item["preliminary"])
            reasons = list(item["reasons"])
            joint = item.get("joint_evidence", ())
            parts = evidence_parts(item)
            semantic_ids = {part["chunk_id"] for part in item.get("semantic_evidence", ())}
            if item["identifier_matched"] and not any(part["chunk_id"] == chunk_id for part in parts):
                parts = ({"chunk_id": chunk_id, "row": row}, *parts)
            semantic_chat = next((part for part in item.get("semantic_evidence", ())
                                  if part["row"]["source_type"] == "chat"), None)
            refs = (semantic_chat["row"]["chat_refs"] if semantic_chat else
                    joint[1]["row"]["chat_refs"] if joint else row.get("chat_refs", []))
            matching_refs = [r for r in refs if
                             (not search_query.chat_platform or search_query.chat_platform.lower() in r["platform"].lower())
                             and (not search_query.conversation or search_query.conversation.lower() in r["conversation"].lower())
                             and (not search_query.sender or search_query.sender.lower() in r["sender"].lower())]
            ref = (matching_refs or refs or [None])[0]
            if row["source_type"] == "chat":
                reasons.insert(0, "合并转发附件上下文" if ref and ref.get("origin") else "聊天上下文")
            elif ref:
                reasons.insert(0, "文件正文 + 聊天来源")
            if reranker_score is not None:
                reasons.insert(0, "AI 精排")
            elif self.retrieval_mode == "semantic":
                reasons.insert(0, "语义排序")
            if extension_bonus:
                reasons.append("类型符合描述")
            if joint:
                reasons.insert(0, "正文与聊天互补")
            if processed.identifier:
                reasons.insert(0, "编号完整命中" if item["identifier_matched"] else "相似候选")
            preview_row = next((part["row"] for part in parts if part["row"]["source_type"] != "chat"),
                               row if row["source_type"] != "chat" else None)
            chunk_results.append(
                SearchResult(
                    file_id=int(row["file_id"]),
                    chunk_id=chunk_id,
                    name=row["name"],
                    path=row["path"],
                    ext=row["ext"],
                    modified_at=row["modified_at"],
                    score=max(0.0, min(final_score, 0.999)),
                    keyword_score=lexical_score,
                    semantic_score=semantic_score,
                    fused_score=fused_score,
                    reranker_score=reranker_score,
                    snippet=make_snippet(identifier_content(row["chunk_text"], row["source_type"])
                                         if processed.identifier else item.get("detail_evidence", {}).get("row", row)["chunk_text"], processed),
                    source_type=row["source_type"],
                    match_reason="、".join(_dedupe(reasons)) or "语义相关",
                    page_no=row["page_no"],
                    sheet_name=row["sheet_name"],
                    is_ocr=str(row["source_type"]).startswith("ocr"),
                    ocr_confidence=row["ocr_confidence"],
                    result_kind=("attachment_pending" if ref and ref.get("attachment_id") and search_query.scope != "messages" else "message") if str(row["path"]).startswith(CHAT_PREFIX) else "file",
                    chat_context=context_label(ref) if ref else "",
                    chat_source=chat_source_reference(ref),
                    identifier_query=processed.identifier,
                    identifier_matched=item["identifier_matched"],
                    preview_anchor=make_preview_anchor(preview_row["chunk_text"], processed,
                                                       source_type=preview_row["source_type"]) if preview_row else "",
                    chat_message_id=ref["message_id"] if ref else None,
                    attachment_id=ref["attachment_id"] if ref else None,
                    link_state=ref["link_state"] or "" if ref else "",
                    source_query=replace(search_query, text="") if ref else None,
                    supporting_evidence=tuple(
                        SearchEvidence(
                            chunk_id=int(part["chunk_id"]), source_type=part["row"]["source_type"],
                            snippet=(make_snippet(identifier_content(part["row"]["chunk_text"],
                                                                     part["row"]["source_type"]), processed)
                                     if processed.identifier else part["row"]["chunk_text"]
                                     if (part["chunk_id"] in semantic_ids
                                         or item.get("detail_evidence") and part["chunk_id"] == chunk_id) else
                                     make_evidence_snippet(part["row"]["chunk_text"], processed.terms)),
                            matched_terms=tuple(term for term in processed.terms
                                                if term_matches(part["row"]["chunk_text"], term, processed)),
                            page_no=part["row"]["page_no"], sheet_name=part["row"]["sheet_name"],
                            query_clause=part.get("query_clause", ""),
                            chat_source=chat_source_reference(next(iter(part["row"].get("chat_refs", ())), None))
                            if part["row"]["source_type"] == "chat" else None,
                            refined=bool(part.get("refined")),
                        ) for part in parts
                    ),
                )
            )

        merged: dict[object, SearchResult] = {}
        def result_order(item):
            return (int(prefer_identifier and item.identifier_matched), item.score)

        for result in sorted(chunk_results, key=result_order, reverse=True):
            key = (result.chat_message_id, result.attachment_id) if result.result_kind != "file" else result.file_id
            if key not in merged:
                merged[key] = result
            elif result.chat_context and not merged[key].chat_context:
                merged[key].chat_context = result.chat_context
                merged[key].chat_message_id = result.chat_message_id
                merged[key].attachment_id = result.attachment_id
                merged[key].source_query = result.source_query
                merged[key].chat_source = result.chat_source
        aggregate_at = time.perf_counter()
        results = sorted(merged.values(), key=result_order, reverse=True)[: search_query.limit]
        diagnostics.identifier_matches = sum(result.identifier_matched for result in results)
        self._check_cancelled()
        annotate_results(self.db, results)
        self._check_cancelled()
        checked_at = time.perf_counter()
        diagnostics.reranker_status = reranker_status
        diagnostics.timings_ms["fusion"] += (fusion_at - fusion_started) * 1000
        diagnostics.timings_ms.update(
            rerank=(rerank_at - fusion_at) * 1000,
            aggregate=(aggregate_at - rerank_at) * 1000,
            file_check=(checked_at - aggregate_at) * 1000,
        )
        self.last_diagnostics = diagnostics
        return results

    def _score_row(self, chunk_id, row, query, *, descriptive=False,
                   keyword_rank=None, semantic_rank=None, semantic_score=0.0):
        lexical_score, filename_score, reasons = lexical_evidence(row, query)
        fused_score = reciprocal_rank_fusion(keyword_rank, semantic_rank)
        extension_bonus = 0.06 if row["ext"].lower() in query.desired_extensions else 0.0
        return {
            "chunk_id": chunk_id, "row": row, "lexical": lexical_score, "filename": filename_score,
            "keyword_rank": keyword_rank,
            "semantic": semantic_score, "fused": fused_score, "extension_bonus": extension_bonus,
            "preliminary": self._base_score(lexical_score, semantic_score, fused_score, filename_score,
                                           extension_bonus, descriptive=descriptive),
            "reasons": reasons,
            "identifier_matched": row_has_identifier(row, query.identifier),
        }

    def _extend_joint_candidates(self, items, query, search_query, *, descriptive):
        terms = [term for term in query.terms if term not in _JOINT_GENERIC_TERMS]
        if len(terms) < 2:
            return CompanionRecall()
        groups = {}
        for item in items:
            row = item["row"]
            if str(row["path"]).startswith(CHAT_PREFIX):
                continue
            groups.setdefault(row["file_id"], []).append(item)
        requests = []
        for fid, candidates in sorted(groups.items(), key=lambda group: (
                -max(float(item["preliminary"]) for item in group[1]), group[0])):
            bodies = [item for item in candidates if item["row"]["source_type"] != "chat"]
            chats = [item for item in candidates if item["row"]["source_type"] == "chat" and any(
                ref.get("target_id") == fid and (
                    not ref.get("linked_hash") or ref.get("linked_hash") == ref.get("target_hash"))
                for ref in item["row"].get("chat_refs", []))]
            if bool(bodies) == bool(chats):
                continue
            # Existing content is not missing because its latest source is old.
            # A chat lookup itself checks hashes and may find an older valid link.
            if bodies:
                seeds = [item for item in bodies if any(
                    ref.get("target_id") == fid for ref in item["row"].get("chat_refs", []))]
            else:
                seeds = chats
            if not seeds:
                continue
            best = max(seeds, key=lambda item: (float(item["preliminary"]), -int(item["chunk_id"])))
            text = best["row"]["chunk_text"].lower()
            needed = tuple(term for term in terms if term.lower() not in text)
            if not needed or len(needed) == len(terms):
                continue
            requests.append(CompanionRequest(fid, "chat" if bodies else "content", needed))
        self._check_cancelled()
        report = recall_companions(self.db, requests, restricted=self._allowed_ids is not None)
        self._check_cancelled()
        if not report.chunk_targets:
            return report
        kinds = {request.file_id: request.source for request in requests}
        existing_ids = {item["chunk_id"] for item in items}
        fresh_ids = set(report.chunk_targets) - existing_ids
        rows = self.chat_store.enrich(self.db.fetch_chunks_by_ids(fresh_ids), search_query)
        for cid, row in rows.items():
            fid = report.chunk_targets[cid]
            if (row["file_id"] != fid or (row["source_type"] == "chat") != (kinds[fid] == "chat")
                    or not self._passes_filters(row, search_query, query)):
                continue
            # A supporting lookup has no global keyword/semantic rank. Do not
            # invent RRF or vector scores for a fragment outside those routes.
            items.append(self._score_row(cid, row, query, descriptive=descriptive))
        return report

    def _add_joint_evidence(self, items, query, *, descriptive):
        # Correct coverage, not hit counts: only one content/context pair from
        # the same explicitly linked file can supply complementary query terms.
        groups = {}
        terms = query.terms or [query.normalized]
        informative = {i for i, term in enumerate(terms) if term not in _JOINT_GENERIC_TERMS}
        for item in items:
            row = item["row"]
            if str(row["path"]).startswith(CHAT_PREFIX):
                continue
            if row["source_type"] == "chat":
                refs = row.get("chat_refs", [])
                if not refs or refs[0].get("target_id") != row["file_id"]:
                    continue
                ref = refs[0]
                if ref.get("linked_hash") and ref.get("target_hash") != ref["linked_hash"]:
                    continue
            text = identifier_content(row["chunk_text"], row["source_type"]) if query.identifier else row["chunk_text"]
            matched = frozenset(i for i, term in enumerate(terms) if term_matches(text, term, query))
            meaningful = matched & informative
            if row["source_type"] == "chat":
                # A repeated attachment filename is not an independent chat clue.
                meaningful = {i for i in meaningful if terms[i].lower() not in row["name"].lower()}
            body, chat = groups.setdefault(row["file_id"], ([], []))
            (chat if row["source_type"] == "chat" else body).append((item, matched, meaningful))
        chosen = {}
        for bodies, chats in groups.values():
            for body, body_terms, body_meaningful in bodies:
                for chat, chat_terms, chat_meaningful in chats:
                    if not body_meaningful - chat_terms or not chat_meaningful - body_terms:
                        continue
                    coverage = 0.68 * len(body_terms | chat_terms) / max(len(terms), 1)
                    if coverage <= max(float(body["lexical"]), float(chat["lexical"])):
                        continue
                    pair = (body, chat)
                    for item in pair:
                        cid = item["chunk_id"]
                        previous = chosen.get(cid)
                        key = (coverage, min(float(body["preliminary"]), float(chat["preliminary"])),
                               -int(body["chunk_id"]), -int(chat["chunk_id"]))
                        if previous is None or key > previous[0]:
                            chosen[cid] = (key, pair)
        for item in items:
            if item["chunk_id"] not in chosen:
                continue
            key, pair = chosen[item["chunk_id"]]
            item["lexical"] = key[0]
            item["joint_evidence"] = tuple({"chunk_id": part["chunk_id"], "row": part["row"]} for part in pair)
            item["preliminary"] = self._base_score(
                item["lexical"], item["semantic"], item["fused"], item["filename"], item["extension_bonus"],
                descriptive=descriptive,
            )

    def _base_score(self, lexical, semantic, fused, filename, extension, *, descriptive=False):
        if self.retrieval_mode == "semantic":
            return semantic
        if self.retrieval_mode == "keyword":
            return min(0.999, lexical * 0.42 + filename * 0.08 + fused * 0.18 + extension)
        if descriptive:
            return min(0.999, lexical * 0.22 + semantic * 0.60 + fused * 0.12 + filename * 0.06 + extension)
        return min(0.999, lexical * 0.42 + semantic * 0.32 + fused * 0.18 + filename * 0.08 + extension)

    def _keyword_candidates(self, query: ProcessedQuery, limit: int) -> dict[int, int]:
        self._check_cancelled()
        ordered_ids: list[int] = []
        fts_query = build_fts_query(query.terms)
        if fts_query:
            try:
                options = {"target_mode": self._target_mode, "restricted": self._allowed_ids is not None}
                if query.identifier:
                    exact_query = build_fts_query([query.identifier])
                    ordered_ids.extend(int(row["chunk_id"]) for row in self.db.search_fts(
                        exact_query, limit, identifier=query.identifier, **options))
                    self._check_cancelled()
                ordered_ids.extend(int(row["chunk_id"]) for row in self.db.search_fts(fts_query, limit, **options))
            except SearchCancelled:
                raise
            except Exception:
                pass
        self._check_cancelled()
        options = {"target_mode": self._target_mode, "restricted": self._allowed_ids is not None}
        like_rows = self.db.search_like_terms(query.terms or [query.normalized], limit=limit, **options)
        self._check_cancelled()
        ordered_ids.extend(int(row["chunk_id"]) for row in like_rows)
        return {chunk_id: rank for rank, chunk_id in enumerate(_dedupe(ordered_ids), start=1)}

    def _prepare_semantic(self, text, clauses=()):
        self._check_cancelled()
        try:
            clause_vectors = None
            if clauses:
                vectors = self.embedding_service.encode_queries([text, *semantic_clause_texts(clauses)])
                if len(vectors) != 3:
                    raise ValueError("Invalid query batch size")
                vector, clause_vectors = vectors[0], vectors[1:]
            else:
                vector = self.embedding_service.encode_query(text)
            self._check_cancelled()
            model = self.embedding_service.active_model_name
            status = f"降级模型：{model}" if self.embedding_service.load_error else f"已启用：{model}"
            return _PreparedSemantic(vector, model, status, clauses, clause_vectors)
        except SearchCancelled:
            raise
        except Exception as exc:
            return _PreparedSemantic(None, None, f"语义召回失败：{exc}")

    def _semantic_candidates(
        self,
        text: str,
        limit: int,
        prepared: _PreparedSemantic | None,
    ) -> tuple[dict[int, float], dict[int, int], str | None, str]:
        self._check_cancelled()
        if self._allowed_ids is not None and not self._allowed_ids:
            return {}, {}, None, "当前筛选范围没有语义候选"
        if not self.vector_index.ids:
            return {}, {}, None, "无语义向量"
        if not self.vector_index.has_vectors(self._allowed_ids):
            return {}, {}, None, "当前筛选范围没有语义向量"
        if prepared is None:
            return {}, {}, None, "语义查询尚未准备"
        if prepared.vector is None:
            return {}, {}, prepared.model, prepared.status
        try:
            query_vector, active_model = prepared.vector, prepared.model
            if active_model != self.indexed_embedding_model:
                return {}, {}, active_model, "查询模型与索引模型不一致，已停用语义召回"
            options = {"allowed_ids": self._allowed_ids} if self._allowed_ids is not None else {}
            if self._target_mode not in self._target_groups:
                self._target_groups[self._target_mode] = self.db.candidate_groups(target_mode=self._target_mode)
            options["target_groups"] = self._target_groups[self._target_mode]
            rows = self.vector_index.search(query_vector, limit=limit, **options)
            self._check_cancelled()
            scores = {chunk_id: max(0.0, min(float(score), 1.0)) for chunk_id, score in rows}
            if prepared.clause_vectors is not None:
                try:
                    extra_ids = set()
                    for vector, source in zip(prepared.clause_vectors, semantic_clause_sources(prepared.clauses)):
                        clause_options = dict(options)
                        if source != "all":
                            clause_options["allowed_ids"] = self._clause_source_ids(source)
                        extra_ids.update(cid for cid, _ in self.vector_index.search(vector, limit=limit, **clause_options))
                        self._check_cancelled()
                    additions = self.vector_index.score_ids(query_vector, extra_ids - scores.keys())
                    scores.update({cid: max(0.0, min(values[0], 1.0)) for cid, values in additions.items()})
                    rows = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
                except SearchCancelled:
                    raise
                except Exception as exc:
                    prepared.clause_vectors = None
                    prepared.clause_error = f"分句召回不可用，保留整句召回：{exc}"
            ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(rows, start=1)}
            return scores, ranks, active_model, prepared.status
        except SearchCancelled:
            raise
        except Exception as exc:
            return {}, {}, None, f"语义召回失败：{exc}"

    def _clause_source_ids(self, source):
        if not self._source_ids:
            groups = {"chat": set(), "content": set()}
            for number, row in enumerate(self.db.conn.execute("SELECT id,source_type FROM chunks")):
                if number % 1024 == 0:
                    self._check_cancelled()
                groups["chat" if row["source_type"] == "chat" else "content"].add(int(row["id"]))
            self._check_cancelled()
            self._source_ids = groups
        ids = self._source_ids[source]
        return ids if self._allowed_ids is None else ids & self._allowed_ids

    def _rerank(
        self,
        query: ProcessedQuery,
        candidates: list[dict[str, object]],
    ) -> dict[int, float]:
        self._check_cancelled()
        if not candidates:
            return {}
        passages = []
        for item in candidates:
            row = item["row"]
            parts = evidence_parts(item)
            if parts:
                sections = [f"文件名：{row['name']}"]
                for part in parts:
                    label = "该文件的聊天线索" if part["row"]["source_type"] == "chat" else "文件正文"
                    sections.append(f"{label}：{make_evidence_snippet(part['row']['chunk_text'], query.terms)}")
                passages.append("\n".join(sections))
                continue
            passages.append(
                f"文件名：{row['name']}\n路径：{row['path']}\n内容：{row['chunk_text']}"
            )
        scores = self.reranker_service.rerank(query.semantic_text, passages)
        self._check_cancelled()
        if scores is None or len(scores) != len(candidates):
            return {}
        return {
            int(item["chunk_id"]): max(0.0, min(float(score), 1.0))
            for item, score in zip(candidates, scores)
        }

    def _reranker_status(self, scores: dict[int, float]) -> str:
        if scores:
            runtime = getattr(getattr(self.reranker_service, "runtime", None), "name", "model")
            return f"已启用：{self.reranker_service.model_name} / {str(runtime).upper()}"
        if not self.reranker_service.enabled:
            return "已关闭"
        if self.reranker_service.load_error:
            return f"不可用：{self.reranker_service.load_error}"
        return "无候选结果"

    def _passes_filters(self, row: Row, query: SearchQuery, processed: ProcessedQuery) -> bool:
        if query.evidence == "content" and row["source_type"] == "chat":
            return False
        if query.evidence == "chat" and row["source_type"] != "chat":
            return False
        if query.file_type and row["ext"].lower() != query.file_type.lower().lstrip("."):
            return False
        if query.path_contains and query.path_contains.lower() not in row["path"].lower():
            return False
        modified_after = query.modified_after
        modified_before = query.modified_before
        moments = [row["modified_at"]]
        if row["source_type"] == "chat" or query.scope == "chat_files" or query.chat_platform or query.conversation or query.sender:
            moments = [ref["sent_at"] for ref in row.get("chat_refs", [])]
        if modified_after or modified_before:
            after = datetime.fromisoformat(modified_after.replace("Z", "+00:00")).timestamp() if modified_after else None
            before = datetime.fromisoformat(modified_before.replace("Z", "+00:00")).timestamp() if modified_before else None
            times = [datetime.fromisoformat(moment.replace("Z", "+00:00")).timestamp() for moment in moments if moment]
            if not any((after is None or moment >= after) and (before is None or moment < before) for moment in times):
                return False
        return True

    def _resolve_indexed_model(self) -> str | None:
        configured = self.db.get_setting("embedding_model")
        models = self.db.vector_models()
        if configured and configured in models:
            return configured
        if models:
            return max(models, key=models.get)
        return configured


def build_fts_query(terms: list[str]) -> str:
    eligible = []
    for term in terms:
        compact = term.strip()
        if len(compact) < 3 and not re.search(r"[a-zA-Z0-9]", compact):
            continue
        eligible.append(f'"{compact.replace(chr(34), chr(34) * 2)}"')
    return " OR ".join(eligible[:12])


def reciprocal_rank_fusion(keyword_rank: int | None, semantic_rank: int | None, *, k: int = 60) -> float:
    score = 0.0
    total_weight = 1.8
    if keyword_rank is not None:
        score += 1.0 * ((k + 1) / (k + keyword_rank))
    if semantic_rank is not None:
        score += 0.8 * ((k + 1) / (k + semantic_rank))
    return score / total_weight


def lexical_evidence(row: Row, query: ProcessedQuery) -> tuple[float, float, list[str]]:
    name = str(row["name"]).lower()
    path = str(row["path"]).lower()
    content = str(row["chunk_text"]).lower()
    if query.identifier:
        content = identifier_content(content, row["source_type"])
    terms = query.terms or [query.normalized]
    body_matches = [term for term in terms if term_matches(content, term, query)]
    matched = [term for term in terms if term_matches(content, term, query) or term_matches(name, term, query)]
    path_only = [term for term in terms if term not in matched and term_matches(path, term, query)]
    coverage = len(matched) / max(len(terms), 1)
    filename_matches = [term for term in terms if term_matches(name, term, query)]
    filename_coverage = len(filename_matches) / max(len(terms), 1)
    compact_query = "".join(query.normalized.split())
    compact_content = "".join(content.split())
    compact_name = "".join(name.split())

    lexical_score = coverage * 0.68 + len(path_only) / max(len(terms), 1) * 0.08
    reasons: list[str] = []
    if body_matches:
        reasons.append("正文关键词")
    if path_only:
        reasons.append("路径关键词")
    if compact_query and compact_query in compact_content and (not query.identifier or
                                                               identifier_position(compact_content, compact_query) >= 0):
        lexical_score += 0.30
        reasons.append("完整短语")
    if filename_matches:
        reasons.append("文件名关键词")
    filename_score = min(1.0, filename_coverage * 0.75)
    if compact_query and compact_query in compact_name and (not query.identifier or
                                                            identifier_position(compact_name, compact_query) >= 0):
        filename_score = 1.0
        lexical_score += 0.18
        reasons.append("文件名完整命中")
    return min(1.0, lexical_score), filename_score, reasons


def term_matches(text: str, term: str, query: ProcessedQuery) -> bool:
    return identifier_position(text, term) >= 0 if term == query.identifier else term.lower() in text.lower()


def make_snippet(text: str, query: ProcessedQuery, *, radius: int = 90) -> str:
    lower = text.lower()
    positions = [lower.find(term.lower()) for term in query.terms if lower.find(term.lower()) >= 0]
    exact = identifier_position(text, query.identifier)
    pos = exact if exact >= 0 else min(positions) if positions else 0
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ") + suffix


def make_preview_anchor(text: str, query: ProcessedQuery, *, source_type: str = "") -> str:
    """Literal selected body evidence, with no chat or decorated ellipses."""
    original = text
    if source_type == "xlsx":
        text = re.sub(r"(?m)^R\d+: ", "", text)
    if source_type in {"csv", "xlsx", "docx"}:
        # A full row can begin many columns before the relevant cell.
        text = text.replace(" | ", "\n")
    exact = identifier_position(text, query.identifier)
    positions = [text.lower().find(term.lower()) for term in query.terms if term.lower() in text.lower()]
    if exact >= 0 or positions:
        pos = exact if exact >= 0 else min(positions)
        start = max(text.rfind("\n", 0, pos) + 1, pos - 80)
        end = text.find("\n", pos)
        return text[start:min(end if end >= 0 else len(text), start + 240)].strip()
    if source_type in {"csv", "xlsx"}:
        return max(text.splitlines(), key=len, default="")[:240].strip()
    # Keep surrounding context for semantic-only hits, especially repeated boilerplate.
    return original[:MAX_ANCHOR_CHARS].strip()


def make_evidence_snippet(text: str, terms: list[str], *, radius: int = 60) -> str:
    """Keep an excerpt around every matched term, merging overlapping windows."""
    lower = text.lower()
    spans = sorted((max(0, pos - radius), min(len(text), pos + len(term) + radius))
                   for term in terms[:16] if term and (pos := lower.find(term.lower())) >= 0)
    if not spans:
        return text[:180].replace("\n", " ") + ("..." if len(text) > 180 else "")
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return ("..." if merged[0][0] else "") + " ... ".join(
        text[start:end].replace("\n", " ") for start, end in merged
    ) + ("..." if merged[-1][1] < len(text) else "")


def _dedupe(items):
    return list(dict.fromkeys(items))
