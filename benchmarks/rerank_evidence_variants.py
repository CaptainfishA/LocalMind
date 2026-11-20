"""Experimental input policies only; none replaces the desktop search policy."""
from copy import deepcopy
from time import perf_counter

from localmind.clause_semantics import evidence_parts
from localmind.desktop.detail_text import fold_repeated_lines
from localmind.query_processing import semantic_clause_texts
from localmind.searcher import SearchService, make_evidence_snippet


class FullEvidenceSearch(SearchService):
    def _rerank(self, query, candidates):
        if not candidates:
            return {}
        passages = []
        for item in candidates:
            row, parts = item["row"], evidence_parts(item)
            if parts:
                sections = [f"文件名：{row['name']}"]
                semantic_ids = {part["chunk_id"] for part in item.get("semantic_evidence", ())}
                for part in parts:
                    label = "该文件的聊天线索" if part["row"]["source_type"] == "chat" else "文件正文"
                    text = part["row"]["chunk_text"]
                    if not (part["chunk_id"] in semantic_ids
                            or item.get("detail_evidence") and part["chunk_id"] == item["chunk_id"]):
                        text = make_evidence_snippet(text, query.terms)
                    else:
                        text = text.replace("\n", " ")
                    sections.append(f"{label}：{text}")
                passages.append("\n".join(sections))
            else:
                passages.append(f"文件名：{row['name']}\n路径：{row['path']}\n内容：{row['chunk_text']}")
        scores = self.reranker_service.rerank(query.semantic_text, passages)
        if scores is None or len(scores) != len(candidates):
            return {}
        return {int(item["chunk_id"]): max(0., min(float(score), 1.)) for item, score in zip(candidates, scores)}


class CompactSearch(FullEvidenceSearch):
    def _rerank(self, query, candidates):
        compact = deepcopy(candidates)
        for item in compact:
            for part in (*item.get("semantic_evidence", ()), *item.get("joint_evidence", ()),
                         *((item["detail_evidence"],) if item.get("detail_evidence") else ())):
                part["row"]["chunk_text"] = fold_repeated_lines(part["row"]["chunk_text"])[0]
        return super()._rerank(query, compact)


class ClauseSearch(SearchService):
    def _rerank(self, query, candidates):
        base = super()._rerank(query, candidates)
        pairs, owners = [], []
        for item in candidates:
            grouped = {}
            for part in item.get("semantic_evidence", ()):
                for clause in part.get("query_clause", "").splitlines():
                    label = "该文件的聊天线索" if part["row"]["source_type"] == "chat" else "文件正文"
                    grouped.setdefault(clause, []).append(label + "：" + part["row"]["chunk_text"])
            if len(grouped) == 2:
                for clause, passages in grouped.items():
                    pairs.append((semantic_clause_texts((clause,))[0],
                        f"文件名：{item['row']['name']}\n" + "\n".join(passages)))
                owners.append(item["chunk_id"])
        if pairs:
            reranker = self.reranker_service
            lengths = [len(ids) for ids in reranker._model.tokenizer(
                [q for q, p in pairs], [p for q, p in pairs], truncation=False, padding=False)["input_ids"]]
            if any(length > reranker._model.max_seq_length for length in lengths):
                raise ValueError("Experimental clause exceeds model context")
            started = perf_counter()
            scores = reranker._model.predict(pairs, batch_size=16 if reranker.runtime.name == "cuda" else 4,
                convert_to_numpy=True, show_progress_bar=False)
            reranker.inference_ms += (perf_counter() - started) * 1000
            reranker.lengths += lengths
            assert len(scores) == len(pairs) and all(0 <= s <= 1 for s in scores)
            for n, cid in enumerate(owners):
                a, b = map(float, scores[n * 2:n * 2 + 2])
                coverage = 2 * a * b / (a + b) if a + b else 0.
                base[cid] = (base[cid] + coverage) / 2
        return base
