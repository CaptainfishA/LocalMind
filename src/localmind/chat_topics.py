"""Conservative, query-local topic suggestions; never changes retrieval targets."""

from collections import OrderedDict
from dataclasses import replace
import hashlib
import re

import jieba
import numpy as np

from .chat_timeline import TimelineError, _validate, snapshot
from .embeddings import HashEmbeddingBackend, normalize_vectors

STOP = frozenset("收到 好的 谢谢 文件 附件 资料 文档 内容 报告 方案 项目 工作 会议 讨论 处理 确认 安排 进行 已经 这个 那个 我们 你们 大家 今天 明天 昨天 可以 需要 一下 相关 完成 测试 通知 上传 下载 更新 发送 合成 消息".split())


def features(text):
    identifiers = frozenset(re.findall(r"(?i)(?<![a-z0-9])[a-z]{1,10}[-_]?\d{2,12}(?:[-_]\d{1,8})?(?![a-z0-9])", text))
    terms = frozenset(word.lower() for word in jieba.lcut(text) if 2 <= len(word.strip()) <= 30
                      and word not in STOP and re.fullmatch(r"[\w\u4e00-\u9fff-]+", word) and not word.isdecimal())
    return terms, frozenset(x.lower() for x in identifiers)


class TopicOrganizer:
    def __init__(self):
        self.cache = OrderedDict()

    def clear(self):
        self.cache.clear()

    def organize(self, path, result, groups, encoder, *, excluded=(), cancelled=lambda: False):
        groups = tuple(groups[:200])
        with snapshot(path, cancelled) as conn:
            revision = conn.execute("SELECT revision FROM index_revisions WHERE id=1").fetchone()[0]
            candidates = []
            for group in groups:
                rows = [_validate(conn, result, ref) for ref in group.refs]
                row = next(r for r in rows if r["id"] == group.anchor_id)
                body = row["display_body"] if row["display_body"] is not None else row["body"]
                stamps = [r["sent_epoch"] for r in rows]
                eligible = (bool(stamps) and all(t is not None for t in stamps) and max(stamps) - min(stamps) <= 1800
                            and 12 <= len(body.strip()) <= 400 and not row["forwarded"])
                terms, ids = features(body) if eligible else (frozenset(), frozenset())
                candidates.append((body, terms, ids, min(stamps) if eligible else None,
                                   max(stamps) if eligible else None))
        if cancelled():
            raise TimelineError("整理已取消")
        # A local cache miss must not download a model or silently use hash vectors.
        previous = encoder.local_files_only
        encoder.local_files_only = True
        try:
            model = encoder.active_model_name
            if model == HashEmbeddingBackend.name:
                return groups, "本地语义模型不可用，保留引用 / 回复分组"
            vectors, pending, keys = {}, [], {}
            for i, (body, terms, ids, low, high) in enumerate(candidates):
                if low is None or not terms:
                    continue
                key = (model, groups[i].key[:3], hashlib.sha256(body.encode()).hexdigest())
                keys[i] = key
                if key in self.cache:
                    vectors[i] = self.cache[key]
                    self.cache.move_to_end(key)
                else:
                    pending.append(i)
            for offset in range(0, len(pending), 32):
                if cancelled():
                    raise TimelineError("整理已取消")
                indices = pending[offset:offset + 32]
                batch = np.asarray(encoder.encode_documents([candidates[i][0] for i in indices]), dtype="float32")
                if batch.ndim != 2 or len(batch) != len(indices) or not np.isfinite(batch).all():
                    raise TimelineError("主题向量无效，保留原讨论")
                batch = normalize_vectors(batch)
                for i, vector in zip(indices, batch):
                    vectors[i] = vector
                    self.cache[keys[i]] = vector
                    while len(self.cache) > 512:
                        self.cache.popitem(last=False)
        finally:
            encoder.local_files_only = previous
        if cancelled():
            raise TimelineError("整理已取消")
        with snapshot(path, cancelled) as conn:
            if conn.execute("SELECT revision FROM index_revisions WHERE id=1").fetchone()[0] != revision:
                self.clear()
                raise TimelineError("索引已更新，请刷新后重新整理")
        clusters = []
        comparisons = 0
        for i, group in enumerate(groups):
            if cancelled():
                raise TimelineError("整理已取消")
            body, terms, ids, low, high = candidates[i]
            target = None
            if i in vectors and group.key not in excluded and comparisons < 2000:
                # Keep all pair checks anchored and bounded; no weak-edge transitive closure.
                nearby = [c for c in clusters if len(c) < 12 and groups[c[0]].key[:3] == group.key[:3]
                          and c[0] in vectors and groups[c[0]].key not in excluded
                          and low is not None and max(high, *(candidates[j][4] for j in c))
                          - min(low, *(candidates[j][3] for j in c)) <= 1800][-8:]
                for cluster in nearby:
                    compatible = True
                    for j in cluster:
                        if comparisons >= 2000:
                            compatible = False
                            break
                        comparisons += 1
                        _, other_terms, other_ids, _, _ = candidates[j]
                        shared = terms & other_terms
                        lexical = bool(ids & other_ids) or (len(shared) >= 2 and len(shared) / max(1, min(len(terms), len(other_terms))) >= .35)
                        if (ids and other_ids and not ids & other_ids or not lexical
                                or float(vectors[i] @ vectors[j]) < (.86 if j == cluster[0] else .82)):
                            compatible = False
                            break
                    if compatible:
                        target = cluster
                        break
            if target is None:
                clusters.append([i])
            else:
                target.append(i)
        output = []
        for cluster in clusters:
            head = groups[cluster[0]]
            if len(cluster) == 1:
                output.append(head)
                continue
            parts = tuple(groups[i] for i in cluster)
            signature = hashlib.sha256(repr(tuple(g.key for g in parts)).encode()).hexdigest()[:20]
            output.append(replace(head, key=(*head.key[:3], "topic:" + signature), components=parts,
                refs=tuple({r.identity: r for g in parts for r in g.refs}.values()),
                hit_ids=tuple(dict.fromkeys(mid for g in parts for mid in g.hit_ids)),
                reason="可能同主题 · 语义与关键词一致"))
        return tuple(output), f"主题整理：{len(groups)} 段 → {len(output)} 段；可拆分，不改变文件排名"
