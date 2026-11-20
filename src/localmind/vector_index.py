from __future__ import annotations

import numpy as np

from .embeddings import normalize_vectors


class LocalVectorIndex:
    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self.ids: list[int] = []
        self._vectors = np.empty((0, dim or 0), dtype="float32")
        self._faiss = None
        self._faiss_index = None
        self._positions: dict[int, int] = {}
        try:
            import faiss

            self._faiss = faiss
        except Exception:
            self._faiss = None

    @property
    def backend_name(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy"

    def has_vectors(self, allowed_ids: set[int] | None = None) -> bool:
        return bool(self.ids) and (allowed_ids is None or not self._positions.keys().isdisjoint(allowed_ids))

    def score_ids(self, queries: np.ndarray, ids) -> dict[int, tuple[float, ...]]:
        selected = sorted({int(cid) for cid in ids if int(cid) in self._positions})
        if not selected:
            return {}
        query = normalize_vectors(queries)
        if query.shape[1] != self.dim or not np.all(np.isfinite(query)):
            raise ValueError("Invalid query vectors")
        positions = [self._positions[cid] for cid in selected]
        values = self._vectors[positions] @ query.T
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite vector scores")
        return {cid: tuple(float(value) for value in row) for cid, row in zip(selected, values)}

    def rebuild(self, ids: list[int], vectors: np.ndarray) -> None:
        self.ids = [int(item) for item in ids]
        self._positions = {cid: i for i, cid in enumerate(self.ids)}
        if len(vectors) == 0:
            self.dim = self.dim or 0
            self._vectors = np.empty((0, self.dim), dtype="float32")
            self._faiss_index = None
            return
        arr = normalize_vectors(vectors)
        self.dim = int(arr.shape[1])
        self._vectors = arr
        if self._faiss is not None:
            index = self._faiss.IndexFlatIP(self.dim)
            index.add(arr)
            self._faiss_index = index
        else:
            self._faiss_index = None

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids:
            return
        new_ids = self.ids + [int(item) for item in ids]
        if len(self._vectors):
            new_vectors = np.vstack([self._vectors, normalize_vectors(vectors)])
        else:
            new_vectors = normalize_vectors(vectors)
        self.rebuild(new_ids, new_vectors)

    def search(self, vector: np.ndarray, *, limit: int = 20, allowed_ids: set[int] | None = None,
               target_groups: dict[int, str] | None = None, per_target: int = 4) -> list[tuple[int, float]]:
        if limit <= 0 or not self.ids or len(self._vectors) == 0:
            return []
        if target_groups is not None:
            if type(per_target) is not int or per_target <= 0:
                raise ValueError("per_target must be positive")
            return self._search_diverse(vector, limit, allowed_ids, target_groups, per_target)
        query = normalize_vectors(vector).astype("float32")
        if allowed_ids is not None:
            positions = np.asarray([self._positions[cid] for cid in allowed_ids if cid in self._positions], dtype=np.int64)
            if not len(positions):
                return []
            scores = self._vectors[positions] @ query[0]
            count = min(limit, len(scores))
            top = np.argpartition(-scores, count - 1)[:count]
            top = top[np.argsort(-scores[top])]
            return [(self.ids[int(positions[i])], float(scores[i])) for i in top]
        if self._faiss_index is not None:
            scores, indexes = self._faiss_index.search(query, min(limit, len(self.ids)))
            return [
                (self.ids[int(idx)], float(score))
                for idx, score in zip(indexes[0], scores[0])
                if int(idx) >= 0
            ]
        scores = self._vectors @ query[0]
        top_indexes = np.argsort(-scores)[:limit]
        return [(self.ids[int(index)], float(scores[int(index)])) for index in top_indexes]

    @staticmethod
    def _select_targets(rows, groups, limit, per_target):
        counts, selected = {}, []
        for cid, score in rows:
            target = groups.get(cid, ("chunk", cid))
            if counts.get(target, 0) >= per_target:
                continue
            selected.append((cid, score))
            counts[target] = counts.get(target, 0) + 1
            if len(selected) == limit:
                break
        return selected

    def _search_diverse(self, vector, limit, allowed_ids, groups, per_target):
        if allowed_ids is not None or self._faiss_index is None:
            positions = np.asarray(
                [self._positions[cid] for cid in allowed_ids if cid in self._positions]
                if allowed_ids is not None else range(len(self.ids)), dtype=np.int64,
            )
            if not len(positions):
                return []
            query = normalize_vectors(vector)
            scores = self._vectors[positions] @ query[0]
            ids = np.asarray(self.ids, dtype=np.int64)[positions]
            order = np.lexsort((ids, -scores))
            rows = ((int(ids[i]), float(scores[i])) for i in order)
            return self._select_targets(rows, groups, limit, per_target)

        # Expand the ranked prefix until diverse candidates are complete. Do not
        # stop at an arbitrary raw-hit cap that can be filled by repeated sends.
        total = len(self.ids)
        count = min(total, max(limit * 2, 32))
        while True:
            rows = self.search(vector, limit=count)
            rows.sort(key=lambda row: (-row[1], row[0]))
            selected = self._select_targets(rows, groups, limit, per_target)
            if count == total:
                return selected
            if len(selected) == limit and selected[-1][1] > rows[-1][1]:
                return selected
            count = min(total, count * 4)
