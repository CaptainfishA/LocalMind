import numpy as np
import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.vector_index import LocalVectorIndex
from localmind.models import SearchQuery
from test_weflow_v04 import db, index_file, ingest, searcher, CountingEmbedding


class ControlledEmbedding(CountingEmbedding):
    def encode_documents(self, texts):
        vectors = np.zeros((len(texts), 512), dtype="float32")
        for i, text in enumerate(texts):
            score = 0.0 if "无关正文" in text else 0.95 if "独立候选" in text else 1.0
            vectors[i, 0] = score
            vectors[i, 1] = np.sqrt(1.0 - score * score)
        return vectors

    def encode_query(self, text):
        result = np.zeros(512, dtype="float32")
        result[0] = 1
        return result


def repeated_sources(db, tmp_path):
    embed = ControlledEmbedding()
    first = index_file(db, tmp_path / "first.txt", "无关正文", embed)
    second = index_file(db, tmp_path / "second.txt", "无关正文", embed)
    messages = [ChatMessage(f"repeat-{i}", "周期维护巡检", attachments=[
        Attachment(first["name"], path=first["path"])]) for i in range(180)]
    messages.append(ChatMessage("independent", "周期维护巡检 独立候选 注意验收", attachments=[
        Attachment(second["name"], path=second["path"])]))
    ingest(db, messages, embedding=embed)
    return first, second, embed


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_repeated_linked_sources_do_not_hide_a_different_file(db, tmp_path, mode):
    first, second, embed = repeated_sources(db, tmp_path)
    service = searcher(db, embed, mode=mode)
    index = service.vector_index._faiss_index
    results = service.search(SearchQuery("周期维护巡检", scope="chat_files", evidence="chat", limit=5))
    assert {r.file_id for r in results} == {first["id"], second["id"]}
    assert service.last_diagnostics.keyword_candidates <= 16
    assert service.last_diagnostics.semantic_candidates <= 8
    assert service.last_diagnostics.total_candidates <= 24
    service.search(SearchQuery("周期维护巡检", scope="chat_files", evidence="chat", limit=5))
    assert service.vector_index._faiss_index is index


def test_message_scope_keeps_distinct_sends_of_the_same_file(db, tmp_path):
    _, _, embed = repeated_sources(db, tmp_path)
    service = searcher(db, embed, mode="keyword")
    results = service.search(SearchQuery("周期维护巡检", scope="messages", limit=10))
    assert len(results) == 10
    assert len({r.chat_message_id for r in results}) == 10


def test_fts_and_substrings_diversify_before_limit(db, tmp_path):
    first, second, _ = repeated_sources(db, tmp_path)
    groups = db.candidate_groups()
    for rows in (db.search_fts('"周期维护巡检"', 5, target_mode="files"),
                 db.search_like_terms(["周期维护巡检"], limit=5, target_mode="files")):
        assert len(rows) == 5
        assert {groups[r["chunk_id"]] for r in rows} == {f"file:{first['id']}", f"file:{second['id']}"}
    ids = ChatStore(db).scope_ids(SearchQuery("", scope="chat_files", sender="nobody"))
    db.set_search_scope(ids)
    assert not db.search_fts('"周期维护巡检"', 5, target_mode="files", restricted=True)


def test_unlinked_same_names_remain_separate_attachments(db):
    ingest(db, [ChatMessage(str(i), "周期维护巡检", attachments=[Attachment("unknown.txt")]) for i in range(25)])
    service = searcher(db, mode="keyword")
    results = service.search(SearchQuery("周期维护巡检", scope="chat_files", limit=10))
    assert len(results) == 10 and all(r.result_kind == "attachment_pending" for r in results)
    assert len({r.attachment_id for r in results}) == 10


def test_group_cache_reused_and_refreshed_after_manual_rebind(db, tmp_path, monkeypatch):
    first, second, embed = repeated_sources(db, tmp_path)
    service = searcher(db, embed, mode="semantic")
    real_groups = db.candidate_groups
    calls = []
    def groups(**kwargs):
        calls.append(kwargs["target_mode"])
        return real_groups(**kwargs)
    monkeypatch.setattr(db, "candidate_groups", groups)
    query = SearchQuery("周期维护巡检", scope="chat_files", evidence="chat", limit=5)
    service.search(query)
    service.search(query)
    assert calls == ["files"]
    service.search(SearchQuery("周期维护巡检", scope="messages", limit=10))
    service.search(query)
    assert calls == ["files", "messages"]
    aid = db.conn.execute("SELECT id FROM chat_attachments WHERE file_id=?", (second["id"],)).fetchone()[0]
    ChatStore(db).bind(aid, first["id"])
    service.refresh_vector_index()
    assert {r.file_id for r in service.search(query)} == {first["id"]}
    assert calls == ["files", "messages", "files"]


def test_long_message_chunks_share_one_candidate_target(db):
    ingest(db, [ChatMessage("long", "周期维护巡检。" * 800),
                ChatMessage("other", "周期维护巡检 独立记录")])
    rows = db.search_like_terms(["周期维护巡检"], limit=100, target_mode="messages")
    assert len(rows) <= 5
    service = searcher(db, mode="keyword")
    results = service.search(SearchQuery("周期维护巡检", scope="messages", limit=10))
    assert len(results) == 2


@pytest.mark.parametrize("backend", ["faiss", "numpy"])
@pytest.mark.parametrize("restricted", [False, True])
def test_diverse_vector_recall_matches_exhaustive_reference(backend, restricted):
    index = LocalVectorIndex()
    if backend == "numpy":
        index._faiss = None
    elif index._faiss is None:
        pytest.skip("FAISS unavailable")
    rng = np.random.default_rng(431)
    vectors = rng.normal(size=(700, 32)).astype("float32")
    vectors[:600] = 0
    vectors[:600, 0] = 1
    ids = list(range(700, 0, -1))
    groups = {cid: "repeated" if i < 600 else f"file:{i // 3}" for i, cid in enumerate(ids)}
    index.rebuild(ids, vectors)
    before = index._faiss_index
    query = np.zeros(32, dtype="float32")
    query[0] = 1
    allowed = set(ids[1::2]) if restricted else None
    scores = index._vectors @ query
    reference = sorted([(cid, float(scores[i])) for i, cid in enumerate(ids)
                        if allowed is None or cid in allowed], key=lambda row: (-row[1], row[0]))
    for limit in (1, 10, 1000):
        counts, expected = {}, []
        for cid, score in reference:
            group = groups[cid]
            if counts.get(group, 0) < 2:
                counts[group] = counts.get(group, 0) + 1
                expected.append((cid, score))
            if len(expected) == limit:
                break
        actual = index.search(query, limit=limit, allowed_ids=allowed, target_groups=groups, per_target=2)
        assert [cid for cid, score in actual] == [cid for cid, score in expected]
        assert np.allclose([s for _, s in actual], [s for _, s in expected], atol=1e-6)
    assert index._faiss_index is before
    assert not index.search(query, allowed_ids=set(), target_groups=groups)
