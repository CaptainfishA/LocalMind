from dataclasses import replace

import numpy as np
import pytest

from localmind.chat_timeline import TimelineError, discussions, window
from localmind.chat_topics import TopicOrganizer
from localmind.embeddings import HashEmbeddingBackend
from test_chat_timeline import add, corpus, message, result


class Encoder:
    local_files_only = False
    active_model_name = "synthetic-topic-test"

    def __init__(self):
        self.calls = []

    def encode_documents(self, texts):
        assert self.local_files_only
        self.calls.append(list(texts))
        return np.ones((len(texts), 8), dtype="float32")


def data(db, texts):
    add(db, [message(i, body=text) for i, text in enumerate(texts)])
    hits = [result(db, i) for i in range(len(texts))]
    groups = discussions(db.db_path, hits[0], hits)[0]
    return hits, groups


def test_topic_merge_requires_content_and_preserves_roots_ranking_and_cache(corpus):
    hits, groups = data(corpus, ["合同 AB123 的验收付款节点已经明确，需要财务审核", "合同 AB123 的验收付款节点需要财务补充审核意见", "收到"])
    organizer, encoder = TopicOrganizer(), Encoder()
    merged, notice = organizer.organize(corpus.db_path, hits[0], groups, encoder)
    assert len(merged) == 2 and len(merged[0].components) == 2
    assert merged[-1] == groups[-1] and "可能同主题" in merged[0].reason
    assert len(window(corpus.db_path, hits[0], merged[0]).rows) == 2
    assert [r.score for r in hits] == [.8, .8, .8]
    assert not encoder.local_files_only
    calls = len(encoder.calls)
    assert organizer.organize(corpus.db_path, hits[0], groups, encoder)[0] == merged
    assert len(encoder.calls) == calls
    restored, _ = organizer.organize(corpus.db_path, hits[0], groups, encoder, excluded=(groups[1].key,))
    assert restored == groups


@pytest.mark.parametrize("texts", [
    ["合同 AB123 的验收付款节点已经明确，需要财务审核", "合同 AB456 的验收付款节点需要财务补充审核意见"],
    ["今天会议讨论这个项目相关方案资料已经完成", "明天会议讨论这个项目相关方案资料已经完成"],
    ["合同 AB123 的验收付款节点已经明确，需要财务审核", "大家周末团建旅游的火车出发时间需要确认"],
])
def test_semantic_similarity_alone_never_proves_same_event(corpus, texts):
    hits, groups = data(corpus, texts)
    assert TopicOrganizer().organize(corpus.db_path, hits[0], groups, Encoder())[0] == groups


def test_hash_backend_does_not_claim_semantic_grouping(corpus):
    hits, groups = data(corpus, ["合同 AB123 的验收付款节点需要财务审核"] * 2)
    encoder = Encoder()
    encoder.active_model_name = HashEmbeddingBackend.name
    output, notice = TopicOrganizer().organize(corpus.db_path, hits[0], groups, encoder)
    assert output == groups and "不可用" in notice and not encoder.calls


def test_topic_spans_and_accounts_are_never_crossed(corpus):
    text = "合同 AB123 的验收付款节点需要财务审核"
    add(corpus, [message(1, body=text), message(35, body=text)])
    add(corpus, [message(2, body=text)], key="other", account="other")
    hits = [result(corpus, 1), result(corpus, 35), result(corpus, 2, "other")]
    groups = discussions(corpus.db_path, hits[0], hits)[0]
    assert TopicOrganizer().organize(corpus.db_path, hits[0], groups, Encoder())[0] == groups


def test_cancel_and_nonfinite_vectors_fail_without_changing_local_setting(corpus):
    hits, groups = data(corpus, ["合同 AB123 的验收付款节点需要财务审核"] * 2)
    encoder = Encoder()
    with pytest.raises(TimelineError):
        TopicOrganizer().organize(corpus.db_path, hits[0], groups, encoder, cancelled=lambda: True)
    encoder.encode_documents = lambda texts: np.full((len(texts), 8), np.nan)
    with pytest.raises(TimelineError):
        TopicOrganizer().organize(corpus.db_path, hits[0], groups, encoder)
    assert not encoder.local_files_only


def test_changed_revision_during_encoding_rejects_stale_suggestions(corpus):
    hits, groups = data(corpus, ["合同 AB123 的验收付款节点需要财务审核"] * 2)
    encoder = Encoder()
    encode = encoder.encode_documents

    def changed(texts):
        with corpus.conn:
            corpus.conn.execute("UPDATE index_revisions SET revision=revision+1 WHERE id=1")
        return encode(texts)

    encoder.encode_documents = changed
    organizer = TopicOrganizer()
    with pytest.raises(TimelineError, match="索引已更新"):
        organizer.organize(corpus.db_path, hits[0], groups, encoder)
    assert not organizer.cache


def test_search_worker_reuses_model_and_always_reports_cancel_completion(corpus):
    from threading import Event
    from localmind.ui import SearchWorker
    hits, groups = data(corpus, ["合同 AB123 的验收付款节点需要财务审核"] * 2)
    worker = SearchWorker(str(corpus.db_path), "unused", "auto")
    encoder = worker.embedding_service = Encoder()
    output = []
    worker.topics_finished.connect(lambda *args: output.append(args))
    request = {"cancel": Event(), "result": hits[0], "groups": groups, "excluded": ()}
    worker.organize_topics(request)
    assert worker.embedding_service is encoder and len(output) == 1 and encoder.calls
    request["cancel"].set()
    worker.organize_topics(request)
    assert len(output) == 2 and output[-1][1] is None
    worker.mark_index_dirty()
    assert not worker.topic_organizer.cache
    worker.close()
