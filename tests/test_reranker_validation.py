from types import SimpleNamespace

import numpy as np
import pytest

from localmind.reranker import RerankerService
from localmind.models import SearchQuery
from test_pending_clause_semantics import pending_corpus  # noqa: F401
from test_single_source_clauses import QUERY


def runtime_model(predictions, *, limit=12, legacy=False):
    calls = []

    def tokenize(queries, passages, **kwargs):
        assert kwargs == dict(truncation=False, padding=False, verbose=False)
        return {"input_ids": [[0] * (len(q) + len(p) + 3) for q, p in zip(queries, passages)]}

    def predict(pairs, **kwargs):
        calls.append(pairs)
        return predictions

    model = SimpleNamespace(tokenizer=tokenize, predict=predict,
                            **{"max_length" if legacy else "max_seq_length": limit})
    service = RerankerService(enabled=True)
    service._model = model
    service._runtime = SimpleNamespace(name="cpu")
    return service, calls


@pytest.mark.parametrize("legacy", [False, True])
def test_pair_limit_includes_query_and_special_tokens_and_recovers(legacy):
    service, calls = runtime_model(np.array([.8]), legacy=legacy)
    assert service.rerank("query", ["abcde"]) is None
    assert not calls and "12 token" in service.load_error
    assert service.rerank("query", ["abcd"]) == pytest.approx([.8])
    assert calls == [[("query", "abcd")]] and service.load_error is None


@pytest.mark.parametrize("bad", [np.array([.5]), np.ones((3,)), np.ones((2, 1, 1)),
                                 np.ones((2, 0)), np.array(.5),
                                 np.array([np.nan, .3]), np.array([.2, np.inf])])
def test_bad_predictions_return_no_partial_scores_and_report_reason(bad):
    service, calls = runtime_model(bad)
    assert service.rerank("q", ["one", "two"]) is None
    assert len(calls) == 1 and service.load_error
    service._model.predict = lambda *args, **kwargs: np.array([.2, .8])
    assert service.rerank("q", ["one", "two"]) == pytest.approx([.2, .8])
    assert service.load_error is None


@pytest.mark.parametrize("predictions,expected", [
    (np.array([.1, .9]), [.1, .9]),
    (np.array([[.1], [.9]]), [.1, .9]),
    (np.array([-1., 1.]), [1 / (1 + np.exp(1)), 1 / (1 + np.exp(-1))]),
    (np.array([[1., 2.], [2., 1.]]), [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))]),
])
def test_valid_prediction_mapping_preserves_order(predictions, expected):
    service, _ = runtime_model(predictions)
    assert service.rerank("q", ["one", "two"]) == pytest.approx(expected)


def test_one_oversized_candidate_falls_back_for_the_whole_ranking():
    service, calls = runtime_model(np.array([.2, .8]))
    assert service.rerank("q", ["ok", "x" * 100]) is None
    assert not calls


def test_legacy_unspecified_length_uses_tokenizer_limit():
    service, calls = runtime_model(np.array([.8]), legacy=True)
    service._model.max_length = None
    service._model.tokenizer.model_max_length = 12
    assert service.rerank("query", ["abcde"]) is None
    assert not calls and "12 token" in service.load_error
    assert service.rerank("query", ["abcd"]) == pytest.approx([.8])


def test_search_keeps_base_ranking_and_reports_real_failure_then_recovers(pending_corpus):
    _, search, *_ = pending_corpus
    query = SearchQuery(QUERY, scope="chat_files", evidence="chat", infer_dates=False)
    before = [(hit.attachment_id, hit.chunk_id, hit.score) for hit in search.search(query)]
    service, calls = runtime_model(np.array([.8]), limit=1)
    search.reranker_service = service
    after = search.search(query)
    assert [(hit.attachment_id, hit.chunk_id, hit.score) for hit in after] == before
    assert not calls and all(hit.reranker_score is None for hit in after)
    assert "token 上限" in search.last_diagnostics.reranker_status
    assert "无候选结果" not in search.last_diagnostics.reranker_status
    service._model.max_seq_length = 10000
    service._model.predict = lambda pairs, **kwargs: np.full(len(pairs), .8)
    recovered = search.search(query)
    assert all(hit.reranker_score == pytest.approx(.8) for hit in recovered)
    assert "已启用" in search.last_diagnostics.reranker_status and service.load_error is None
