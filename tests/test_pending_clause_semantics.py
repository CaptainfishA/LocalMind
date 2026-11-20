import os
from pathlib import Path

import numpy as np
import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.clause_semantics import file_clause_evidence, coverage_ranks
from localmind.desktop.detail_text import evidence_text, fold_repeated_lines
from localmind.models import SearchQuery
from localmind.reranker import RerankerService
from localmind.searcher import SearchService
from test_search_snapshot import CountedIndex, FixedQuery, document
from test_single_source_clauses import ConditionEmbedding, QUERY, BODY_A, BODY_B
from test_file_preview import preview_app, wait  # noqa: F401


@pytest.fixture
def pending_corpus(tmp_path):
    db = LocalMindDB(tmp_path / "pending.db")
    store = ChatStore(db)
    filler = "归档时按材料清单核对页数，留存记录由指定人员负责管理。\n" * 50
    try:
        for key, body in (("target", BODY_A + "\n" + filler + BODY_B),
                          ("decoy", "普通家具采购明细。\n" + filler)):
            store.ingest(platform="wechat", account="synthetic", conversation=key + "资料组", source_key=key,
                messages=[ChatMessage("one", body, sender="同事", sent_at="2026-10-27T10:00:00+08:00",
                                      attachments=[Attachment("同名资料.pdf")])])
        ids, vectors, parts = [], [], {}
        rows = db.conn.execute("""SELECT u.chunk_id,c.chunk_text,s.source_key,u.attachment_id,u.message_id,u.owner_file_id
            FROM chat_units u JOIN chunks c ON c.id=u.chunk_id
            JOIN chat_messages m ON m.id=u.message_id JOIN chat_sources s ON s.id=m.source_id ORDER BY u.chunk_id""").fetchall()
        for row in rows:
            key, text = row["source_key"], row["chunk_text"]
            vector = ((.25, .96, .1, 0) if BODY_A in text else (.2, .1, .97, 0) if BODY_B in text
                      else (.85, .51, .05, 0) if key == "decoy" else (.05, .1, .1, .98))
            ids.append(row["chunk_id"])
            vectors.append(vector)
            parts.setdefault(key, []).append(dict(row))
        assert any(BODY_A in part["chunk_text"] for part in parts["target"])
        assert any(BODY_B in part["chunk_text"] for part in parts["target"])
        assert not any(BODY_A in part["chunk_text"] and BODY_B in part["chunk_text"] for part in parts["target"])
        db.insert_vectors(ids, np.array(vectors, dtype="float32"), model_name=FixedQuery.active_model_name)
        embedding, index = ConditionEmbedding(db), CountedIndex()
        service = SearchService(db, embedding_service=embedding, vector_index=index,
                                reranker_service=RerankerService(enabled=False))
        yield db, service, embedding, index, parts
    finally:
        db.close()


@pytest.mark.parametrize("evidence", ["all", "chat"])
def test_two_parts_of_one_unlocated_attachment_are_scored_together(pending_corpus, evidence):
    db, service, embedding, index, parts = pending_corpus
    target = parts["target"][0]
    hits = service.search(SearchQuery(QUERY, scope="chat_files", evidence=evidence, infer_dates=False))
    assert hits[0].attachment_id == target["attachment_id"]
    assert hits[0].result_kind == "attachment_pending"
    assert len(hits) == 2 and {hit.name for hit in hits} == {"同名资料.pdf"}
    proof = hits[0].supporting_evidence
    assert {part.chunk_id for part in proof} <= {row["chunk_id"] for row in parts["target"]}
    assert any(BODY_A in part.snippet for part in proof)
    assert any(BODY_B in part.snippet for part in proof)
    assert all(part.chat_source.message_id == target["message_id"] for part in proof)
    assert all(part.chat_source.attachment_id == target["attachment_id"] for part in proof)
    assert not hits[0].preview_anchor
    assert len(embedding.calls) == 1 and len(embedding.calls[0]) == 3
    assert index.rebuilds == 1
    assert service.search(SearchQuery(QUERY, scope="chat_files", evidence="content")) == []


def test_pending_evidence_never_borrows_a_same_named_attachment(pending_corpus):
    _, service, _, _, parts = pending_corpus
    hits = service.search(SearchQuery(QUERY, scope="chat_files", evidence="chat", conversation="decoy资料组"))
    assert len(hits) == 1 and hits[0].attachment_id == parts["decoy"][0]["attachment_id"]
    assert {part.chunk_id for part in hits[0].supporting_evidence} <= {row["chunk_id"] for row in parts["decoy"]}


@pytest.mark.parametrize("mode", ["keyword", "semantic"])
def test_pending_nonhybrid_modes_keep_original_results(pending_corpus, mode):
    _, service, _, _, _ = pending_corpus
    service.retrieval_mode = mode
    query = SearchQuery(QUERY, scope="chat_files")
    service.enable_clause_semantics = False
    before = [(hit.attachment_id, hit.chunk_id, hit.score) for hit in service.search(query)]
    service.enable_clause_semantics = True
    assert [(hit.attachment_id, hit.chunk_id, hit.score) for hit in service.search(query)] == before
    assert service.last_diagnostics.clause_count == 0


def test_later_binding_switches_identity_without_rebuilding_chat_vectors(pending_corpus, tmp_path):
    db, service, _, index, parts = pending_corpus
    query = SearchQuery(QUERY, scope="chat_files", evidence="chat")
    pending = service.search(query)[0]
    assert pending.attachment_id == parts["target"][0]["attachment_id"]
    fid, _, _ = document(db, tmp_path / "实际文件.txt", "合成正文", (1., 0., 0., 0.))
    service.search(query)
    rebuilt = index.rebuilds
    ChatStore(db).bind(pending.attachment_id, fid)
    linked = service.search(query)[0]
    assert linked.result_kind == "file" and linked.file_id == fid
    assert index.rebuilds == rebuilt
    assert linked.attachment_id == pending.attachment_id
    assert {part.chunk_id for part in linked.supporting_evidence} == {part.chunk_id for part in pending.supporting_evidence}
    db.delete_file_by_path(str(tmp_path / "实际文件.txt"))
    missing = service.search(query)[0]
    assert missing.result_kind == "attachment_pending" and missing.attachment_id == pending.attachment_id
    assert {part.chunk_id for part in missing.supporting_evidence} == {part.chunk_id for part in pending.supporting_evidence}


def pending_item(cid, owner, mid, aid):
    return {"chunk_id": cid, "row": {"file_id": owner, "path": f"localmind-chat://{cid}", "name": "同名资料.pdf",
        "source_type": "chat", "chat_refs": [{"chunk_id": cid, "owner_file_id": owner,
                                              "message_id": mid, "attachment_id": aid}]}}


@pytest.mark.parametrize("other_mid,other_aid", [(20, 200), (10, 101), (10, None)])
def test_other_message_sibling_attachment_and_plain_message_cannot_supply_a_condition(other_mid, other_aid):
    first, second = pending_item(1, 101, 10, 100), pending_item(2, 102, other_mid, other_aid)
    keys = {1: "chat:10:100", 2: f"chat:{other_mid}:{other_aid or 0}"}
    result = file_clause_evidence([first, second], {1: (.9, .1), 2: (.1, .9)},
                                 ("描述甲", "描述乙"), target_keys=keys)
    assert len(result) == 2
    assert all(value[0] == pytest.approx(.18) for value in result.values())
    assert all(len(value[1]) == 1 for value in result.values())


@pytest.mark.parametrize("field,value", [("chunk_id", 2), ("owner_file_id", 999), ("message_id", None),
                                         ("attachment_id", -1)])
def test_invalid_pending_identity_cannot_join_coverage(field, value):
    item = pending_item(1, 101, 10, 100)
    item["row"]["chat_refs"][0][field] = value
    assert file_clause_evidence([item], {1: (1., 1.)}, ("描述甲", "描述乙"), target_keys={1: "chat:10:100"}) == {}


def test_mismatched_recall_target_cannot_merge_unrelated_evidence():
    items = [pending_item(1, 101, 10, 100), pending_item(2, 102, 20, 200)]
    result = file_clause_evidence(items, {1: (.9, .1), 2: (.1, .9)}, ("描述甲", "描述乙"),
                                 target_keys={1: "chat:10:100", 2: "chat:10:100"})
    assert result["chat:10:100"][0] == pytest.approx(.18)
    assert [part["chunk_id"] for part in result["chat:10:100"][1]] == [1]


def test_window_duplicates_count_as_one_semantic_target():
    items = [pending_item(i, 1000 + i, 10, 100) for i in range(1, 51)] + [pending_item(51, 1051, 20, 200)]
    scores = {i: (.9, .9) if i < 51 else (.5, .5) for i in range(1, 52)}
    targets = {i: "chat:10:100" if i < 51 else "chat:20:200" for i in range(1, 52)}
    coverage = file_clause_evidence(items, scores, ("描述甲", "描述乙"), target_keys=targets)
    assert coverage_ranks(coverage) == {"chat:10:100": 1, "chat:20:200": 2}
    assert [part["chunk_id"] for part in coverage["chat:10:100"][1]] == [1]


def test_pending_chat_never_satisfies_explicit_body_requirement(pending_corpus):
    _, service, _, _, _ = pending_corpus
    hits = service.search(SearchQuery("聊天里说轮岗新人培训演示时会用的材料；正文说明断电以后机房怎样持续供电",
                                      scope="chat_files"))
    assert hits and all(hit.semantic_score == 0 and not hit.supporting_evidence for hit in hits)


@pytest.mark.parametrize("text", ["普通一行", "短句\n" * 8, "完整而且较长的资料背景说明需要保留。\n" * 2,
                                "完整而且较长的资料背景说明需要保留。\n另一段\n完整而且较长的资料背景说明需要保留。"])
def test_short_or_nonadjacent_repetition_is_not_folded(text):
    result, folded = fold_repeated_lines(text)
    assert not folded and result == "\n".join(text.splitlines())


def test_collapsed_repetition_is_labeled_and_full_evidence_can_be_recovered(pending_corpus):
    _, service, _, _, _ = pending_corpus
    hit = service.search(SearchQuery(QUERY, scope="chat_files", evidence="chat"))[0]
    before = tuple(part.snippet for part in hit.supporting_evidence)
    collapsed, more = evidence_text(hit)
    expanded, _ = evidence_text(hit, expanded=True)
    assert more and "相同内容共" in collapsed and "相同内容共" not in expanded
    assert len(collapsed) < len(expanded)
    assert BODY_A in collapsed and BODY_B in collapsed
    assert tuple(part.snippet for part in hit.supporting_evidence) == before


def test_binding_during_recall_preserves_pending_snapshot_then_refreshes(pending_corpus, tmp_path, monkeypatch):
    db, service, _, index, parts = pending_corpus
    fid, _, _ = document(db, tmp_path / "known.txt", "独立的合成正文", (1., 0., 0., 0.))
    aid = parts["target"][0]["attachment_id"]
    writer = LocalMindDB(db.db_path)
    recall = service._semantic_candidates

    def bind_after_recall(*args):
        result = recall(*args)
        ChatStore(writer).bind(aid, fid)
        return result

    try:
        monkeypatch.setattr(service, "_semantic_candidates", bind_after_recall)
        query = SearchQuery(QUERY, scope="chat_files", evidence="chat")
        before = service.search(query)[0]
        assert before.result_kind == "attachment_pending" and before.attachment_id == aid
        assert not service.last_diagnostics.index_current
        rebuilt = index.rebuilds
        monkeypatch.setattr(service, "_semantic_candidates", recall)
        after = service.search(query)[0]
        assert after.result_kind == "file" and after.file_id == fid
        assert after.attachment_id == aid and service.last_diagnostics.index_current
        assert index.rebuilds == rebuilt
    finally:
        writer.close()


def test_pending_detail_recovers_tail_evidence_and_returns_preview_space(preview_app, pending_corpus, tmp_path):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from localmind.ui import MainWindow

    db, service, _, _, parts = pending_corpus
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    query = SearchQuery(QUERY, scope="chat_files", evidence="chat")
    hits = service.search(query)
    window = MainWindow(db)
    window.search_input.setText(QUERY)
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    window.evidence_filter.setCurrentIndex(window.evidence_filter.findData("chat"))
    window.show()
    try:
        window._search_finished(hits, service.last_diagnostics, {"total_ms": 31, "index_refresh_ms": 0})
        assert window.selected_result().attachment_id == parts["target"][0]["attachment_id"]
        assert "待定位" in window.detail_meta.text()
        assert window.file_preview.isHidden() and window.file_preview._target is None
        assert window.file_actions[0].text() == "定位并打开"
        assert BODY_A in window.preview.toPlainText() and BODY_B in window.preview.toPlainText()
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            preview_app.processEvents()
            assert window.preview_tabs.height() >= window.detail_splitter.height() - 8
            for widget in (window.preview, window.file_actions[0], window.bind_result_button):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                window.preview.verticalScrollBar().setValue(0)
                assert window.grab().save(str(Path(output) / f"pending-detail-{width}.png"))
        fid, _, _ = document(db, tmp_path / "known.txt", "可预览的合成文件", (1., 0., 0., 0.))
        ChatStore(db).bind(hits[0].attachment_id, fid)
        linked = service.search(query)
        window._search_finished(linked, service.last_diagnostics, {"total_ms": 32, "index_refresh_ms": 0})
        wait(preview_app, lambda: window.file_preview._payload is not None)
        assert not window.file_preview.isHidden()
        assert window.file_actions[0].text() == "打开文件" and "待定位" not in window.detail_meta.text()
        window.detail_splitter.setSizes([250, 240])
        preview_app.processEvents()
        sizes = window.detail_splitter.sizes()
        window._clear_result_details()
        assert window.file_preview.isHidden()
        window._search_finished(linked, service.last_diagnostics, {"total_ms": 32, "index_refresh_ms": 0})
        preview_app.processEvents()
        assert window.detail_splitter.sizes() == sizes
    finally:
        window.close()
        preview_app.processEvents()
