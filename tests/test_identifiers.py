import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.db import LocalMindDB
from localmind.identifiers import identifier_position, indexed_identifier
from localmind.models import SearchQuery, TextChunk
from localmind.query_processing import process_query
from localmind.searcher import lexical_evidence
from test_weflow_v04 import index_file, searcher


@pytest.mark.parametrize("text,code", [
    ("访客登记 FK-864", "fk-864"), ("编号SKU120的来访资料", "sku120"),
    ("SOP_321.pdf", "sop_321"), ("采购-FK-864_说明.pdf", "fk-864"),
    ("ＦＫ－８６４", "fk-864"), ("FK-864 和 fk-864", "fk-864"),
    ("访客登记", ""), ("2026-10-26", ""), ("12345678", ""), ("V1.2", ""),
    ("2026-10-26T08:30:00.000Z", ""),
    ("FK-864 或 FK-865", ""), ("FK-864 和 FK-865", ""), ("不是FK-864", ""),
    ("without FK-864", ""), ("x" * 65 + "864", ""),
])
def test_single_identifier_preserves_atomic_term_and_avoids_boolean_inference(text, code):
    query = process_query(text)
    assert query.identifier == code
    if code:
        assert query.terms[0] == code
        assert "fk" not in query.terms and "864" not in query.terms


@pytest.mark.parametrize("text,matched", [
    ("FK-864", True), ("编号fk-864。", True), ("FK-864.pdf", True),
    ("采购-FK-864_说明.pdf", True), ("(FK-864)", True),
    ("FK-8640", False), ("AFK-864", False), ("A-FK-864", False),
    ("FK-864-02", False), ("FK-864_backup", False), ("FK-864.1", False),
    ("FK-864A", False), ("FK864", False),
])
def test_identifier_boundary(text, matched):
    assert (identifier_position(text, "fk-864") >= 0) is matched


def test_indexed_chat_header_is_not_identifier_evidence():
    header = "平台: wechat\n会话: FK-864\n发送者: FK-864\n消息时间: 未知\n"
    assert not indexed_identifier(header + "消息: 普通登记资料", "chat", "fk-864")
    assert indexed_identifier(header + "附件: FK-864.txt\n消息: 请核对", "chat", "fk-864")
    assert indexed_identifier(header + "消息: 编号FK-864。", "chat", "fk-864")
    assert indexed_identifier("会话: FK-864", "txt", "fk-864")


def test_wrong_identifier_has_no_partial_term_or_phrase_bonus():
    query = process_query("FK-864")
    row = {"name": "FK-8640.txt", "path": "D:/FK-8640.txt", "chunk_text": "编号 FK-8640", "source_type": "txt"}
    score, name_score, reasons = lexical_evidence(row, query)
    assert score == name_score == 0 and not reasons
    assert lexical_evidence({**row, "chunk_text": "编号 FK-864"}, query)[0] > 0


def test_complete_code_elsewhere_does_not_validate_a_prefix_phrase():
    query = process_query("访客登记 FK-864")
    row = {"name": "资料.txt", "path": "D:/资料.txt", "source_type": "txt",
           "chunk_text": "访客登记 FK-8640。另一份编号 FK-864。"}
    assert "完整短语" not in lexical_evidence(row, query)[2]
    assert "完整短语" in lexical_evidence({**row, "chunk_text": "访客登记 FK-864。"}, query)[2]


def test_exact_recall_is_scoped_and_runs_before_candidate_limit(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        for i in range(110):
            index_file(db, tmp_path / f"FK-864{i:03}.txt", f"访客登记 FK-864{i:03}，登记访客来访事宜。")
        target = index_file(db, tmp_path / "correct.txt", "登记表唯一编号 FK-864")
        service = searcher(db, mode="keyword")
        service.embedding_service.encode_query = Mock(side_effect=AssertionError("keyword must not load model"))
        hits = service.search(SearchQuery("访客登记 FK-864", limit=1))
        assert hits[0].file_id == target["id"] and hits[0].identifier_matched
        assert service.last_diagnostics.identifier_matches == 1
        cid = hits[0].chunk_id
        assert [r["chunk_id"] for r in db.search_fts('"FK-864"', 1, identifier="fk-864", target_mode="files")] == [cid]
        db.set_search_scope(set())
        assert db.search_fts('"FK-864"', 1, identifier="fk-864", restricted=True, target_mode="files") == []
        hits = service.search(SearchQuery("访客登记 FK-864", path_contains="FK-8640"))
        assert hits and all(not r.identifier_matched for r in hits)
        assert all(r.file_id != target["id"] for r in hits)
        assert service.last_diagnostics.identifier_matches == 0
    finally:
        db.close()


@pytest.mark.parametrize("evidence,exact", [("all", True), ("chat", True), ("content", False)])
def test_wechat_chat_evidence_never_leaks_into_body_only_filter(tmp_path, evidence, exact):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        path = tmp_path / "linked.txt"
        target = index_file(db, path, "访客登记办理说明")
        ChatStore(db).ingest(platform="wechat", account="synthetic", source_key="group",
                             conversation="交接组", messages=[ChatMessage("one", "来访登记表的编号 FK-864",
                             attachments=[Attachment(path.name, path=str(path))])])
        service = searcher(db, mode="keyword")
        hits = service.search(SearchQuery("访客登记 FK-864", scope="chat_files", evidence=evidence))
        assert hits[0].file_id == target["id"]
        assert hits[0].identifier_matched is exact
        if exact:
            assert hits[0].source_type == "chat" and "FK-864" in hits[0].snippet
            assert hits[0].chat_source.conversation == "交接组"
        assert not service.search(SearchQuery("访客登记 FK-864", scope="chat_files", conversation="别的群"))
    finally:
        db.close()


def test_pending_attachment_identifier_and_messages_keep_their_identity(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        ChatStore(db).ingest(platform="wechat", account="synthetic", source_key="group", conversation="登记组",
                             messages=[ChatMessage("a", "访客登记", attachments=[Attachment("FK-864.txt")]),
                                       ChatMessage("b", "登记资料", attachments=[Attachment("FK-8640.txt")])])
        service = searcher(db, mode="keyword")
        for scope in ("chat_files", "messages"):
            hits = service.search(SearchQuery("访客登记 FK-864", scope=scope))
            assert hits[0].identifier_matched and hits[0].name == "FK-864.txt"
            assert hits[0].chat_source.external_id == "a"
            assert hits[0].result_kind == ("message" if scope == "messages" else "attachment_pending")
    finally:
        db.close()


def test_identifier_priority_does_not_change_pure_semantic_ranking(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        a = index_file(db, tmp_path / "correct.txt", "FK-864")
        b = index_file(db, tmp_path / "other.txt", "FK-8640")
        ids = {r["file_id"]: r["id"] for r in db.conn.execute("SELECT id,file_id FROM chunks")}
        service = searcher(db, mode="semantic")
        scores = {ids[b["id"]]: .9, ids[a["id"]]: .5}
        service._semantic_candidates = lambda *args: (scores, {cid: i for i, cid in enumerate(scores, 1)}, "mock", "mock")
        hits = service.search("FK-864")
        assert [r.file_id for r in hits] == [b["id"], a["id"]]
        assert [r.score for r in hits] == [.9, .5]
        assert not hits[0].identifier_matched and hits[1].identifier_matched
        service.retrieval_mode = "hybrid"
        assert service.search("FK-864")[0].file_id == a["id"]
    finally:
        db.close()


def test_preferred_chunk_contains_the_code_and_does_not_claim_missing_file(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        doc = index_file(db, tmp_path / "target.txt", "访客登记办理说明")
        db.insert_chunks(doc["id"], [TextChunk(chunk_text="登记说明。" * 50 + "完整编号 FK-864。", source_type="txt")])
        service = searcher(db, mode="keyword")
        hits = service.search("访客登记 FK-864")
        assert hits[0].identifier_matched and "FK-864" in hits[0].snippet
        hits = service.search("访客登记 FK-999")
        assert hits and all(not hit.identifier_matched for hit in hits)
        assert all(hit.match_reason.startswith("相似候选") for hit in hits)
        assert service.last_diagnostics.identifier_matches == 0
    finally:
        db.close()


def test_reranker_cannot_displace_complete_code_with_partial_code(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    try:
        target = index_file(db, tmp_path / "target.txt", "FK-864")
        index_file(db, tmp_path / "other.txt", "FK-8640")
        service = searcher(db)
        service.reranker_service = SimpleNamespace(
            enabled=True, recommended_top_k=30, model_name="synthetic", load_error=None,
            runtime=SimpleNamespace(name="cpu"),
            rerank=lambda query, passages: [.99 if "FK-8640" in p else 0 for p in passages])
        hits = service.search("FK-864")
        assert hits[0].file_id == target["id"] and hits[0].identifier_matched
        assert hits[0].reranker_score == 0
        assert hits[0].score < hits[1].score
    finally:
        db.close()


def test_cancel_after_exact_recall_preserves_last_completed_query(tmp_path, monkeypatch):
    from localmind.searcher import SearchCancelled
    db = LocalMindDB(tmp_path / "index.db")
    try:
        index_file(db, tmp_path / "target.txt", "FK-864 登记表")
        service = searcher(db, mode="keyword")
        service.search("登记表")
        previous = service.last_diagnostics
        stop, calls = Event(), []
        service.should_cancel = stop.is_set
        original = db.search_fts
        def cancelling(*args, **kwargs):
            calls.append(kwargs.get("identifier"))
            rows = original(*args, **kwargs)
            stop.set()
            return rows
        monkeypatch.setattr(db, "search_fts", cancelling)
        with pytest.raises(SearchCancelled):
            service.search("FK-864")
        assert calls == ["fk-864"] and service.last_diagnostics is previous
        assert not db.conn.in_transaction
    finally:
        db.close()


def test_identifier_results_are_readable_and_reset_with_the_next_query(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication
    from localmind.ui import MainWindow
    from test_result_lifecycle import wait_for

    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db = LocalMindDB(tmp_path / "index.db")
    index_file(db, tmp_path / "访客登记.txt", "访客登记，编号 FK-864；另一份是 FK-8640。")
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("search_retrieval_mode", "keyword")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    window = MainWindow(db)
    service = searcher(db, mode="keyword")
    window.show()
    try:
        for text, found in (("访客登记 FK-864", True), ("访客登记 FK-999", False),
                            ("访客登记 " + "A" * 60 + "1234", False), ("访客登记", None)):
            window.search_input.setText(text)
            results = service.search(text)
            window._search_finished(results, service.last_diagnostics, {"total_ms": 12, "index_refresh_ms": 0})
            if found is not None:
                assert ("编号完整命中 1 项" if found else "候选中未见完整编号") in window.search_status.text()
                assert ("编号完整命中" if found else "未见完整编号") in window.identifier_notice.text()
                assert not window.identifier_notice.isHidden()
                assert results[0].identifier_query in window.identifier_notice.toolTip()
                if found:
                    highlights = [s.cursor.selectedText() for s in window.preview.extraSelections()
                                  if s.format.background().color().name() == "#fff2a8"]
                    assert sum(term.lower() == "fk-864" for term in highlights) == 1
            else:
                assert "编号" not in window.search_status.text() and "编号" not in window.detail_meta.text()
                assert window.identifier_notice.isHidden()
            wait_for(app, lambda: window.file_preview._payload is not None)
            for width, height in ((980, 650), (1240, 800)):
                window.resize(width, height)
                app.processEvents()
                for widget in (window.identifier_notice, window.detail_meta, window.search_status,
                               window.file_preview, window.file_actions[0]):
                    if widget.isHidden():
                        continue
                    corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                    assert corner.x() <= width and corner.y() <= height
                output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
                if output and found is not None:
                    Path(output).mkdir(parents=True, exist_ok=True)
                    assert window.grab().save(str(Path(output) / f"identifier-{found}-{len(text)}-{width}.png"))
        window._clear_result_details()
        assert window.identifier_notice.isHidden() and not window.identifier_notice.text()
    finally:
        window.close()
        app.processEvents()
