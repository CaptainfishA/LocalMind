import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.models import SearchQuery
from localmind.query_processing import process_query
from localmind.searcher import make_evidence_snippet
from test_weflow_v04 import db, index_file, ingest, searcher


QUERY = "苏州审计组现场核验的采购订单和供应商发票对账资料"
BODY = "采购订单与供应商发票逐笔对账，金额差异超过五十元需要重新确认。"
CONTEXT = "这份已经给苏州审计组，准备现场核验时使用。"


def linked(db, tmp_path, key, body=BODY, context=CONTEXT, conversation="合成资料组", sent_at=None):
    target = index_file(db, tmp_path / f"{key}.txt", body)
    ingest(db, [ChatMessage(key, context, sent_at=sent_at, sender="模拟同事", attachments=[
        Attachment(target["name"], path=target["path"])])], source=key, conversation=conversation)
    return target


@pytest.mark.parametrize("scope", ["all", "chat_files"])
@pytest.mark.parametrize("mode", ["keyword", "hybrid"])
def test_joint_query_uses_content_and_context_of_the_same_file(db, tmp_path, scope, mode):
    linked(db, tmp_path, "decoy-body", context="成都培训课堂练习")
    linked(db, tmp_path, "decoy-chat", body="仓库盘点破损商品")
    target = linked(db, tmp_path, "target")
    service = searcher(db, mode=mode)
    result = service.search(SearchQuery(QUERY, scope=scope))[0]
    assert result.file_id == target["id"]
    assert result.supporting_evidence
    body, chat = result.supporting_evidence
    assert body.source_type != "chat" and chat.source_type == "chat"
    assert "采购订单" in body.snippet and "苏州" in chat.snippet
    assert "正文与聊天互补" in result.match_reason
    assert "苏州" in result.chat_context
    assert result.source_query.text == ""
    assert {body.chunk_id, chat.chunk_id} <= set(db.candidate_groups())


@pytest.mark.parametrize("evidence", ["content", "chat"])
def test_single_evidence_scope_does_not_combine(db, tmp_path, evidence):
    linked(db, tmp_path, "target")
    results = searcher(db, mode="keyword").search(SearchQuery(QUERY, scope="chat_files", evidence=evidence))
    assert results
    assert all(not r.supporting_evidence for r in results)
    assert all((r.source_type == "chat") == (evidence == "chat") for r in results)


def test_semantic_and_message_modes_do_not_combine(db, tmp_path):
    linked(db, tmp_path, "target")
    for mode, scope in (("semantic", "chat_files"), ("keyword", "messages"), ("keyword", "files")):
        results = searcher(db, mode=mode).search(SearchQuery(QUERY, scope=scope))
        assert results and not any(r.supporting_evidence for r in results)


def test_different_files_and_unconfirmed_same_names_never_combine(db, tmp_path):
    first = linked(db, tmp_path, "first", context="无关内容")
    linked(db, tmp_path, "second", body="无关内容")
    ingest(db, [ChatMessage("unlinked", CONTEXT, attachments=[Attachment(first["name"])])], source="unlinked")
    results = searcher(db, mode="keyword").search(SearchQuery(QUERY, scope="chat_files"))
    assert results and not any(r.supporting_evidence for r in results)
    assert any(r.result_kind == "attachment_pending" for r in results)


@pytest.mark.parametrize("query,context,name", [
    ("采购订单资料", "资料已经发给大家", "target"),
    ("采购订单材料", "材料已经发给大家", "target"),
    ("采购订单苏州", "请查收", "苏州"),
    ("采购订单同事", "请查收", "target"),
    ("采购订单", "采购订单", "target"),
])
def test_generic_duplicate_and_filename_only_context_is_not_complementary(db, tmp_path, query, context, name):
    linked(db, tmp_path, name, body="采购订单", context=context)
    results = searcher(db, mode="keyword").search(SearchQuery(query, scope="chat_files"))
    assert results and not any(r.supporting_evidence for r in results)


def test_pair_uses_the_matching_old_source_not_the_latest_reference(db, tmp_path):
    target = linked(db, tmp_path, "target", conversation="旧群", sent_at="2026-09-01T10:00:00")
    for i in range(5):
        ingest(db, [ChatMessage(str(i), "成都课堂材料", sent_at="2026-10-01T10:00:00", attachments=[
            Attachment(target["name"], path=target["path"])])], source=f"recent{i}", conversation="新群")
    service = searcher(db, mode="keyword")
    result = service.search(SearchQuery(QUERY, scope="chat_files"))[0]
    assert result.supporting_evidence and "旧群" in result.chat_context and "新群" not in result.chat_context
    restricted = service.search(SearchQuery(QUERY, scope="chat_files", modified_after="2026-10-01T00:00:00"))
    assert not any(r.supporting_evidence for r in restricted)
    assert not any(r.supporting_evidence for r in service.search(
        SearchQuery(QUERY, scope="chat_files", conversation="新群")))


def test_changed_content_hash_disables_joint_association(db, tmp_path):
    target = linked(db, tmp_path, "target")
    index_file(db, Path(target["path"]), BODY + "更新后的版本")
    results = searcher(db, mode="keyword").search(SearchQuery(QUERY, scope="chat_files"))
    assert results and not any(r.supporting_evidence for r in results)


def test_no_hit_count_bonus_and_no_phrase_across_source_boundary(db):
    service = searcher(db, mode="keyword")
    query = process_query("alpha beta")
    def item(cid, source, text):
        return {"chunk_id": cid, "row": {"file_id": 1, "path": "/file.txt", "name": "file.txt",
                "source_type": source, "chunk_text": text, "chat_refs": [{"target_id": 1}]},
                "lexical": .2, "semantic": .5, "filename": 0, "fused": .5,
                "extension_bonus": 0, "preliminary": .3}
    items = [item(1, "txt", "alpha"), item(2, "chat", "beta")]
    service._add_joint_evidence(items, query, descriptive=False)
    expected = items[0]["preliminary"]
    assert items[0]["lexical"] <= .68
    many = [item(1, "txt", "alpha")] + [item(i, "chat", "beta") for i in range(2, 200)]
    service._add_joint_evidence(many, query, descriptive=False)
    assert all(i["preliminary"] == expected for i in many)
    assert len(many[0]["joint_evidence"]) == 2
    assert all("joint_evidence" not in part for part in many[0]["joint_evidence"])

    separate = [item(1, "txt", "alpha"), item(2, "chat", "beta"), item(3, "chat", "gamma")]
    query = process_query("alpha beta gamma")
    service._add_joint_evidence(separate, query, descriptive=False)
    assert separate[0]["lexical"] == pytest.approx(.68 * 2 / len(query.terms))
    assert len(separate[0]["joint_evidence"]) == 2


def test_reranker_receives_both_sources_once_per_pair(db, tmp_path):
    target = linked(db, tmp_path, "target")
    service = searcher(db, mode="keyword")
    passages = []
    service.reranker_service = SimpleNamespace(
        enabled=True, recommended_top_k=10, load_error=None, model_name="fake",
        runtime=SimpleNamespace(name="fake"),
        rerank=lambda query, texts: (passages.extend(texts) or [.8] * len(texts)),
    )
    result = service.search(SearchQuery(QUERY, scope="chat_files"))[0]
    assert result.file_id == target["id"] and result.supporting_evidence
    assert result.reranker_score == .8
    assert len(passages) == 1
    assert "文件正文：" in passages[0] and "该文件的聊天线索：" in passages[0]
    assert "采购订单" in passages[0] and "苏州" in passages[0]


def test_evidence_excerpt_keeps_separated_terms_without_full_document():
    text = "保密前缀" + "无关" * 100 + "alpha" + "无关" * 500 + "beta" + "无关" * 100
    snippet = make_evidence_snippet(text, ["alpha", "beta"])
    assert "alpha" in snippet and "beta" in snippet and "保密前缀" not in snippet
    assert len(snippet) < 300
    assert "alpha beta" == make_evidence_snippet("alpha beta", ["alpha", "beta"])


def test_joint_evidence_preview_is_plain_text_and_preserves_target(db, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication
    from localmind.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    for key in ("privacy_accepted_v02", "weflow_bulk_privacy_accepted"):
        db.set_setting(key, "1")
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    db.set_setting("model_cache_dir", str(tmp_path / "cache"))
    target = linked(db, tmp_path, "sample", context=CONTEXT + "<b>普通文字</b>")
    service = searcher(db, mode="keyword")
    results = service.search(SearchQuery(QUERY, scope="chat_files"))
    window = MainWindow(db)
    window.show()
    try:
        window.search_input.setText(QUERY)
        window.retrieval_buttons["keyword"].setChecked(True)
        window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
        window._search_finished(results, service.last_diagnostics, {"total_ms": 12, "index_refresh_ms": 0})
        text = window.preview.toPlainText()
        assert "文件正文" in text and "聊天线索" in text
        assert "命中词：" not in text and "<b>普通文字</b>" in text
        highlighted = [s.cursor.selectedText() for s in window.preview.extraSelections()]
        assert "采购" in highlighted and "苏州" in highlighted
        assert len(highlighted) <= 128
        assert window.selected_result().path == target["path"]
        assert "合成资料组" in window.context_preview.toPlainText()
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            for widget in (window.preview, window.search_button):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"joint-evidence-{width}.png"))
            window.preview_tabs.setCurrentIndex(1)
            app.processEvents()
            widget = window.source_history_button
            corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
            assert corner.x() <= width and corner.y() <= height
            window.preview_tabs.setCurrentIndex(0)
        partial = replace(service.last_diagnostics, companion_status="互补检查未完成：达到时间预算",
                          companion_requested_files=40, companion_checked_files=32, companion_candidates=2)
        window._search_finished(results, partial, {"total_ms": 68, "index_refresh_ms": 0})
        assert "互补未完成" in window.search_status.text()
        assert "达到时间预算" in window.search_status.toolTip()
        assert "32/40" in window.search_status.toolTip()
        window.resize(980, 650)
        app.processEvents()
        assert window.width() == 980 and window.height() == 650
        if output:
            assert window.grab().save(str(Path(output) / "companion-partial-980.png"))
        window.preview_tabs.setCurrentIndex(1)
        assert window.selected_result().path == target["path"]
        single = service.search(SearchQuery("采购订单", scope="chat_files", evidence="content"))
        window._search_finished(single, service.last_diagnostics, {"total_ms": 10, "index_refresh_ms": 0})
        assert not window.preview.extraSelections()
    finally:
        window.close()
        app.processEvents()
