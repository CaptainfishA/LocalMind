from dataclasses import replace
import os
from pathlib import Path

import pytest

from localmind.models import SearchQuery
from localmind.clause_semantics import file_clause_evidence
from localmind.desktop.detail_text import evidence_text
from localmind.query_processing import process_query, semantic_clauses, semantic_clause_sources, semantic_clause_texts
from test_clause_semantics import corpus, item  # noqa: F401
from test_file_preview import preview_app, wait  # noqa: F401


@pytest.mark.parametrize("label", ["正文说明", "正文里描述", "文件内容介绍", "资料的正文中说明"])
def test_descriptive_verbs_have_explicit_body_source(label):
    clauses = semantic_clauses(process_query(label + "服务器房间断电后的供电保障；聊天里说杭州轮岗交接时用的材料"))
    assert len(clauses) == 2
    assert semantic_clause_sources(clauses) == ("content", "chat")
    assert semantic_clause_texts(clauses) == ("服务器房间断电后的供电保障", "杭州轮岗交接时用的材料")


@pytest.mark.parametrize("text", [
    "正文说明书排版的规范", "内容介绍信的字体规范", "内容描述符的命名规则", "正文说明文字的大小",
    "关于正文说明的写法", "《正文说明服务器故障》的标题", "请找含有正文描述字段的记录",
])
def test_labels_inside_nouns_quotes_or_ordinary_descriptions_are_not_stripped(text):
    assert semantic_clause_sources((text,)) == ("all",)
    assert semantic_clause_texts((text,)) == (text,)


@pytest.mark.parametrize("text,description", [
    ("聊天线索：杭州轮岗交接时用的材料", "杭州轮岗交接时用的材料"),
    ("请帮我找文件内容是服务器断电保障，供新人学习", "服务器断电保障，供新人学习"),
    ("群消息确认这份还未通过审批", "这份还未通过审批"),
    ("正文描述“内容是备用电池”的字段", "“内容是备用电池”的字段"),
    ("正文说明不是采购说明而是设备维护", "不是采购说明而是设备维护"),
    ("聊天里说John's交接表，包括资产编号", "John's交接表，包括资产编号"),
])
def test_encoding_removes_only_the_recognized_heading_once(text, description):
    assert semantic_clause_texts((text,)) == (description,)


@pytest.mark.parametrize("query", [
    "正文说明；聊天里说杭州轮岗交接时用的材料",
    "正文介绍；聊天里说杭州轮岗交接时用的材料",
    "正文描述；聊天里说杭州轮岗交接时用的材料",
    "正文说明不要机房停电的资料；聊天里说杭州轮岗交接时用的材料",
    "正文说明机房停电的资料；或者聊天里说杭州轮岗交接时用的材料",
])
def test_empty_or_nonconjunctive_descriptions_keep_existing_fallback(query):
    assert not semantic_clauses(process_query(query))


@pytest.mark.parametrize("verb", ["说明", "描述", "介绍"])
def test_body_requirement_cannot_be_satisfied_by_stronger_chat_only_evidence(verb):
    clauses = ("聊天里说杭州轮岗交接时使用", f"正文{verb}服务器房间断电后的供电保障")
    candidates = [item(1, 10, "chat"), item(2, 10), item(3, 11, "chat")]
    coverage = file_clause_evidence(candidates, {1: (.9, .99), 2: (.1, .6), 3: (1., 1.)}, clauses)
    assert coverage[10][0] == pytest.approx(2 * .9 * .6 / 1.5)
    assert [part["chunk_id"] for part in coverage[10][1]] == [1, 2]
    assert coverage[11] == (0, ())


def test_same_batched_inference_uses_clean_text_and_keeps_original_evidence(corpus, monkeypatch):
    _, service, embedding, index, fid, _, _, body, chat = corpus
    service.enable_clause_semantics = True
    text = "聊天线索：杭州轮岗交接时要用的材料；正文说明服务器房间停电后的供电保障"
    query = process_query(text)
    seen = []
    encode = embedding.encode_queries
    def record(texts):
        seen.append(tuple(texts))
        return encode(texts)
    monkeypatch.setattr(embedding, "encode_queries", record)
    results = service.search(SearchQuery(text, scope="chat_files"))
    assert seen == [(query.semantic_text, "杭州轮岗交接时要用的材料", "服务器房间停电后的供电保障")]
    assert embedding.calls == [(3, False)] and index.rebuilds == 1
    target = next(hit for hit in results if hit.file_id == fid)
    assert target == results[0]
    assert {part.chunk_id for part in target.supporting_evidence} == {body, chat}
    assert {part.query_clause for part in target.supporting_evidence} == set(semantic_clauses(query))
    assert "聊天 / 正文" in service.last_diagnostics.clause_status


@pytest.mark.parametrize("mode", ["keyword", "semantic"])
def test_nonhybrid_modes_do_not_encode_source_subqueries(corpus, mode):
    _, service, embedding, _, *_ = corpus
    query = SearchQuery("聊天里说杭州交接用的材料；正文说明机房断电后的保障", scope="chat_files")
    service.retrieval_mode = mode
    before = service.search(query)
    service.enable_clause_semantics = True
    embedding.calls.clear()
    after = service.search(replace(query))
    assert [(r.file_id, r.score) for r in after] == [(r.file_id, r.score) for r in before]
    assert all(count == 1 for count, _ in embedding.calls)


@pytest.mark.parametrize("expanded", [False, True])
def test_body_first_display_does_not_mutate_ranked_evidence_or_source_identity(corpus, expanded):
    _, service, _, _, *_ = corpus
    service.enable_clause_semantics = True
    hit = service.search(SearchQuery("聊天里说杭州轮岗交接时用的材料；正文说明服务器房间停电后的供电保障",
                                     scope="chat_files"))[0]
    original = hit.supporting_evidence
    assert original[0].source_type == "chat"
    text, _ = evidence_text(hit, expanded=expanded)
    assert text.startswith("文件正文") and text.index("文件正文") < text.index("聊天线索")
    assert hit.supporting_evidence is original and original[0].source_type == "chat"
    assert "轮岗组" in text and "机房备用电池的维护计划" in text


def test_new_source_wording_keeps_body_visible_and_preview_target_stable(preview_app, corpus):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from localmind.scanner import build_file_record
    from localmind.ui import MainWindow

    db, service, _, _, *_ = corpus
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    for name in ("target.txt", "decoy.txt"):
        record = build_file_record(db.db_path.parent / name)
        with db.conn:
            db.conn.execute("UPDATE files SET modified_at=? WHERE path=?", (record.modified_at, record.path))
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(db.db_path.parent / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    service.enable_clause_semantics = True
    query = "聊天里说杭州轮岗交接时用的材料；正文说明服务器房间停电后的供电保障"
    hits = service.search(SearchQuery(query, scope="chat_files"))
    window = MainWindow(db)
    window.search_input.setText(query)
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    window.show()
    try:
        window._search_finished(hits, service.last_diagnostics, {"total_ms": 30, "index_refresh_ms": 0})
        wait(preview_app, lambda: window.file_preview._payload is not None)
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            preview_app.processEvents()
            preview_app.processEvents()
            assert window.preview.firstVisibleBlock().text().startswith("文件正文"), (
                width, window.preview.toPlainText(), window.preview.textCursor().position(),
                window.preview.verticalScrollBar().value(), window.preview.viewport().height())
            cursor = window.preview.document().find("机房备用电池的维护计划")
            assert not cursor.isNull() and window.preview.viewport().rect().contains(window.preview.cursorRect(cursor).center())
            for widget in (window.preview, window.file_preview, window.file_actions[0]):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"body-first-{width}.png"))
        window.evidence_expand.click()
        assert "正文说明服务器房间" in window.preview.toPlainText()
        assert "聊天里说杭州轮岗" in window.preview.toPlainText()
        assert window.file_preview._target[0] == hits[0].path
        assert window.selected_result() is hits[0]
    finally:
        window.close()
        preview_app.processEvents()
