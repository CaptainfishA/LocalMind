import os
from pathlib import Path

import pytest

from localmind.models import SearchQuery
from localmind.query_processing import process_query, semantic_clauses, semantic_clause_sources
from test_clause_semantics import corpus  # noqa: F401


CHAT = "聊天里说杭州轮岗交接时要用的材料"
CONTENT = "文件内容是服务器房间停电后的供电保障"


@pytest.mark.parametrize("opening", ["帮我找一下", "请帮我搜索", "我想查找", "找一下", "麻烦你帮我找"])
def test_request_only_prefix_does_not_become_a_third_requirement(opening):
    assert semantic_clauses(process_query(f"{opening}，{CHAT}，{CONTENT}")) == (CHAT, CONTENT)


@pytest.mark.parametrize("prefix", ["", "帮我找一下", "请帮我搜索", "我想查找"])
def test_attached_request_keeps_explicit_source(prefix):
    clauses = semantic_clauses(process_query(f"{prefix}{CHAT}；{CONTENT}"))
    assert clauses == (prefix + CHAT, CONTENT)
    assert semantic_clause_sources(clauses) == ("chat", "content")


@pytest.mark.parametrize("chat_label,content_label", [
    ("聊天线索：", "文件内容："), ("群消息:", "正文:"), ("聊天记录：", "资料的正文："),
])
def test_explicit_colon_labels_route_to_their_sources(chat_label, content_label):
    clauses = semantic_clauses(process_query(f"{chat_label}杭州轮岗交接要用；{content_label}机房停电的供电保障"))
    assert len(clauses) == 2
    assert semantic_clause_sources(clauses) == ("chat", "content")


def test_inner_punctuation_preserves_all_text_within_explicit_source_groups():
    text = f"帮我找一下，{CHAT}，作为工作参考；{CONTENT}，包含具体说明。"
    processed = process_query(text)
    clauses = semantic_clauses(processed)
    assert clauses == (CHAT + ",作为工作参考", CONTENT + ",包含具体说明")
    assert all(clause in processed.original for clause in clauses)
    assert semantic_clause_sources(clauses) == ("chat", "content")


@pytest.mark.parametrize("quotes", [("“", "”"), ("「", "」"), ("『", "』"), ("《", "》"), ('"', '"'), ("'", "'")])
def test_quoted_punctuation_does_not_split_literal_content(quotes):
    opening, closing = quotes
    first = f"聊天里说{opening}交接材料，给杭州同事{closing}"
    clauses = semantic_clauses(process_query(first + "；" + CONTENT))
    assert clauses == (process_query(first).original, CONTENT)
    assert not semantic_clauses(process_query(opening + CHAT + "；" + CONTENT + closing))


def test_nested_quotes_and_english_apostrophes_remain_inside_one_source():
    first = '聊天里说“同事要求「交班，发给新人」”'
    assert semantic_clauses(process_query(first + "；" + CONTENT)) == (process_query(first).original, CONTENT)
    first = "聊天里说Peter's交接手册供杭州同事学习"
    assert semantic_clauses(process_query(first + "；" + CONTENT)) == (first, CONTENT)


@pytest.mark.parametrize("text", [
    f"帮我找一下，{CHAT}，不是{CONTENT}",
    f"帮我找一下，{CHAT}，或者{CONTENT}",
    f"帮我找一下，{CHAT}，{CONTENT}，不要过期内容",
    f"保留重要的客户交付条件，{CHAT}，{CONTENT}",
    f"{CHAT}，{CONTENT}，聊天里说上海项目组也要使用",
    f"杭州轮岗交接时要用的材料，作为工作参考，{CONTENT}",
    f"“{CHAT}；{CONTENT}",
    f"{CHAT}”；{CONTENT}",
    f"{CHAT}“；{CONTENT}”",
    f"帮我找一下，{CHAT}，文档名为 target.pdf，{CONTENT}",
    "聊天线索：；文件内容：备用电源",
    "文档的正文中提到；聊天里说杭州轮岗交接时用的材料",
    "资料的正文：“”；聊天里说杭州轮岗交接时用的材料",
    "请帮我找文件内容是这个；聊天里说杭州轮岗交接时用的材料",
])
def test_unsupported_relations_literals_or_incomplete_quotes_keep_whole_query(text):
    assert semantic_clauses(process_query(text)) == ()


def test_role_labeled_inner_text_is_not_discarded_for_length_limits():
    text = CHAT + "，" + "工作参考" * 60 + "；" + CONTENT
    assert semantic_clauses(process_query(text)) == ()


@pytest.mark.parametrize("form", [
    f"帮我找一下，{CHAT}，{CONTENT}",
    "聊天线索：杭州轮岗交接时要用的材料；文件内容：服务器房间停电后的供电保障",
    f"{CHAT}，作为工作参考，{CONTENT}，包含具体说明",
])
def test_query_variations_use_existing_batched_scoped_retrieval(corpus, form):
    _, service, embed, index, fid, _, _, body, chat = corpus
    service.enable_clause_semantics = True
    hits = service.search(SearchQuery(form, scope="chat_files", conversation="轮岗组"))
    assert hits[0].file_id == fid
    assert service.last_diagnostics.clause_count == 2
    assert "聊天 / 正文" in service.last_diagnostics.clause_status
    assert embed.calls == [(3, False)]
    assert {part.chunk_id for part in hits[0].supporting_evidence} == {body, chat}
    service.search(SearchQuery(form, scope="chat_files", conversation="轮岗组"))
    assert index.rebuilds == 1


@pytest.mark.parametrize("mode,evidence,scope", [
    ("keyword", "all", "chat_files"), ("semantic", "all", "chat_files"),
    ("hybrid", "content", "chat_files"), ("hybrid", "chat", "chat_files"),
    ("hybrid", "all", "messages"), ("hybrid", "all", "files"),
])
def test_other_retrieval_routes_do_not_silently_change(corpus, mode, evidence, scope):
    _, service, _, _, *_ = corpus
    query = SearchQuery(f"帮我找一下，{CHAT}，{CONTENT}", evidence=evidence, scope=scope)
    service.retrieval_mode = mode
    before = [(r.path, r.score) for r in service.search(query)]
    service.enable_clause_semantics = True
    assert [(r.path, r.score) for r in service.search(query)] == before
    assert service.last_diagnostics.clause_count == 0


def test_grouped_descriptions_keep_readable_evidence_and_preview(corpus):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication
    from localmind.ui import MainWindow
    from test_result_lifecycle import wait_for

    db, service, _, _, fid, _, _, _, _ = corpus
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    db.set_setting("privacy_accepted_v02", "1")
    db.set_setting("model_cache_dir", str(db.db_path.parent / "cache"))
    for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
        db.set_setting(key, "0")
    query = f"帮我找一下，{CHAT}，作为工作参考；{CONTENT}，包含具体说明"
    service.enable_clause_semantics = True
    results = service.search(SearchQuery(query, scope="chat_files"))
    window = MainWindow(db)
    window.search_input.setText(query)
    window.scope_filter.setCurrentIndex(window.scope_filter.findData("chat_files"))
    window.show()
    try:
        window._search_finished(results, service.last_diagnostics, {"total_ms": 34, "index_refresh_ms": 0})
        assert window.selected_result().file_id == fid
        assert "分句语义" in window.search_status.text()
        assert "杭州轮岗交接安排" in window.preview.toPlainText()
        assert "机房备用电池的维护计划" in window.preview.toPlainText()
        assert "对应描述" not in window.preview.toPlainText()
        window.evidence_expand.click()
        text = window.preview.toPlainText()
        assert "作为工作参考" in text and "包含具体说明" in text
        assert "帮我找一下" not in text
        window.evidence_expand.click()
        wait_for(app, lambda: window.file_preview._payload is not None)
        for width, height in ((980, 650), (1240, 800)):
            window.resize(width, height)
            app.processEvents()
            for widget in (window.search_input, window.preview, window.file_preview, window.file_actions[0]):
                corner = widget.mapTo(window, QPoint(widget.width(), widget.height()))
                assert corner.x() <= width and corner.y() <= height
            output = os.environ.get("LOCALMIND_UI_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(Path(output) / f"query-forms-{width}.png"))
    finally:
        window.close()
        app.processEvents()
