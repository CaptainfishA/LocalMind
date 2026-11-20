"""Render or interact with an isolated desktop containing synthetic office data only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from time import perf_counter, sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", default="benchmarks/results/ui-v062-preview")
    args = parser.parse_args()
    if not args.interactive:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase
    import pymupdf
    from docx import Document
    from openpyxl import Workbook
    from localmind.chats import Attachment, ChatMessage, ChatStore
    from localmind.db import LocalMindDB
    from localmind.embeddings import EmbeddingService
    from localmind.indexer import IndexingService
    from localmind.models import SearchQuery
    from localmind.ocr import OcrService
    from localmind.reranker import RerankerService
    from localmind.searcher import SearchService
    from localmind.ui import MainWindow

    app = QApplication([])
    for font in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(font).exists():
            QFontDatabase.addApplicationFont(font)
    with tempfile.TemporaryDirectory(prefix="localmind-ui-preview-") as temp:
        root = Path(temp)
        files = root / "演示资料"
        files.mkdir()
        samples = [
            ("星河项目_报价方案.docx", "星河项目报价方案\n交付范围：产品设计、交互原型与前端实现。\n报价总计 68,000 元。项目分为调研、设计和交付三个阶段。\n本方案已经根据客户反馈更新，包含验收安排与付款节点。"),
            ("星河项目_需求确认.docx", "星河项目需求确认\n本期报价涵盖工作台、搜索页与资料管理。\n双方确认验收时间为十月底，附件中列明设计交付物。"),
            ("星河项目_会议纪要.md", "# 星河项目会议纪要\n\n本周讨论了报价方案与交付时间。需要补充页面范围、验收条件和后续维护费用。"),
            ("供应商报价对比.xlsx", "供应商,报价,交付周期\n示例设计团队,68000,四周\n示例开发团队,72000,五周"),
            ("报价审批流程.md", "# 报价审批\n\n项目负责人确认交付范围，财务审核报价金额，客户确认后进入合同签署。"),
            ("项目交付清单.md", "# 交付清单\n\n源文件、交互说明、组件规范、验收记录及报价附件。"),
            ("星河项目_验收说明.pdf", "星河项目报价与验收\n第一阶段：产品设计与交互原型\n第二阶段：前端实现与联调\n交付文件：设计源文件、组件规范、验收记录"),
        ]
        for name, content in samples:
            path = files / name
            if path.suffix == ".docx":
                document = Document()
                for line in content.splitlines():
                    document.add_paragraph(line)
                document.save(path)
            elif path.suffix == ".pdf":
                document = pymupdf.open()
                for number in range(2):
                    page = document.new_page()
                    page.draw_rect((36, 35, 559, 100), fill=(0.14, 0.48, 0.39), color=(0.14, 0.48, 0.39))
                    page.insert_text((52, 76), "星河项目 / 验收说明", fontname="china-s", fontsize=23, color=(1, 1, 1))
                    page.insert_textbox((52, 134, 540, 650), content + f"\n\n第 {number + 1} 页 · 合成预览资料",
                                        fontname="china-s", fontsize=17, lineheight=1.8)
                document.save(path)
                document.close()
            elif path.suffix == ".xlsx":
                workbook = Workbook()
                for line in content.splitlines():
                    workbook.active.append(line.split(","))
                workbook.save(path)
            else:
                path.write_text(content, encoding="utf-8")
        db = LocalMindDB(root / "preview.db")
        for key in ("privacy_accepted_v02", "weflow_privacy_accepted"):
            db.set_setting(key, "1")
        for key in ("startup_scan", "monitor_enabled", "embedding_enabled", "ocr_enabled", "reranker_enabled"):
            db.set_setting(key, "0")
        db.set_setting("search_retrieval_mode", "keyword")
        db.set_setting("model_cache_dir", str(root / "cache"))
        db.add_watched_folder(str(files))
        embedding = EmbeddingService(force_hash=True)
        IndexingService(db, embedding_service=embedding, ocr_service=OcrService(enabled=False)).scan_and_index(
            [files], include_embeddings=False)
        store = ChatStore(db)
        for number, (name, content) in enumerate(samples[:3]):
            store.ingest(platform="wechat", account="演示账号", conversation="星河项目协作群",
                         source_key="synthetic-preview", kind="offline", messages=[
                             ChatMessage(str(number), "这是本周确认的报价资料，包含客户反馈和验收安排。",
                                         sender="示例联系人", sent_at="2026-10-20T02:30:00Z",
                                         attachments=[Attachment(name, str(files / name))])])
        window = MainWindow(db)
        try:
            window.setWindowTitle(window.windowTitle() + " - 界面预览（合成数据）")
            window.search_input.setText("星河项目报价")
            service = SearchService(db, embedding_service=embedding, reranker_service=RerankerService(enabled=False),
                                    retrieval_mode="keyword")
            start = perf_counter()
            results = service.search(SearchQuery("星河项目报价"))
            elapsed = (perf_counter() - start) * 1000
            window._search_finished(results, service.last_diagnostics, {"total_ms": elapsed, "index_refresh_ms": 0})
            window.show()
            app.processEvents()
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=True)
            def wait_preview():
                deadline = perf_counter() + 12
                while perf_counter() < deadline:
                    app.processEvents()
                    if not window.file_preview._debounce.isActive() and window.file_preview._process is None:
                        break
                    sleep(0.01)
                app.processEvents()
            wait_preview()
            for width, height in ((1380, 860), (980, 650)):
                window.resize(width, height)
                for index in range(5):
                    window.tabs.setCurrentIndex(index)
                    app.processEvents()
                    window.grab().save(str(output / f"page-{index}-{width}.png"))
                window.tabs.setCurrentIndex(1)
                window.chat_panel.data_tabs.setCurrentIndex(1)
                app.processEvents()
                window.grab().save(str(output / f"chat-sources-{width}.png"))
                window.chat_panel.data_tabs.setCurrentIndex(0)
            window.tabs.setCurrentIndex(0)
            for ext in ("pdf", "xlsx", "docx"):
                row = next(i for i, result in enumerate(window.results) if result.ext == ext)
                window.result_table.selectRow(row)
                window.show_selected_result()
                wait_preview()
                for width, height in ((1380, 860), (980, 650)):
                    window.resize(width, height)
                    app.processEvents()
                    window.grab().save(str(output / f"preview-{ext}-{width}.png"))
            window.resize(1380, 860)
            window.focus_search()
            app.processEvents()
            if args.interactive:
                app.exec()
        finally:
            window.close()
            app.processEvents()


if __name__ == "__main__":
    main()
