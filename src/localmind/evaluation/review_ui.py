"""Standalone human review workspace for a private frozen WeChat pilot."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from PySide6.QtCore import QLockFile, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPlainTextEdit, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..desktop.theme import APP_STYLESHEET
from ..desktop.widgets import button, label
from .review import EVIDENCE, ReviewStore, validate_question


class ReviewWindow(QDialog):
    def __init__(self, root):
        super().__init__()
        self.lock = QLockFile(str(Path(root).resolve() / "review.lock"))
        if not self.lock.tryLock(0):
            raise ValueError("此评测集已在另一个窗口中打开")
        try:
            self.store = ReviewStore(root)
        except Exception:
            self.lock.unlock()
            raise
        self.state = copy.deepcopy(self.store.state)
        self.index, self.loading, self.dirty = 0, False, False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(700)
        self.timer.timeout.connect(self.save)
        self.setWindowTitle("LocalMind · 微信评测集人工确认")
        self.resize(1320, 860)
        self.setMinimumSize(980, 680)
        self.setStyleSheet(APP_STYLESHEET + "QPlainTextEdit { font-size: 15px; border: 1px solid #dce2e5; border-radius: 4px; padding: 7px; } QListWidget { font-size: 14px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        header = QHBoxLayout()
        header.addWidget(label("微信检索 · 人工确认", "pageTitle"))
        header.addStretch()
        self.progress = label("")
        header.addWidget(self.progress)
        header.addWidget(button("导出已确认", "download", self.export))
        layout.addLayout(header)
        layout.addWidget(label("本地私密资料 · 已索引摘录快照 · 自动拟题待核对", wrap=True))
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        self.questions = QListWidget()
        self.questions.setMinimumWidth(145)
        self.questions.setMaximumWidth(200)
        splitter.addWidget(self.questions)
        middle = QWidget()
        form = QVBoxLayout(middle)
        form.setContentsMargins(14, 0, 14, 0)
        form.addWidget(label("检索问题", "sectionTitle"))
        self.query = QPlainTextEdit()
        self.query.setMaximumHeight(115)
        form.addWidget(self.query)
        options = QHBoxLayout()
        self.evidence = QComboBox()
        for key, title in EVIDENCE.items():
            self.evidence.addItem(title, key)
        options.addWidget(self.evidence)
        self.outcome = QComboBox()
        for title, key in (("存在完整答案", "answerable"), ("快照内无完整答案", "no_complete_answer"), ("暂不确定", "uncertain")):
            self.outcome.addItem(title, key)
        options.addWidget(self.outcome)
        form.addLayout(options)
        form.addWidget(label("判定依据 / 必须满足的条件", "sectionTitle"))
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(95)
        form.addWidget(self.notes)
        form.addWidget(label("候选文件", "sectionTitle"))
        self.find = QLineEdit()
        self.find.setPlaceholderText("筛选文件名")
        form.addWidget(self.find)
        self.files = QListWidget()
        self.files.setMinimumHeight(130)
        form.addWidget(self.files, 1)
        grading = QHBoxLayout()
        grading.addWidget(label("所选文件"))
        self.grade = QComboBox()
        for text, value in (("未标注", None), ("0 · 不相关", 0), ("1 · 部分相关", 1), ("2 · 可接受完整答案", 2), ("3 · 精确目标", 3)):
            self.grade.addItem(text, value)
        grading.addWidget(self.grade, 1)
        form.addLayout(grading)
        form.addWidget(button("其余未标注设为不相关", "check", self.mark_remaining))
        self.all_reviewed = QCheckBox("已核对全部快照候选（仅限摘录范围）")
        form.addWidget(self.all_reviewed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(middle)
        scroll.setMinimumWidth(330)
        splitter.addWidget(scroll)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.file_title = label("", "sectionTitle", wrap=True)
        detail_layout.addWidget(self.file_title)
        tabs = QTabWidget()
        self.body, self.chat, self.meta = [QPlainTextEdit() for _ in range(3)]
        for title, view in (("正文摘录", self.body), ("聊天依据", self.chat), ("文件状态", self.meta)):
            view.setReadOnly(True)
            tabs.addTab(view, title)
        detail_layout.addWidget(tabs)
        splitter.addWidget(detail)
        splitter.setSizes([165, 480, 620])
        footer = QHBoxLayout()
        self.previous = button("上一题", "chevron-left", lambda: self.navigate(self.index - 1), compact=True)
        self.next = button("下一题", "chevron-right", lambda: self.navigate(self.index + 1), compact=True)
        footer.addWidget(self.previous)
        footer.addWidget(self.next)
        footer.addWidget(button("保存草稿", "download", self.save))
        footer.addWidget(button("跳过本题", "x", self.skip))
        footer.addStretch()
        footer.addWidget(button("确认并继续", "check", self.confirm, primary=True))
        layout.addLayout(footer)
        self.receipt = label("", wrap=True)
        layout.addWidget(self.receipt)
        self.query.textChanged.connect(self.changed)
        self.notes.textChanged.connect(self.changed)
        self.evidence.currentIndexChanged.connect(self.changed)
        self.outcome.currentIndexChanged.connect(self.changed)
        self.all_reviewed.toggled.connect(self.changed)
        self.grade.currentIndexChanged.connect(self.grade_changed)
        self.files.currentRowChanged.connect(self.show_file)
        self.questions.currentRowChanged.connect(self.navigate)
        self.find.textChanged.connect(self.filter_files)
        for doc in self.store.documents.values():
            self.files.addItem(doc["name"])
            self.files.item(self.files.count() - 1).setData(Qt.ItemDataRole.UserRole, doc["file_id"])
        self.refresh_progress()
        self.load_question(min(max(0, self.state.get("current", 0)), len(self.state["questions"]) - 1))
        self.receipt.setText(f"进度文件：{self.store.root / 'review.json'}")

    def refresh_progress(self):
        values = self.state["questions"]
        confirmed = sum(q["status"] == "confirmed" for q in values)
        skipped = sum(q["status"] == "skipped" for q in values)
        self.progress.setText(f"已确认 {confirmed} / {len(values)} · 跳过 {skipped}")
        self.questions.blockSignals(True)
        self.questions.clear()
        names = {"draft": "待确认", "confirmed": "已确认", "skipped": "已跳过"}
        for i, q in enumerate(values):
            self.questions.addItem(f"{i+1:02d}   {names[q['status']]}")
        self.questions.setCurrentRow(self.index)
        self.questions.blockSignals(False)

    def load_question(self, index):
        self.loading = True
        self.index = index
        q = self.state["questions"][index]
        self.query.setPlainText(q["text"])
        self.notes.setPlainText(q["notes"])
        self.evidence.setCurrentIndex(self.evidence.findData(q["evidence"]))
        self.outcome.setCurrentIndex(self.outcome.findData(q["outcome"]))
        self.all_reviewed.setChecked(q["all_reviewed"])
        self.find.clear()
        for i in range(self.files.count()):
            item = self.files.item(i)
            fid = item.data(Qt.ItemDataRole.UserRole)
            grade = q["judgments"].get(fid)
            item.setText(f"{'·' if grade is None else grade}   {self.store.documents[fid]['name']}" + ("  [拟题来源]" if fid == q["suggested_file"] else ""))
            if fid == q["suggested_file"]:
                self.files.setCurrentRow(i)
        self.show_file(self.files.currentRow())
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(index + 1 < len(self.state["questions"]))
        self.loading = False
        self.refresh_progress()

    def show_file(self, index):
        if index < 0:
            return
        fid = self.files.item(index).data(Qt.ItemDataRole.UserRole)
        doc = self.store.documents[fid]
        self.file_title.setText(doc["name"])
        self.body.setPlainText("\n\n".join(
            f"片段 {r['id']} · 页 {r['page_no'] or '-'} · 工作表 {r['sheet_name'] or '-'}" + (" · 已节选" if r["clipped"] else "") + "\n" + r["text"] for r in doc["body"]
        ) or "没有已索引正文；不能据此判定文件没有内容。")
        source_text = "\n\n".join(
            f"{r['conversation']} · {r['sender']} · {r['sent_at'] or '时间未知'}\n关联消息 {r['message_id']} / 附件 {r['attachment_id']}\n"
            f"{r['body']}\n\n关联上下文（不一定由同一发送者所说）：\n{r['context']}" + ("\n[已节选]" if r["clipped"] else "") for r in doc["sources"]
        )
        indexed_context = "\n\n".join(f"索引片段 {r['id']} / 关联消息 {r['message_id']}\n{r['text']}" + ("\n[已节选]" if r["clipped"] else "") for r in doc.get("chat_chunks", []))
        self.chat.setPlainText((source_text + "\n\n已索引上下文（可能包含相邻消息或引用，不归于单一说话人）：\n" + indexed_context).strip() if source_text or indexed_context else "没有冻结的聊天来源。")
        self.meta.setPlainText(f"{doc['path']}\n\n快照状态：{doc['file_state']}\n解析状态：{doc['parse_status']}\n索引哈希：{doc['indexed_hash'] or '未知'}\n正文未全量包含：{doc['body_more']}\n来源未全量包含：{doc['sources_more']}\n聊天片段未全量包含：{doc.get('chat_more', False)}\n\n这是索引摘录，不是文件/聊天的完整备份。")
        self.grade.blockSignals(True)
        self.grade.setCurrentIndex(self.grade.findData(self.state["questions"][self.index]["judgments"].get(fid)))
        self.grade.blockSignals(False)

    def filter_files(self, text):
        for i in range(self.files.count()):
            item = self.files.item(i)
            item.setHidden(text.casefold() not in item.text().casefold())

    def changed(self, *_):
        if self.loading:
            return
        q = self.state["questions"][self.index]
        q.update(text=self.query.toPlainText(), notes=self.notes.toPlainText(),
                 evidence=self.evidence.currentData(), outcome=self.outcome.currentData(),
                 all_reviewed=self.all_reviewed.isChecked(), status="draft")
        self.dirty = True
        self.receipt.setText("修改待保存")
        self.refresh_progress()
        self.timer.start()

    def grade_changed(self, *_):
        if self.loading or self.files.currentRow() < 0:
            return
        fid = self.files.currentItem().data(Qt.ItemDataRole.UserRole)
        q = self.state["questions"][self.index]
        value = self.grade.currentData()
        if value is None:
            q["judgments"].pop(fid, None)
        else:
            q["judgments"][fid] = value
        self.files.currentItem().setText(f"{'·' if value is None else value}   {self.store.documents[fid]['name']}" + ("  [拟题来源]" if fid == q["suggested_file"] else ""))
        self.changed()

    def save(self):
        self.timer.stop()
        if not self.dirty:
            return True
        try:
            self.state["current"] = self.index
            self.store.save(self.state)
            self.state = copy.deepcopy(self.store.state)
        except (ValueError, OSError) as exc:
            self.receipt.setText(f"未保存：{exc}")
            return False
        self.dirty = False
        self.receipt.setText(f"已保存 · 修订 {self.state['revision']} · {self.store.root.name}")
        return True

    def navigate(self, index):
        if self.loading or index < 0 or index >= len(self.state["questions"]):
            return
        if not self.save():
            self.refresh_progress()
            return
        self.load_question(index)
        self.state["current"] = index
        self.dirty = True
        self.save()

    def confirm(self):
        q = self.state["questions"][self.index]
        try:
            validate_question(q, self.store.ids, confirming=True)
        except ValueError as exc:
            QMessageBox.information(self, "尚不能确认", str(exc))
            return
        q["status"] = "confirmed"
        self.dirty = True
        if self.save():
            self.refresh_progress()
            if self.index + 1 < len(self.state["questions"]):
                self.navigate(self.index + 1)

    def skip(self):
        self.state["questions"][self.index]["status"] = "skipped"
        self.dirty = True
        if self.save():
            self.refresh_progress()
            self.navigate(min(self.index + 1, len(self.state["questions"]) - 1))

    def mark_remaining(self):
        if QMessageBox.question(self, "核对全部候选", "确认已经核对本范围全部候选？将所有未标注文件设为 0 分（不相关），不会改动已有评分。") != QMessageBox.StandardButton.Yes:
            return
        q = self.state["questions"][self.index]
        for fid in self.store.ids:
            q["judgments"].setdefault(fid, 0)
        self.changed()
        self.load_question(self.index)

    def export(self):
        if not self.save():
            return
        try:
            path = self.store.export()
        except (ValueError, OSError) as exc:
            QMessageBox.information(self, "未导出", str(exc))
            return
        self.receipt.setText(f"已导出人工确认题目：{path.name}（本地私密数据）")

    def closeEvent(self, event):
        if not self.save():
            if QMessageBox.question(self, "保存失败", "放弃本窗口尚未保存的修改并关闭？已保存记录不会被删除。") != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.lock.unlock()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = ReviewWindow(args.dataset)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
