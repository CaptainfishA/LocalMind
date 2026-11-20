"""Private, bounded annotation snapshots; never writes to the source index."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from ..file_freshness import inspect_file

SCHEMA = "localmind.wechat-review.v1"
EVIDENCE = {"all": "全部线索", "content": "文件正文", "chat": "聊天线索"}
STATUSES = {"draft", "confirmed", "skipped"}
OUTCOMES = {"answerable", "no_complete_answer", "uncertain"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    from PySide6.QtCore import QIODevice, QSaveFile

    target = QSaveFile(str(path))
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if not target.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError("无法写入评测文件")
    if target.write(payload) != len(payload) or not target.commit():
        target.cancelWriting()
        raise OSError("评测文件保存失败，原记录保持不变")


def readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    deadline = time.monotonic() + 60
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    return conn


def inspect_source(path):
    with closing(readonly(path)) as conn:
        return dict(conn.execute("""SELECT COUNT(*) attachments,
            COUNT(DISTINCT a.file_id) linked_files, COUNT(DISTINCT s.id) sources
            FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
            JOIN chat_sources s ON s.id=m.source_id WHERE s.platform='wechat'""").fetchone())


def extract_corpus(path, *, limit=100, seed=20261105):
    """Freeze indexed excerpts, not originals or the whole database/settings."""
    if not 1 <= limit <= 200:
        raise ValueError("候选文件上限应在 1 至 200 之间")
    with closing(readonly(path)) as conn:
        conn.execute("BEGIN")
        rows = conn.execute("""SELECT DISTINCT f.* FROM files f
            JOIN chat_attachments a ON a.file_id=f.id
            JOIN chat_messages m ON m.id=a.message_id JOIN chat_sources s ON s.id=m.source_id
            WHERE s.platform='wechat' ORDER BY f.id LIMIT 20001""").fetchall()
        if len(rows) > 20000:
            raise ValueError("已关联文件超过抽样预算，请先指定较小的独立索引")
        total = len(rows)
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
        documents = []
        for row in rows:
            body = conn.execute("""SELECT id,substr(chunk_text,1,2000) text,page_no,sheet_name,
                source_type,length(chunk_text)>2000 clipped FROM chunks
                WHERE file_id=? AND source_type!='chat' ORDER BY id LIMIT 41""", (row["id"],)).fetchall()
            sources = conn.execute("""SELECT a.id attachment_id,m.id message_id,s.id source_id,
                s.conversation,m.sender,m.sent_at,substr(m.body,1,2000) body,
                substr(a.context,1,3000) context,a.origin,
                (length(m.body)>2000 OR length(a.context)>3000) clipped
                FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
                JOIN chat_sources s ON s.id=m.source_id WHERE a.file_id=? AND s.platform='wechat'
                ORDER BY a.id DESC LIMIT 13""", (row["id"],)).fetchall()
            chat_chunks = conn.execute("""SELECT c.id,substr(c.chunk_text,1,2000) text,
                u.message_id,u.attachment_id,length(c.chunk_text)>2000 clipped
                FROM chat_units u JOIN chunks c ON c.id=u.chunk_id
                JOIN chat_attachments a ON a.id=u.attachment_id
                JOIN chat_messages m ON m.id=u.message_id JOIN chat_sources s ON s.id=m.source_id
                WHERE a.file_id=? AND s.platform='wechat' ORDER BY c.id DESC LIMIT 41""", (row["id"],)).fetchall()
            documents.append({
                "file_id": f"f{len(documents)+1:04d}", "source_file_id": row["id"],
                "name": row["name"], "path": row["path"], "ext": row["ext"],
                "indexed_hash": row["file_hash"], "modified_at": row["modified_at"],
                "parse_status": row["parse_status"], "file_state": inspect_file(row),
                "body": [dict(r) for r in body[:40]], "sources": [dict(r) for r in sources[:12]],
                "chat_chunks": [dict(r) for r in chat_chunks[:40]], "chat_more": len(chat_chunks) > 40,
                "excerpt_only": True, "body_more": len(body) > 40, "sources_more": len(sources) > 12,
            })
        return documents, {"linked_files_available": total, "selected_files": len(documents),
                           "selection": "seeded_linked_wechat_subset", "seed": seed,
                           "unlinked_attachments_included": False, "raw_files_copied": False}


def cue(text, name=""):
    """Extractive prompts are drafts, never claimed to be natural user queries."""
    import jieba.analyse

    if name:
        text = text.replace(name, " ").replace(Path(name).stem, " ")
    text = re.sub(r"(?m)^(?:平台|会话|发送者|消息时间|附件|文件名提及，待确认):.*$", "", text)
    text = text.replace("上下文（引用 / 相邻消息，非附件内容）:", "")
    words = jieba.analyse.extract_tags(text[:12000], topK=12)
    words = [w for w in words if len(w) >= 2 and re.search(r"[\u4e00-\u9fffA-Za-z]", w)
             and w not in {"文件", "附件", "消息", "微信", "发送", "聊天", "来源", "http", "https", "收到"}]
    return "、".join(words[:3])[:70]


def make_questions(documents, count=30):
    if not 1 <= count <= 150 or not documents:
        raise ValueError("没有可用资料或题量超出范围")
    # Prefer distinct files; duplicates are reported rather than silently called independent tasks.
    cues = {d["file_id"]: (
        cue("\n".join(r["text"] for r in d["body"]), d["name"]),
        cue("\n".join([r["context"] or r["body"] for r in d["sources"]] +
                       [r["text"] for r in d.get("chat_chunks", [])]), d["name"])) for d in documents}
    questions = []
    used = set()
    seen_queries = set()
    for index in range(count):
        kind = ("content", "chat", "all", "content", "chat")[index % 5]
        available = [d for d in documents if d["file_id"] not in used] or documents
        eligible = [d for d in available if (cues[d["file_id"]][0] if kind == "content" else
                    cues[d["file_id"]][1] if kind == "chat" else all(cues[d["file_id"]]))]
        for doc in eligible + [d for d in available if d not in eligible]:
            content, chat = cues[doc["file_id"]]
            selected_kind = kind
            if kind == "content" and content:
                query = f"找一份内容涉及{content}的资料"
            elif kind == "chat" and chat:
                query = f"聊天里讨论{chat}时提到的那份文件"
            elif content and chat:
                selected_kind, query = "all", f"聊天中提到{chat}，文件内容涉及{content}的资料"
            else:
                selected_kind, query = "all", f"找一下《{Path(doc['name']).stem}》这份资料"
            if query not in seen_queries:
                break
        else:
            raise ValueError("资料不足以拟出互不重复的问题，请减少题量或增加资料")
        kind = selected_kind
        used.add(doc["file_id"])
        seen_queries.add(query)
        questions.append({"query_id": f"q{index+1:03d}", "text": query, "evidence": kind,
                          "origin": "extractive_draft", "suggested_file": doc["file_id"],
                          "judgments": {}, "outcome": "answerable", "status": "draft",
                          "notes": "", "group": "", "split": "unassigned", "all_reviewed": False})
    return questions


def build(source, output, *, count=30, limit=100):
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("输出目录已存在，不能覆盖人工标注")
    documents, metadata = extract_corpus(source, limit=limit)
    questions = make_questions(documents, count)
    corpus = {"schema": SCHEMA, "created_at": datetime.now(UTC).isoformat(),
              "selection": metadata, "documents": documents}
    root.mkdir(parents=True)
    atomic_json(root / "corpus.json", corpus)
    state = {"schema": SCHEMA, "dataset_id": uuid.uuid4().hex, "corpus_digest": digest(corpus),
             "revision": 0, "current": 0, "questions": questions}
    atomic_json(root / "review.json", state)
    return {**metadata, "questions": len(questions), "confirmed": 0,
            "distinct_suggested_targets": len({q["suggested_file"] for q in questions})}


def validate_question(q, ids, *, confirming=False):
    if not isinstance(q.get("text"), str) or len(q["text"]) > 2000:
        raise ValueError("问题长度不能超过 2000 字")
    if q.get("evidence") not in EVIDENCE or q.get("status") not in STATUSES or q.get("outcome") not in OUTCOMES:
        raise ValueError("题目状态或范围无效")
    if q.get("split") not in {"unassigned", "dev", "test"}:
        raise ValueError("数据划分无效")
    grades = q.get("judgments", {})
    if not isinstance(grades, dict) or any(k not in ids or type(v) is not int or v not in range(4) for k, v in grades.items()):
        raise ValueError("答案标注无效")
    if confirming or q["status"] == "confirmed":
        if not q["text"].strip() or not q.get("notes", "").strip():
            raise ValueError("确认前请填写问题和判定依据")
        if q["outcome"] == "uncertain":
            raise ValueError("不确定的题目只能保存草稿或跳过")
        full = any(v >= 2 for v in grades.values())
        if q["outcome"] == "answerable" and not full:
            raise ValueError("请选择至少一个完整答案（2 或 3 分）")
        if q["outcome"] == "no_complete_answer" and (full or set(grades) != ids or not q.get("all_reviewed")):
            raise ValueError("无完整答案需核对全部候选且没有 2/3 分答案；未标注不等于不相关")


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.corpus = json.loads((self.root / "corpus.json").read_text(encoding="utf-8"))
        self.state = json.loads((self.root / "review.json").read_text(encoding="utf-8"))
        if self.corpus.get("schema") != SCHEMA or self.state.get("schema") != SCHEMA:
            raise ValueError("不是受支持的微信评测集")
        if digest(self.corpus) != self.state.get("corpus_digest"):
            raise ValueError("冻结资料已经变化，停止载入")
        self.documents = {d["file_id"]: d for d in self.corpus["documents"]}
        if not self.documents or len(self.documents) != len(self.corpus["documents"]):
            raise ValueError("资料身份重复或为空")
        self.ids = set(self.documents)
        questions = self.state["questions"]
        if not questions or len({q["query_id"] for q in questions}) != len(questions):
            raise ValueError("题目为空或身份重复")
        for q in questions:
            validate_question(q, self.ids)
            if q["suggested_file"] not in self.ids:
                raise ValueError("建议目标不存在")
        self.disk_digest = digest(self.state)

    def save(self, state):
        # Detect a second annotator/window before replacing saved progress.
        current = json.loads((self.root / "review.json").read_text(encoding="utf-8"))
        if digest(current) != self.disk_digest:
            raise ValueError("另一个窗口已修改标注，请关闭当前窗口后重新载入")
        self.check_corpus()
        for q in state["questions"]:
            validate_question(q, self.ids)
        new = copy.deepcopy(state)
        new["revision"] = self.state["revision"] + 1
        atomic_json(self.root / "review.json", new)
        self.state, self.disk_digest = new, digest(new)

    def export(self):
        self.check_corpus()
        current = json.loads((self.root / "review.json").read_text(encoding="utf-8"))
        if digest(current) != self.disk_digest:
            raise ValueError("标注文件已被外部修改，请重新载入后导出")
        confirmed = [q for q in self.state["questions"] if q["status"] == "confirmed"]
        if not confirmed:
            raise ValueError("尚无人工确认题目")
        for q in confirmed:
            validate_question(q, self.ids, confirming=True)
        payload = {"schema": SCHEMA, "dataset_id": self.state["dataset_id"],
                   "corpus_digest": self.state["corpus_digest"], "documents": self.corpus["documents"],
                   "questions": confirmed, "strict_positive_grade": 2,
                   "unjudged_is_negative": False, "independent_test_set": False,
                   "evaluation_scope": "frozen_index_excerpts_only",
                   "excluded_questions": len(self.state["questions"]) - len(confirmed)}
        path = self.root / f"confirmed-r{self.state['revision']}-{uuid.uuid4().hex[:8]}.json"
        atomic_json(path, payload)
        return path

    def check_corpus(self):
        current = json.loads((self.root / "corpus.json").read_text(encoding="utf-8"))
        if digest(current) != self.state["corpus_digest"]:
            raise ValueError("冻结资料已经变化，不能继续保存或导出")


def main():
    parser = argparse.ArgumentParser(description="本地微信资料人工评测集（含私密正文，不上传）")
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("inspect")
    check.add_argument("source")
    create = sub.add_parser("build")
    create.add_argument("--source", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--questions", type=int, default=30)
    create.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    result = inspect_source(args.source) if args.action == "inspect" else build(
        args.source, args.output, count=args.questions, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
