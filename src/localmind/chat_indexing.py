"""Content indexing for an explicit selection of already located attachments."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .chats import ChatStore
from .config import SUPPORTED_EXTENSIONS
from .indexer import IndexingService, ScanSummary
from .weflow import SyncCancelled, WeFlowError
from .wechat_files import AttachmentLocator


def content_status(row, *, include_embeddings=True, model_name=None):
    if not row or not row.get("file_id"):
        return "未索引"
    if row.get("parse_status") == "failed":
        return "失败，待重试"
    if row.get("ocr_status") in {"failed", "skipped", "pending", "disabled"}:
        return "OCR 待完成"
    if not row.get("content_chunks"):
        return "无可检索正文" if row.get("parse_status") in {"parsed", "empty"} else "未索引"
    if (model_name and row.get("content_vectors") == row["content_chunks"]):
        return "正文 + 语义"
    return "正文已索引，语义待补" if include_embeddings else "关键词正文"


def index_attachments(db, attachment_ids, *, cache_dir, embedding_service, ocr_service,
                      include_embeddings=True, should_cancel=lambda: False, progress=lambda _: None):
    if (not isinstance(attachment_ids, list) or not attachment_ids or len(attachment_ids) > 500
            or any(type(aid) is not int or aid <= 0 for aid in attachment_ids)):
        raise ValueError("请选择 1 至 500 个附件")
    ids = list(dict.fromkeys(attachment_ids))
    locator = AttachmentLocator(db, cache_dir=cache_dir, cancel=should_cancel)
    store = ChatStore(db)
    outcomes, paths = [], {}
    summary = ScanSummary()
    for aid in ids:
        if should_cancel():
            summary.cancelled = True
            break
        outcome = {"attachment": aid, "name": "", "state": "pending", "detail": ""}
        outcomes.append(outcome)
        try:
            attachment = locator.attachment(aid)
            outcome["name"] = attachment["name"]
            located = locator.confirmed_location(aid)
            path = located["path"]
            if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise WeFlowError("当前不支持此文件类型的正文解析", code="file_type_unsupported")
            outcome["path"] = path
            outcome["expected_sha"] = db.conn.execute(
                "SELECT sha256 FROM chat_attachment_locations WHERE attachment_id=?", (aid,)).fetchone()[0]
            paths.setdefault(path, []).append(outcome)
            progress(f"已确认 {len(paths)} 个文件 · {attachment['name']}")
        except SyncCancelled:
            summary.cancelled = True
            outcome.update(state="cancelled", detail="已取消")
            break
        except (WeFlowError, OSError) as exc:
            outcome.update(state="unavailable", detail=str(exc))
    if not summary.cancelled and paths:
        service = IndexingService(db, embedding_service=embedding_service, ocr_service=ocr_service)
        summary = service.index_paths(list(paths), include_embeddings=include_embeddings, force_recheck=True,
                                      should_cancel=should_cancel, progress=progress)
    links_changed = store.resolve_links(attachment_ids=[row["attachment"] for rows in paths.values() for row in rows])
    model_name = db.get_setting("embedding_model") if include_embeddings else None
    for path, rows in paths.items():
        file = db.get_file_by_path(path)
        for outcome in rows:
            if file is None:
                outcome.update(state="cancelled" if summary.cancelled else "unavailable",
                               detail="已取消，尚未提交" if summary.cancelled else "文件已移除或暂不可读取")
            elif file["file_hash"] != outcome["expected_sha"]:
                outcome.update(state="cancelled" if summary.cancelled else "incomplete",
                               detail="新内容尚未提交" if summary.cancelled else "文件内容已变化，请重新确认附件路径后重试")
            elif file["parse_status"] == "failed":
                outcome.update(state="failed", detail=file["error"] or "正文解析失败")
            elif file["ocr_status"] in {"failed", "skipped", "pending", "disabled"}:
                outcome.update(state="incomplete", detail="OCR 尚未完成，可启用 OCR 后重试")
            else:
                count = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE file_id=?", (file["id"],)).fetchone()[0]
                if not count:
                    outcome.update(state="empty", detail="未提取到可检索正文")
                elif include_embeddings and (not model_name or not db.file_has_vectors(file["id"], model_name)):
                    outcome.update(state="incomplete", detail="语义向量尚未完成，可重试补齐")
                else:
                    outcome.update(state="ready", detail="正文 + 语义" if include_embeddings else "关键词正文")
    for aid in ids[len(outcomes):]:
        outcomes.append({"attachment": aid, "name": "", "state": "cancelled", "detail": "尚未处理"})
    return {"summary": asdict(summary), "outcomes": outcomes, "unique_files": len(paths),
            "ready": sum(r["state"] == "ready" for r in outcomes),
            "issues": sum(r["state"] in {"unavailable", "failed", "incomplete"} for r in outcomes),
            "empty": sum(r["state"] == "empty" for r in outcomes), "cancelled": summary.cancelled,
            "changed": summary.files_indexed > 0 or summary.errors > 0 or links_changed > 0}
