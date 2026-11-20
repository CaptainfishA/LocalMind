"""User-requested repair of already imported WeChat data; no Alpha transport."""
from dataclasses import asdict

from .chats import ChatStore
from .embeddings import HashEmbeddingBackend
from .indexer import IndexingService
from .scanner import build_file_record


def maintenance_targets(db):
    sources = [dict(row) for row in db.conn.execute(
        "SELECT * FROM chat_sources WHERE platform='wechat' ORDER BY id")]
    paths = [row[0] for row in db.conn.execute("""SELECT DISTINCT f.path FROM files f
        JOIN chat_attachments a ON a.file_id=f.id JOIN chat_messages m ON m.id=a.message_id
        JOIN chat_sources s ON s.id=m.source_id WHERE s.platform='wechat'
        AND f.scan_status!='chat' ORDER BY f.path""")]
    return sources, paths


def update_wechat_semantics(db, *, embedding_service, ocr_service,
                            should_cancel=lambda: False, progress=lambda _: None):
    report = dict(changed=False, cancelled=False, sources_done=0, sources_total=0,
                  updated_units=0, reused_units=0, files_total=0, file_summary={}, issues=[])
    if should_cancel():
        return {**report, "cancelled": True}
    sources, paths = maintenance_targets(db)
    report.update(sources_total=len(sources), files_total=len(paths))
    if not sources and not paths:
        return report
    progress("检查本地缓存语义模型")
    model = embedding_service.active_model_name
    if model == HashEmbeddingBackend.name or embedding_service.load_error:
        raise ValueError("本地语义模型未就绪；未更新索引。请先检查模型缓存。")
    existing = set(db.vector_models())
    if existing and existing != {model}:
        raise ValueError("已有向量模型与当前模型不一致；未清空或混用索引")
    if should_cancel():
        return {**report, "cancelled": True}
    service = IndexingService(db, embedding_service=embedding_service, ocr_service=ocr_service)
    if paths:
        progress(f"更新 {len(paths)} 个已索引附件正文")
        available = []
        for path in paths:
            if should_cancel():
                return {**report, "cancelled": True}
            try:
                if build_file_record(path) is not None:
                    available.append(path)
            except OSError:
                pass
        if len(available) != len(paths):
            report["issues"].append(f"{len(paths) - len(available)} 个附件不可读取；保留已有索引，请检查文件路径")
        summary = service.index_paths(available, force_recheck=True, should_cancel=should_cancel, progress=progress)
        report["file_summary"] = asdict(summary)
        report["changed"] = bool(summary.files_indexed or summary.errors)
        report["cancelled"] = summary.cancelled
        if summary.errors:
            report["issues"].append(f"{summary.errors} 个文件未完成，可在附件正文状态中查看")
    store = ChatStore(db)
    for number, source in enumerate(sources, 1):
        if report["cancelled"] or should_cancel():
            report["cancelled"] = True
            break
        progress(f"更新本地会话 {number}/{len(sources)}")
        try:
            result = store.ingest(
                platform=source["platform"], account=source["account"], conversation=source["conversation"],
                source_key=source["source_key"], kind=source["kind"], messages=[], refresh_existing=True,
                embedding_service=embedding_service, should_cancel=should_cancel,
            )
            report["sources_done"] += 1
            report["updated_units"] += result["updated_units"]
            report["reused_units"] += result["reused_units"]
            report["changed"] |= result["updated_units"] > 0
        except Exception as exc:
            if should_cancel():
                report["cancelled"] = True
                break
            report["issues"].append(f"会话 {number} 未完成（{type(exc).__name__}）")
    return report
