from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore
from localmind.chat_indexing import content_status, index_attachments
from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.models import SearchQuery
from localmind.ocr import OcrService
from localmind.wechat_files import AttachmentLocator
from test_weflow_v04 import index_file, searcher


@pytest.fixture
def db(tmp_path):
    value = LocalMindDB(tmp_path / "index.db")
    yield value
    value.close()


def prepare(db, tmp_path, contents):
    messages, files = [], {}
    for number, (name, content) in enumerate(contents):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        files[name] = path
        messages.append(ChatMessage(str(number), "发给项目组的文件", attachments=[Attachment(name)]))
    ChatStore(db).ingest(platform="wechat", account="test", conversation="项目组", source_key="test", messages=messages)
    locator = AttachmentLocator(db, cache_dir=tmp_path / "cache")
    attachments = ChatStore(db).attachments()
    for attachment in attachments:
        locator.bind(attachment["id"], str(files[attachment["name"]]))
    return [a["id"] for a in attachments], files


def run(db, tmp_path, ids, **kwargs):
    return index_attachments(db, ids, cache_dir=tmp_path / "cache",
        embedding_service=kwargs.pop("embedding", EmbeddingService(force_hash=True)),
        ocr_service=OcrService(enabled=False), **kwargs)


def test_confirmed_attachments_batch_body_search_and_repeat(db, tmp_path):
    ids, files = prepare(db, tmp_path, [
        ("材料甲.txt", "海南项目的验收验证码为蓝色印章。"),
        ("材料乙.txt", "机房备用电源每个季度需要做放电检查。"),
        ("材料丙.md", "邮件系统备份计划安排在星期日。")])
    before = {name: path.read_bytes() for name, path in files.items()}
    embedding = EmbeddingService(force_hash=True)
    embedding.encode_documents = Mock(wraps=embedding.encode_documents)
    report = run(db, tmp_path, ids, embedding=embedding)
    assert report["ready"] == 3 and report["issues"] == 0 and report["unique_files"] == 3
    assert report["summary"]["files_indexed"] == 3
    assert embedding.encode_documents.call_count == 1
    assert len(embedding.encode_documents.call_args.args[0]) == 3
    results = searcher(db, mode="keyword").search(SearchQuery("验收验证码", scope="chat_files"))
    assert results[0].path == str(files["材料甲.txt"])
    assert "项目组" in results[0].chat_context and results[0].source_type == "txt"
    for row in ChatStore(db).attachments():
        assert content_status(row, model_name=db.get_setting("embedding_model")) == "正文 + 语义"
    report = run(db, tmp_path, ids, embedding=embedding)
    assert report["ready"] == 3 and report["summary"]["files_indexed"] == 0
    assert embedding.encode_documents.call_count == 1
    assert {name: path.read_bytes() for name, path in files.items()} == before
    assert db.conn.execute("SELECT COUNT(*) FROM watched_folders").fetchone()[0] == 0


def test_duplicate_attachment_paths_are_indexed_once(db, tmp_path):
    ids, files = prepare(db, tmp_path, [("a.txt", "目标内容"), ("b.txt", "另一个文件")])
    locator = AttachmentLocator(db, cache_dir=tmp_path / "cache")
    for aid in ids:
        locator.bind(aid, str(files["a.txt"]))
    report = run(db, tmp_path, ids + ids)
    assert report["ready"] == 2 and report["unique_files"] == 1
    assert report["summary"]["files_indexed"] == 1


def test_missing_unsupported_and_broken_files_do_not_block_good_file(db, tmp_path):
    ids, files = prepare(db, tmp_path, [
        ("good.txt", "有效内容可以检索"), ("missing.txt", "已删除"), ("archive.zip", "unsupported"),
        ("broken.docx", "not an office document")])
    files["missing.txt"].unlink()
    report = run(db, tmp_path, ids, include_embeddings=False)
    assert report["ready"] == 1 and report["issues"] == 3
    assert report["summary"]["errors"] == 1
    assert db.get_file_by_path(str(files["archive.zip"])) is None
    assert any(row["state"] == "failed" for row in report["outcomes"])
    broken = next(a for a in ChatStore(db).attachments() if a["name"] == "broken.docx")
    assert content_status(broken, include_embeddings=False) == "失败，待重试"


def test_cancel_before_work_does_not_load_model(db, tmp_path):
    ids, _ = prepare(db, tmp_path, [("one.txt", "one")])
    embedding = Mock()
    report = run(db, tmp_path, ids, embedding=embedding, should_cancel=lambda: True)
    assert report["cancelled"] and report["ready"] == 0
    assert report["summary"]["files_indexed"] == 0
    assert len(report["outcomes"]) == 1
    embedding.encode_documents.assert_not_called()


def test_cancel_during_embedding_does_not_commit_pending_batch(db, tmp_path):
    ids, files = prepare(db, tmp_path, [("one.txt", "first"), ("two.txt", "second")])
    cancelled = Event()
    embedding = EmbeddingService(force_hash=True)
    encode = embedding.encode_documents
    def encode_then_cancel(texts):
        result = encode(texts)
        cancelled.set()
        return result
    embedding.encode_documents = encode_then_cancel
    report = run(db, tmp_path, ids, embedding=embedding, should_cancel=cancelled.is_set)
    assert report["cancelled"] and report["summary"]["files_indexed"] == 0
    assert all(db.get_file_by_path(str(path)) is None for path in files.values())
    cancelled.clear()
    assert run(db, tmp_path, ids)["ready"] == 2


def test_cancel_preserves_committed_files_and_accurate_ready_count(db, tmp_path, monkeypatch):
    from localmind import indexer
    ids, _ = prepare(db, tmp_path, [("one.txt", "first"), ("two.txt", "second")])
    monkeypatch.setattr(indexer, "INDEX_FILE_BATCH_SIZE", 1)
    cancelled = Event()
    def progress(text):
        if text.startswith("更新索引"):
            cancelled.set()
    report = run(db, tmp_path, ids, include_embeddings=False, should_cancel=cancelled.is_set, progress=progress)
    assert report["cancelled"] and report["summary"]["files_indexed"] == 1
    assert report["ready"] == 1
    assert sum(row["state"] == "cancelled" for row in report["outcomes"]) == 1


def test_unconfirmed_attachments_never_lookup_or_read_api(db, tmp_path, monkeypatch):
    ChatStore(db).ingest(platform="wechat", account="test", conversation="项目组", source_key="test",
                        messages=[ChatMessage("1", "file", attachments=[Attachment("unknown.txt")])])
    aid = ChatStore(db).attachments()[0]["id"]
    monkeypatch.setattr(AttachmentLocator, "locate", Mock(side_effect=AssertionError("must not locate")))
    monkeypatch.setattr(ChatStore, "_candidates", Mock(side_effect=AssertionError("must not search candidates")))
    report = run(db, tmp_path, [aid])
    assert report["issues"] == 1 and report["unique_files"] == 0
    assert "先打开定位" in report["outcomes"][0]["detail"]
    AttachmentLocator.locate.assert_not_called()
    ChatStore._candidates.assert_not_called()


def test_manual_rebinding_detaches_old_body_and_links_new_one(db, tmp_path):
    ids, files = prepare(db, tmp_path, [("a.txt", "旧版本独有标记")])
    assert run(db, tmp_path, ids)["ready"] == 1
    original = ChatStore(db).attachments()[0]["file_id"]
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("新版本独有标记", encoding="utf-8")
    locator = AttachmentLocator(db, cache_dir=tmp_path / "cache")
    locator.bind(ids[0], str(replacement))
    assert ChatStore(db).attachments()[0]["file_id"] is None
    assert run(db, tmp_path, ids)["ready"] == 1
    assert ChatStore(db).attachments()[0]["file_id"] != original
    results = searcher(db, mode="keyword").search(SearchQuery("新版本独有标记", scope="chat_files"))
    assert results[0].path == str(replacement)
    assert files["a.txt"].read_text(encoding="utf-8") == "旧版本独有标记"


def test_changed_confirmation_requires_rebinding(db, tmp_path):
    ids, files = prepare(db, tmp_path, [("a.txt", "original")])
    files["a.txt"].write_text("changed")
    report = run(db, tmp_path, ids)
    assert report["issues"] == 1 and report["ready"] == 0
    assert db.get_file_by_path(str(files["a.txt"])) is None


def test_keyword_index_can_be_upgraded_to_semantic(db, tmp_path):
    ids, _ = prepare(db, tmp_path, [("a.txt", "团队活动预算每人三百元")])
    assert run(db, tmp_path, ids, include_embeddings=False)["ready"] == 1
    row = ChatStore(db).attachments()[0]
    assert content_status(row, include_embeddings=False) == "关键词正文"
    assert content_status(row, include_embeddings=True) == "正文已索引，语义待补"
    assert run(db, tmp_path, ids)["ready"] == 1
    row = ChatStore(db).attachments()[0]
    assert content_status(row, model_name=db.get_setting("embedding_model")) == "正文 + 语义"


def test_empty_document_is_reported_and_reused(db, tmp_path):
    ids, _ = prepare(db, tmp_path, [("empty.txt", " \n ")])
    first = run(db, tmp_path, ids)
    assert first["empty"] == 1 and first["ready"] == 0
    row = ChatStore(db).attachments()[0]
    assert content_status(row, model_name=db.get_setting("embedding_model")) == "无可检索正文"
    assert run(db, tmp_path, ids)["summary"]["files_indexed"] == 0


def test_ocr_disabled_is_pending_not_an_empty_document(db, tmp_path):
    from localmind.models import OcrResult
    ids, _ = prepare(db, tmp_path, [("scan.png", "synthetic OCR fixture")])
    first = run(db, tmp_path, ids, include_embeddings=False)
    assert first["issues"] == 1 and first["ready"] == 0 and first["empty"] == 0
    assert content_status(ChatStore(db).attachments()[0], include_embeddings=False) == "OCR 待完成"
    class FakeOCR:
        enabled = True
        readiness_status = "ready"
        load_error = None
        def ocr_image(self, *args, **kwargs):
            return OcrResult("扫描附件中的唯一验收编号", "done", confidence=0.95)
    second = index_attachments(db, ids, cache_dir=tmp_path / "cache", ocr_service=FakeOCR(),
                               embedding_service=EmbeddingService(force_hash=True), include_embeddings=False)
    assert second["ready"] == 1
    assert searcher(db, mode="keyword").search(SearchQuery("唯一验收编号", scope="chat_files"))


def test_indexing_selection_does_not_associate_unselected_attachments(db, tmp_path):
    ids, files = prepare(db, tmp_path, [("selected.txt", "selected"), ("unselected.txt", "unselected")])
    selected = next(row["id"] for row in ChatStore(db).attachments() if row["name"] == "selected.txt")
    index_file(db, files["unselected.txt"], "unselected")
    assert run(db, tmp_path, [selected])["ready"] == 1
    unselected = next(row for row in ChatStore(db).attachments() if row["name"] == "unselected.txt")
    assert unselected["file_id"] is None


@pytest.mark.parametrize("ids", [[], [True], [-1], ["1"], list(range(1, 502))])
def test_invalid_selections_rejected(db, tmp_path, ids):
    with pytest.raises(ValueError):
        run(db, tmp_path, ids)
