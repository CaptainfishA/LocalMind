import hashlib
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest

from localmind.chats import Attachment, ChatMessage, ChatStore, digest
from localmind.db import LocalMindDB
from localmind.wechat_files import AttachmentLocator, api_identity, local_file, save_api_identities, RESOLVE_TOOL
from localmind.weflow import SyncCancelled, WeFlowClient, WeFlowError


def setup(db, *, name="report.txt", size=None, md5="", kind="explicit", origin=""):
    ChatStore(db).ingest(platform="wechat", account="owner", conversation="group",
        source_key="source", kind="weflow_api",
        messages=[ChatMessage("123", "[文件] " + name, sent_at="2026-10-20T00:00:00+00:00",
                              attachments=[Attachment(name, size=size, md5=md5, kind=kind, origin=origin)],
                              forwarded=bool(origin))])
    return ChatStore(db).attachments()[0]["id"]


@pytest.fixture
def db(tmp_path):
    value = LocalMindDB(tmp_path / "index.db")
    yield value
    value.close()


def locator(db, tmp_path, **kwargs):
    return AttachmentLocator(db, cache_dir=tmp_path / "cache", **kwargs)


def test_manual_file_opens_without_index_and_without_token(db, tmp_path):
    aid = setup(db)
    file = tmp_path / "report.txt"
    file.write_text("local test")
    service = locator(db, tmp_path)
    service.bind(aid, str(file))
    assert db.get_file_by_path(str(file)) is None
    assert service.locate(aid)["path"] == str(file)
    assert ChatStore(db).attachments()[0]["local_path"] == str(file)


def test_changed_or_deleted_cached_file_not_silently_opened(db, tmp_path):
    aid = setup(db)
    path = tmp_path / "report.txt"
    path.write_text("original")
    service = locator(db, tmp_path)
    service.bind(aid, str(path))
    path.write_text("changed")
    with pytest.raises(WeFlowError, match="Token"):
        service.locate(aid)
    path.unlink()
    with pytest.raises(WeFlowError, match="Token"):
        service.locate(aid)


def test_filename_candidates_require_confirmation_even_one(db, tmp_path):
    aid = setup(db)
    root = tmp_path / "wechat"
    root.mkdir()
    path = root / "report.txt"
    path.write_text("content")
    service = locator(db, tmp_path, root=str(root))
    result = service.locate(aid)
    assert result["candidates"] == [str(path)]
    assert "path" not in result
    assert db.conn.execute("SELECT COUNT(*) FROM chat_attachment_locations").fetchone()[0] == 0


def test_hash_uniquely_locates_file(db, tmp_path):
    aid = setup(db, md5=hashlib.md5(b"content").hexdigest(), size=7)
    root = tmp_path / "wechat"
    root.mkdir()
    path = root / "report.txt"
    path.write_bytes(b"content")
    assert locator(db, tmp_path, root=str(root)).locate(aid)["path"] == str(path)


def test_multiple_same_named_files_not_automatically_opened(db, tmp_path):
    aid = setup(db)
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "report.txt").write_text(name)
    assert len(locator(db, tmp_path, root=str(tmp_path)).locate(aid)["candidates"]) == 2


@pytest.mark.parametrize("path", ["relative.txt", r"\\server\secret.txt", "//server/share/file", "https://bad/file", "localmind-chat://id", "C:\\file.txt:stream"])
def test_reject_nonlocal_and_device_paths(path):
    with pytest.raises(WeFlowError):
        local_file(path)


def remote(db, aid, client):
    source = {"id": "msg-" + "a" * 64, "conversation_id": "group",
              "sent_at_unix_ms": "1792454400000", "source_type": 49, "platform_message_id": "123"}
    with db.conn:
        save_api_identities(db, "source", client.base_url, "account", dict([api_identity(source)]))
    return dict(db.conn.execute("SELECT * FROM chat_api_messages").fetchone()), source


def client():
    value = WeFlowClient(token="not-a-real-token", account="account")
    value._accounts_checked = True
    value.protocol = "alpha"
    return value


def test_rpc_exact_identity_schema_and_account(db, tmp_path):
    aid = setup(db)
    api = client()
    stored, raw = remote(db, aid, api)
    calls = []
    def request(request, **kwargs):
        call = json.loads(request.data)
        calls.append(call)
        if call["method"] == "initialize":
            result = {"capabilities": {"tools": {}}}
        elif call["method"] == "tools/list":
            result = {"tools": [{"name": RESOLVE_TOOL, "inputSchema": {
                "type": "object", "properties": {k: {"type": "string"} for k in
                ("account_id", "conversation_id", "message_id", "sent_at_unix_ms", "source_type")}}}]}
        else:
            result = {"structuredContent": {**raw, "refs": []}}
        return {"jsonrpc": "2.0", "id": call["id"], "result": result}
    api._request_json = request
    service = locator(db, tmp_path, client=api)
    assert service.resolve_remote(stored)["id"] == raw["id"]
    assert calls[-1]["params"]["arguments"]["account_id"] == "account"
    with pytest.raises(WeFlowError, match="白名单"):
        service.rpc("tools/call", {"name": "wechat.chat.send_text"})
    raw["conversation_id"] = "wrong"
    with pytest.raises(WeFlowError, match="不匹配"):
        service.resolve_remote(stored)


def test_wrong_account_cannot_resolve_attachment(db, tmp_path):
    aid = setup(db)
    api = client()
    remote(db, aid, api)
    api.account_id = "different"
    with pytest.raises(WeFlowError, match="不同"):
        locator(db, tmp_path, client=api).locate(aid)


class Response(io.BytesIO):
    def __init__(self, content, length=None):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content) if length is None else length)}


def test_download_from_loopback_is_verified_and_reused_offline(db, tmp_path):
    aid = setup(db, size=7)
    api = client()
    remote(db, aid, api)
    api.opener = Mock()
    api.opener.open.return_value = Response(b"content")
    service = locator(db, tmp_path, client=api)
    service.resolve_remote = lambda _: {"refs": [{"kind": "file", "uri": "alpha-media://sha256/" + "b" * 64}]}
    result = service.locate(aid)
    path = Path(result["path"])
    assert path.name == "report.txt" and path.read_bytes() == b"content"
    assert locator(db, tmp_path).locate(aid)["path"] == str(path)
    request = api.opener.open.call_args.args[0]
    assert request.full_url.startswith("http://127.0.0.1:5031/api/v1/media/")
    assert request.get_header("Authorization") == "Bearer not-a-real-token"
    assert not list(tmp_path.rglob("*.part"))


def test_available_local_path_is_used_without_download(db, tmp_path):
    aid = setup(db, size=7)
    api = client()
    remote(db, aid, api)
    path = tmp_path / "existing.txt"
    path.write_bytes(b"content")
    service = locator(db, tmp_path, client=api)
    service.resolve_remote = lambda _: {"local_path": str(path), "refs": []}
    assert service.locate(aid)["path"] == str(path)


@pytest.mark.parametrize("uri", ["https://example.com/file", "http://127.0.0.1:9999/api/v1/media/a",
                               "//example.com/api/v1/media/a", "/api/v1/media/a?access_token=x",
                               "/api/v1/media/proxy/a", "file:///C:/secret.txt"])
def test_remote_urls_never_receive_token(db, tmp_path, uri):
    aid = setup(db)
    api = client()
    api.opener = Mock()
    service = locator(db, tmp_path, client=api)
    with pytest.raises(WeFlowError):
        service.download(service.attachment(aid), {"uri": uri})
    api.opener.open.assert_not_called()


def test_404_materialization_reason_and_no_partial_file(db, tmp_path):
    aid = setup(db)
    api = client()
    api.opener = Mock()
    api.opener.open.side_effect = urllib.error.HTTPError("local", 404, "", {}, None)
    service = locator(db, tmp_path, client=api)
    with pytest.raises(WeFlowError, match="尚未保存"):
        service.download(service.attachment(aid), {"uri": "/api/v1/media/test"})
    assert not list(tmp_path.rglob("*.part"))


def test_truncated_download_removed(db, tmp_path):
    aid = setup(db)
    api = client()
    api.opener = Mock()
    api.opener.open.return_value = Response(b"partial", 100)
    service = locator(db, tmp_path, client=api)
    with pytest.raises(WeFlowError, match="不完整"):
        service.download(service.attachment(aid), {"uri": "/api/v1/media/test"})
    assert not list(tmp_path.rglob("*.part"))


def test_forward_card_does_not_open_first_unrelated_file(db, tmp_path):
    aid = setup(db, origin="record 1.2")
    api = client()
    remote(db, aid, api)
    service = locator(db, tmp_path, client=api)
    service.resolve_remote = lambda _: {"refs": [{"kind": "file", "uri": "/api/v1/media/test"}]}
    with pytest.raises(WeFlowError, match="合并转发"):
        service.locate(aid)


def test_cancellation_does_not_bind(db, tmp_path):
    aid = setup(db)
    with pytest.raises(SyncCancelled):
        locator(db, tmp_path, cancel=lambda: True).locate(aid)
    assert db.conn.execute("SELECT COUNT(*) FROM chat_attachment_locations").fetchone()[0] == 0


def test_legacy_index_backfills_exact_api_identity_without_reindex(db, tmp_path):
    aid = setup(db)
    api = client()
    source_key = "weflow:" + digest("\0".join([api.base_url, "owner", "group"]))
    with db.conn:
        db.conn.execute("UPDATE chat_sources SET source_key=?", (source_key,))
    api.sessions = lambda **kwargs: [{"id": "different"}, {"id": "group"}]
    api.identity = lambda *args: {"owner": "owner"}
    raw = {"id": "msg-" + "a" * 64, "platform_message_id": "123", "conversation_id": "group",
           "sent_at_unix_ms": "1792454400000", "source_type": 49}
    calls = []
    def get(route, **params):
        calls.append((route, params))
        return {"account_id": "account", "talker": "group", "messages": [raw], "hasMore": False}
    api.get = get
    before = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    service = locator(db, tmp_path, client=api)
    remote = service.remote_identity(service.attachment(aid))
    assert remote["api_id"] == raw["id"] and remote["account_id"] == "account"
    assert calls[0][1]["start"] == calls[0][1]["end"]
    assert calls[0][1]["limit"] == 200
    assert db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == before
    assert service.remote_identity(service.attachment(aid)) == remote
    assert len(calls) == 1


def test_cached_local_file_does_not_parse_stale_token_or_url(db, tmp_path):
    aid = setup(db)
    path = tmp_path / "local.txt"
    path.write_text("content")
    factory = Mock(side_effect=ValueError("must not connect"))
    service = locator(db, tmp_path, client_factory=factory)
    service.bind(aid, str(path))
    assert service.locate(aid)["path"] == str(path)
    factory.assert_not_called()


def test_clear_cache_never_deletes_manually_linked_original(db, tmp_path):
    from localmind.cache_manager import prepare_model_cache_dir, clear_model_cache
    aid = setup(db)
    path = tmp_path / "original.txt"
    path.write_text("original")
    cache = prepare_model_cache_dir(tmp_path / "cache")
    service = AttachmentLocator(db, cache_dir=cache)
    service.bind(aid, str(path))
    clear_model_cache(cache)
    assert path.read_text() == "original"
    assert service.locate(aid)["path"] == str(path)


def test_later_content_index_connects_previously_confirmed_path(db, tmp_path):
    from test_weflow_v04 import index_file, searcher
    from localmind.models import SearchQuery
    aid = setup(db)
    path = tmp_path / "different-name.txt"
    path.write_text("signed content")
    locator(db, tmp_path).bind(aid, str(path))
    target = index_file(db, path, "signed content")
    ChatStore(db).resolve_links()
    assert ChatStore(db).attachments()[0]["file_id"] == target["id"]
    results = searcher(db, mode="keyword").search(SearchQuery("signed content", scope="chat_files"))
    assert any(result.path == str(path) for result in results)


def test_sync_persists_exact_api_id_and_display_file_size(db):
    from test_weflow_alpha import alpha_client
    from localmind.weflow import sync_session
    api = alpha_client()
    row = api.opener.rows[0]
    row.update(id="msg-" + "c" * 64, text="[文件] example.pdf",
               display={"kind": "file", "fileName": "example.pdf", "fileSize": "1234"})
    sync_session(db, api, "test-chat", start=1780000000, end=1780001000)
    saved = dict(db.conn.execute("SELECT * FROM chat_api_messages").fetchone())
    assert saved["api_id"] == row["id"] and saved["account_id"] == "test-account"
    assert ChatStore(db).attachments()[0]["size"] == 1234


def test_multiple_resources_reject_even_when_outer_local_path_present(db, tmp_path):
    aid = setup(db)
    api = client()
    remote(db, aid, api)
    path = tmp_path / "report.txt"
    path.write_text("content")
    service = locator(db, tmp_path, client=api)
    service.resolve_remote = lambda _: {"local_path": str(path), "refs": [
        {"kind": "file", "uri": "/api/v1/media/a"}, {"kind": "file", "uri": "/api/v1/media/b"}]}
    with pytest.raises(WeFlowError, match="多个文件"):
        service.locate(aid)


def test_conflicting_exact_message_references_stop_sync(db):
    from test_weflow_alpha import alpha_client
    from localmind.weflow import sync_session
    api = alpha_client()
    first = {**api.opener.rows[0], "id": "msg-" + "a" * 64}
    api.opener.rows = [first, {**first, "id": "msg-" + "b" * 64}]
    with pytest.raises(WeFlowError, match="不同的 API"):
        sync_session(db, api, "test-chat", start=1780000000, end=1780001000)
    assert not ChatStore(db).sources()
