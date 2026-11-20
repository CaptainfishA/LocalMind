from __future__ import annotations

import pytest
import io
import json
from urllib.parse import parse_qs, urlsplit

from localmind.db import LocalMindDB
from localmind.embeddings import EmbeddingService
from localmind.weflow import WeFlowClient, WeFlowError, _api_error, decode_messages, sync_session


@pytest.mark.parametrize("status,code,expected", [
    (401, "HTTP_API_UNAUTHORIZED", "鉴权失败"),
    (423, "HTTP_API_OPERATION_BLOCKED", "应用锁"),
    (504, "HTTP_API_REQUEST_TIMEOUT", "查询超时"),
    (404, "HTTP_API_ROUTE_NOT_FOUND", "不支持此接口"),
    (403, "HTTP_API_HOST_DENIED", "本机监听地址"),
])
def test_alpha_error_diagnostics(status, code, expected):
    error = _api_error(status, {"error": {"code": code, "message": "private chat text must not appear"}})
    assert expected in str(error)
    assert error.status == status and error.code == code
    assert "private chat" not in str(error)


def test_alpha_query_timeout_and_token_validation():
    assert WeFlowClient().timeout == 150
    with pytest.raises(WeFlowError, match="ASCII"):
        WeFlowClient(token="含有中文的凭证")


def alpha_message(number, **extra):
    return {"id": f"native-{number}", "platform_message_id": str(9000000000000000000 + number),
            "server_id": 9000000000000000000 + number, "local_id": number,
            "conversation_id": "test-chat", "sender_id": "test-member", "source_type": 1,
            "sent_at_unix_ms": 1780000000123 + number, "text": f"测试文档说明{number}",
            "kind": "text", "direction": 1, "refs": [], **extra}


class AlphaTransport:
    """Redacted Alpha 0.1.1 response shapes, synthetic content only."""
    def __init__(self):
        self.calls = []
        self.response_account = "test-account"
        self.selected_account = "test-account"
        self.ready = True
        self.has_more = True
        self.next_offset = 7
        self.rows = [alpha_message(i) for i in range(3)]

    def open(self, request, timeout):
        url = urlsplit(request.full_url)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        self.calls.append((url.path, query))
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer synthetic-token"
        if url.path not in {"/health", "/api/v1/accounts"}:
            assert query["account"] == "test-account"
        if url.path == "/health":
            value = {"status": "ok", "scope": "gateway_liveness", "database_open_reported": True}
        elif url.path == "/api/v1/accounts":
            value = {"accounts": [{"id": "test-account", "display_name": "测试账号", "ready": self.ready,
                                    "enabled": True, "selected": False}], "selected_account_id": self.selected_account}
        elif url.path == "/api/v1/sessions":
            offset = int(query["offset"])
            value = {"items": [{"id": "test-chat" if offset == 0 else "second-chat", "display_name": "测试群"}],
                     "account_id": self.response_account, "hasMore": self.has_more if offset == 0 else False,
                     "nextOffset": self.next_offset, "scanComplete": True}
        elif url.path.startswith("/api/v1/sessions/"):
            assert "since" not in query and query["limit"] == "1"
            value = {"meta": {"ownerId": "test-owner", "name": "测试群"},
                     "members": [{"platformId": "test-member", "accountName": "测试成员"}],
                     "messages": [], "sync": {"hasMore": False}}
        elif url.path == "/api/v1/messages":
            offset = int(query["offset"])
            value = {"account_id": self.response_account, "talker": "test-chat",
                     "messages": self.rows[:2] if offset == 0 else self.rows[2:],
                     "hasMore": offset == 0, "nextOffset": self.next_offset}
        else:
            raise AssertionError("unexpected route")
        return io.BytesIO(json.dumps(value).encode())


def alpha_client(**kwargs):
    client = WeFlowClient(token="synthetic-token", **kwargs)
    client.opener = AlphaTransport()
    return client


def test_alpha_fields_millisecond_time_and_stable_ids():
    msg = decode_messages([alpha_message(1)], {"test-member": "测试成员"})[0]
    assert msg.external_id == "9000000000000000001"
    assert msg.body == "测试文档说明1" and msg.sender == "测试成员"
    assert decode_messages([alpha_message(1, sender_name="显示昵称")])[0].sender == "显示昵称"
    assert msg.sent_at.startswith("2026-05-")
    pending = decode_messages([alpha_message(1, platform_message_id="", server_id=0)])[0]
    assert pending.external_id == "alpha:native-1"
    file = decode_messages([alpha_message(2, text="[文件] 设计附件.pdf", kind="file")])[0]
    assert file.attachments[0].name == "设计附件.pdf"
    with pytest.raises(WeFlowError, match="毫秒"):
        decode_messages([alpha_message(1, sent_at_unix_ms="1780000000123.5")])


def test_alpha_accounts_and_session_cursor():
    client = alpha_client()
    assert len(client.sessions()) == 2
    assert client.protocol == "alpha" and client.account_id == "test-account"
    assert client.available_accounts[0]["name"] == "测试账号"
    calls = [q for p, q in client.opener.calls if p == "/api/v1/sessions"]
    assert [q["offset"] for q in calls] == ["0", "7"]


@pytest.mark.parametrize("setting,value,expected", [
    ("response_account", "different-account", "不同账号"),
    ("ready", False, "未就绪"),
    ("next_offset", 0, "未推进"),
    ("has_more", "false", "分页标记"),
])
def test_alpha_fail_closed(setting, value, expected):
    client = alpha_client()
    setattr(client.opener, setting, value)
    with pytest.raises(WeFlowError, match=expected):
        client.sessions()


def test_alpha_explicit_account_overrides_default():
    client = alpha_client(account="test-account")
    client.opener.selected_account = "another-default"
    assert client.sessions()
    with pytest.raises(WeFlowError, match="未就绪"):
        alpha_client(account="unknown-account").accounts()


def test_alpha_sync_cursor_dedup_and_rollback(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    client = alpha_client()
    embedding = EmbeddingService(force_hash=True)
    try:
        result = sync_session(db, client, "test-chat", start=1779999999, end=1780000010, embedding_service=embedding)
        assert result["received"] == 3 and result["total"] == 3
        repeat = sync_session(db, client, "test-chat", start=1779999999, end=1780000010, embedding_service=embedding)
        assert repeat["updated_units"] == 0
        pulls = [q for p, q in client.opener.calls if p.startswith("/api/v1/sessions/")]
        assert len(pulls) == 2  # One metadata request per Alpha sync, not three.
        checkpoint = db.conn.execute("SELECT value FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()[0]
        client.opener.response_account = "other-account"
        with pytest.raises(WeFlowError, match="不同账号"):
            sync_session(db, client, "test-chat", start=1779999999, end=1780000020, embedding_service=embedding)
        assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 3
        assert db.conn.execute("SELECT value FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchone()[0] == checkpoint
    finally:
        db.close()


def test_alpha_wrong_conversation_is_rejected(tmp_path):
    db = LocalMindDB(tmp_path / "index.db")
    client = alpha_client()
    client.opener.rows[0]["conversation_id"] = "different-chat"
    try:
        with pytest.raises(WeFlowError, match="不同会话"):
            sync_session(db, client, "test-chat", start=1779999999, end=1780000010)
        assert db.conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 0
        assert not db.conn.execute("SELECT * FROM settings WHERE key LIKE 'weflow_checkpoint:%'").fetchall()
    finally:
        db.close()


def test_alpha_xml_file_and_virtual_media_references():
    xml = "<msg><appmsg><type>6</type><title>最终方案.pdf</title><appattach><totallen>123</totallen></appattach></appmsg></msg>"
    msg = decode_messages([alpha_message(1, kind="xml", text=xml)])[0]
    assert msg.attachments[0].name == "最终方案.pdf" and msg.attachments[0].size == 123
    assert msg.body == "[文件] 最终方案.pdf"
    media = decode_messages([alpha_message(2, kind="media", text="", refs=[
        {"kind": "image", "mime_type": "image/jpeg", "id": "synthetic-ref", "uri": "alpha-media://synthetic-ref"}])])[0]
    assert not media.attachments  # A virtual media resource is not a disk file.
    issues = []
    link = decode_messages([alpha_message(3, kind="xml", text='<msg><img url="https://example.invalid/logo.png"/></msg>')], issues=issues)
    assert not link and issues[0]["code"] == "structured_unsupported"
