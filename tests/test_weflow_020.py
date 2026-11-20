from __future__ import annotations

import io
import http.client
import json
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from localmind.weflow import WeFlowClient, WeFlowError, decode_messages
from test_weflow_alpha import AlphaTransport, alpha_message
from localmind.weflow_protocol import integer, session_page


class Alpha020Transport(AlphaTransport):
    """Public Alpha 0.2.0 limits, with only synthetic account/message data."""
    def open(self, request, timeout):
        url = urlsplit(request.full_url)
        query = parse_qs(url.query)
        if url.path == "/api/v1/sessions" and int(query["limit"][0]) > 200:
            raise urllib.error.HTTPError(request.full_url, 400, "bad request", {},
                                         io.BytesIO(b'{"error":{"code":"INVALID_LIMIT"}}'))
        return super().open(request, timeout)


def test_alpha020_session_limit():
    client = WeFlowClient(token="synthetic-token")
    client.opener = Alpha020Transport()
    assert len(client.sessions()) == 2
    assert all(int(query["limit"]) <= 200 for path, query in client.opener.calls if path == "/api/v1/sessions")


def test_wire_long_integer_strings():
    rows = [alpha_message(1, sent_at_unix_ms="1780000000124", platform_message_id="9000000000000000001")]
    result = decode_messages(rows)[0]
    assert result == decode_messages([alpha_message(1)])[0]
    assert result.external_id == "9000000000000000001"
    page = session_page({"items": [], "hasMore": True, "nextOffset": "200"}, 0, alpha=True)
    assert page.next_offset == 200


@pytest.mark.parametrize("value", [True, 1.5, "1e6", " 123", "-1", "１２３", "9" * 21, None])
def test_malformed_integer_rejected(value):
    with pytest.raises(WeFlowError):
        integer(value, "测试字段")


def test_float_message_id_rejected():
    with pytest.raises(WeFlowError, match="精度"):
        decode_messages([alpha_message(1, platform_message_id=9e18)])


@pytest.mark.parametrize("payload", [
    {"items": [], "nextOffset": 200},
    {"items": [], "hasMore": True},
    {"items": [], "hasMore": False, "scanComplete": False},
    {"items": {}, "hasMore": False},
])
def test_alpha_incomplete_pages_rejected(payload):
    with pytest.raises(WeFlowError):
        session_page(payload, 0, alpha=True)


@pytest.mark.parametrize("path", ["/api/v1/messages/send-text", "/api/v1/exports", "/api/v1/media/example",
                                  "/api/v1/push/messages", "/health?access_token=secret", "/health/other"])
def test_client_read_only_routes(path):
    with pytest.raises(WeFlowError, match="只读"):
        WeFlowClient().get(path)


def test_request_account_and_token_invariants():
    client = WeFlowClient(account="selected")
    with pytest.raises(WeFlowError, match="账号"):
        client.get("/api/v1/messages", account="other")
    with pytest.raises(WeFlowError, match="请求头"):
        client.get("/api/v1/messages", access_token="must-not-be-sent")


class ResponseTransport:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def open(self, *_args, **_kwargs):
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.value).encode())


@pytest.mark.parametrize("payload", [{"status": "error", "code": "READ_LIMIT_EXCEEDED"},
                                     {"success": False}, {"error": {"code": "INVALID_LIMIT", "message": "private-data"}}])
def test_http200_failure_is_not_empty_success(payload):
    client = WeFlowClient()
    client.opener = ResponseTransport(payload)
    with pytest.raises(WeFlowError, match="失败状态") as error:
        client.get("/api/v1/messages")
    assert "private-data" not in str(error.value)


def test_truncated_response_is_sanitized():
    client = WeFlowClient()
    client.opener = ResponseTransport(error=http.client.IncompleteRead(b"private-message", 200))
    with pytest.raises(WeFlowError, match="传输中断") as error:
        client.get("/api/v1/messages")
    assert "private-message" not in str(error.value)
