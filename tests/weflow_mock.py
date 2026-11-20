"""Synthetic WeFlow HTTP contract fixture. Contains no real account data."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


def message(identity, text="模拟消息", timestamp=1780000000, **extra):
    return {"serverId": str(identity), "localId": identity, "localType": 1,
            "createTime": timestamp, "senderUsername": "mock_member", "isSend": 0,
            "content": text, "rawContent": text, **extra}


class MockWeFlow:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []
        self.owner = "mock_owner"
        self.session = "mock_project@chatroom"
        self.token = "synthetic-test-token"
        self.fail_offset = None
        self.repeat_page = False
        self.switch_owner = False
        self.redirect = False
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                url = urlsplit(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                state.calls.append((url.path, query))
                if state.redirect:
                    self.send_response(302)
                    self.send_header("Location", "http://example.invalid/no-token-here")
                    self.end_headers()
                    return
                if url.path != "/health" and self.headers.get("Authorization") != "Bearer " + state.token:
                    self.send_response(401)
                    self.end_headers()
                    return
                status = 200
                if url.path == "/health":
                    value = {"status": "ok"}
                elif url.path == "/api/v1/sessions":
                    value = {"success": True, "sessions": [{"username": state.session, "displayName": "模拟项目群"}]}
                elif url.path == "/api/v1/group-members":
                    value = {"success": True, "members": [{"wxid": "mock_member", "groupNickname": "张工"}]}
                elif url.path.startswith("/api/v1/sessions/"):
                    value = {"meta": {"ownerId": state.owner, "name": "模拟项目群", "platform": "im"}, "members": []}
                elif url.path == "/api/v1/messages":
                    offset, limit = int(query["offset"]), int(query["limit"])
                    rows = sorted([r for r in state.rows if int(query["start"]) <= r["createTime"] <= int(query["end"])], key=lambda r: (r["createTime"], str(r["serverId"])), reverse=True)
                    page = rows[:limit] if state.repeat_page else rows[offset:offset + limit]
                    value = {"success": True, "talker": query["talker"], "messages": page,
                             "hasMore": state.repeat_page or offset + limit < len(rows)}
                    if state.fail_offset is not None and offset >= state.fail_offset:
                        status = 500
                        value = {"success": False}
                    if state.switch_owner:
                        state.owner = "different_mock_owner"
                else:
                    status, value = 404, {"success": False}
                body = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
