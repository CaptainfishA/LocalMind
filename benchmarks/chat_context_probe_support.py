"""Bounded sample loading; never mistake unsupported ordering for an empty chat."""

import time

from localmind.wechat_files import api_identity
from localmind.weflow import WeFlowError


def sample(client, session, diagnostics, days=7):
    last = session.get("last_activity")
    end = int(last) + 1 if type(last) is int and last > 0 else int(time.time())
    start = max(0, end - days * 86400)
    payload = client.get("/api/v1/messages", talker=session["id"], start=start,
                         end=end, limit=200, offset=0, media="0", order="desc")
    diagnostics.update(response_keys=sorted(payload),
                       field_types={k: type(payload.get(k)).__name__
                                    for k in ("messages", "items", "hasMore", "scanComplete")},
                       window_days=days, session_activity_known=type(last) is int)
    client._check_response_account(payload)
    rows = payload.get("messages")
    if payload.get("talker") != session["id"] or not isinstance(rows, list) or len(rows) > 200:
        raise WeFlowError("消息列表结构或会话身份不兼容", code="sample_shape")
    diagnostics["sample_count"] = len(rows)
    diagnostics["message_keys"] = sorted({k for row in rows[:2] if isinstance(row, dict) for k in row})
    if any(not isinstance(row, dict) or row.get("conversation_id") != session["id"]
           or row.get("account_id", client.account_id) != client.account_id for row in rows):
        raise WeFlowError("消息列表中存在不同账号或会话，已停止读取", code="sample_identity")
    usable = [row for row in rows if api_identity(row)]
    stamps = [int(row["sent_at_unix_ms"]) for row in usable]
    diagnostics.update(selectable_count=len(usable),
                       descending=all(a >= b for a, b in zip(stamps, stamps[1:])),
                       incomplete=payload.get("hasMore") is not False or payload.get("scanComplete") is not True)
    # Ordering is a presentation detail, not an identity check. Do not claim the
    # first page is the latest if Alpha ignored order=desc.
    usable.sort(key=lambda row: int(row["sent_at_unix_ms"]), reverse=True)
    return usable
