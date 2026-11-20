import importlib.util
from pathlib import Path

import pytest

from test_alpha_chat_context import Client, MID, STAMP
from localmind.weflow import WeFlowError


spec = importlib.util.spec_from_file_location("chat_context_probe_support", Path(__file__).parents[1] / "benchmarks/chat_context_probe_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_sample_accepts_ignored_order_and_sorts_only_selected_page():
    client = Client()
    row = client.http["messages"][0]
    client.http["messages"].append({**row, "id": "msg-" + "2" * 64, "sent_at_unix_ms": str(STAMP + 1000)})
    diagnostics = {}
    rows = support.sample(client, {"id": "chat", "last_activity": STAMP // 1000 + 10}, diagnostics)
    assert len(rows) == 2 and rows[-1]["id"] == MID
    assert diagnostics["descending"] is False
    assert "synthetic-token" not in str(diagnostics) and "合成上下文" not in str(diagnostics)
    assert client.calls[-1]["params"]["end"] - client.calls[-1]["params"]["start"] == 7 * 86400


def test_sample_rejects_cross_conversation_and_reports_shape_only():
    client = Client()
    client.http["messages"][0]["conversation_id"] = "other"
    diagnostics = {}
    with pytest.raises(WeFlowError, match="不同账号或会话"):
        support.sample(client, {"id": "chat"}, diagnostics)
    assert diagnostics["sample_count"] == 1


def test_sample_without_usable_id_is_not_misreported_as_no_messages():
    client = Client()
    client.http["messages"][0]["id"] = "unknown-id"
    diagnostics = {}
    assert support.sample(client, {"id": "chat"}, diagnostics) == []
    assert diagnostics["sample_count"] == 1 and diagnostics["selectable_count"] == 0
