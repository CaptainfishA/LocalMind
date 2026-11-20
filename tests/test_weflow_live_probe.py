from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from localmind.weflow import WeFlowClient, WeFlowError
from test_weflow_alpha import alpha_message


spec = importlib.util.spec_from_file_location("live_probe", Path(__file__).parents[1] / "benchmarks" / "verify_weflow_live.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def sample_client(rows, **extra):
    client = WeFlowClient(account="test-account")
    client.protocol = "alpha"
    calls = []

    def get(path, **params):
        calls.append((path, params))
        return {"account_id": "test-account", "talker": "test-chat", "messages": rows, "hasMore": True, **extra}

    client.get = get
    return client, calls


def test_recent_sample_requests_descending_not_default_history():
    client, calls = sample_client([alpha_message(2, sent_at_unix_ms="1780000002000"), alpha_message(1)])
    payload, timestamps = probe.recent_sample(client, "test-chat")
    assert len(payload["messages"]) == 2 and timestamps[0] > timestamps[1]
    assert calls == [("/api/v1/messages", {"talker": "test-chat", "limit": 200, "offset": 0, "media": "0", "order": "desc"})]


def test_recent_sample_rejects_ignored_sort_parameter():
    client, _ = sample_client([alpha_message(1), alpha_message(2, sent_at_unix_ms=1780000002000)])
    with pytest.raises(WeFlowError, match="倒序"):
        probe.recent_sample(client, "test-chat")


@pytest.mark.parametrize("extra", [{"account_id": "another"}, {"talker": "another"}, {"hasMore": "false"}, {"messages": {}}])
def test_recent_sample_validates_response_scope(extra):
    client, _ = sample_client([alpha_message(1)], **extra)
    with pytest.raises(WeFlowError):
        probe.recent_sample(client, "test-chat")


def test_recent_sample_accepts_empty_and_same_second_rows():
    client, _ = sample_client([], hasMore=False)
    assert probe.recent_sample(client, "test-chat")[1] == []
    client, _ = sample_client([alpha_message(2), alpha_message(1)])
    assert len(probe.recent_sample(client, "test-chat")[1]) == 2


def test_xml_file_diagnostics_are_structural_and_redacted():
    filename = "private-attachment.docx"
    raw = f"<msg><appmsg><type>6</type><title>{filename}</title><appattach><totallen>123</totallen></appattach></appmsg></msg>"
    row = alpha_message(1, source_type=25769803825, kind="xml", text=raw)
    result = probe.attachment_diagnostics([row])
    group = result["source_types"]["25769803825"]
    assert group["appmsg_type_6"] == group["recognized_explicit"] == 1
    assert group["recognized_attachments"] == 1
    assert result["descending_timestamps"] is True
    encoded = json.dumps(result)
    assert filename not in encoded and row["id"] not in encoded and row["sender_id"] not in encoded
    # A missing [文件] prefix must not imply that the sample has no attachments.
    assert not raw.startswith("[文件]")


def test_diagnostics_report_xml_failures_instead_of_absence():
    result = probe.attachment_diagnostics([alpha_message(1, source_type=49, text="<msg><appmsg>", kind="xml")])
    assert result["source_types"]["49"]["malformed_xml"] == 1
    assert result["source_types"]["49"]["recognized_attachments"] == 0
