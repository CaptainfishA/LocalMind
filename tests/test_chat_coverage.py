import pytest

from localmind.chats import ChatStore
from localmind.weflow import sync_session
from test_chat_timeline import corpus
from test_weflow_alpha import alpha_client
from test_weflow_batch import START, END


def test_coverage_records_actual_retry_and_recovery_without_certifying_all_history(corpus):
    client = alpha_client()
    client.opener.rows[0]["text"] = {"unsupported": True}
    sync_session(corpus, client, "test-chat", start=START, end=END)
    row = corpus.conn.execute("SELECT * FROM chat_coverage ORDER BY id DESC LIMIT 1").fetchone()
    assert row["start_epoch"] == START and row["end_epoch"] == END
    assert row["state"] == "partial" and row["issue_count"] == 1
    client.opener.rows[0]["text"] = "修复后的合成消息"
    sync_session(corpus, client, "test-chat", start=START, end=END + 1000)
    row = corpus.conn.execute("SELECT * FROM chat_coverage ORDER BY id DESC LIMIT 1").fetchone()
    assert row["state"] == "completed" and row["issue_count"] == 0
    ChatStore(corpus).remove_source(row["source_id"])
    assert not corpus.conn.execute("SELECT 1 FROM chat_coverage").fetchone()


def test_failed_sync_does_not_claim_completed_range(corpus):
    client = alpha_client()
    client.opener.rows[0]["sent_at_unix_ms"] = "invalid"
    with pytest.raises(Exception):
        sync_session(corpus, client, "test-chat", start=START, end=END)
    assert not corpus.conn.execute("SELECT 1 FROM chat_coverage").fetchone()
