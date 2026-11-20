from collections import Counter

import pytest

from localmind.db import LocalMindDB


@pytest.fixture
def corpus(tmp_path):
    db = LocalMindDB(tmp_path / "fts-scope.db")
    db.conn.executemany(
        "INSERT INTO files(id,path,name,ext,size,created_at,modified_at,"
        "scan_status,parse_status,ocr_status) "
        "VALUES(?,?,?,'txt',0,'2026-01-01','2026-01-01','indexed','parsed','not_required')",
        [(i, f"/synthetic/{i}.txt", f"{i}.txt") for i in range(1, 101)],
    )
    rows = [(i, (i - 1) // 10 + 1, "maintenance report " * (i % 3 + 1)
             if i % 5 else "unrelated text") for i in range(1, 1001)]
    db.conn.executemany(
        "INSERT INTO chunks(id,file_id,chunk_text,source_type,created_at) "
        "VALUES(?,?,?,'text','2026-01-01')", rows,
    )
    db.conn.executemany(
        "INSERT INTO chunk_fts(rowid,file_name,file_path,chunk_text,source_type) "
        "VALUES(?,'synthetic','/synthetic',?,'text')",
        [(cid, text) for cid, _, text in rows],
    )
    db.conn.commit()
    yield db
    db.close()


@pytest.mark.parametrize("target_mode", [None, "files", "messages"])
@pytest.mark.parametrize("scope", [set(), {1}, {5, 10}, set(range(1, 30)),
                                 set(range(1, 1001, 2)), set(range(1, 1002))])
def test_restricted_fts_matches_filtered_global_ranking(corpus, target_mode, scope):
    all_rows = corpus.search_fts('"maintenance"', limit=2000)
    groups = corpus.candidate_groups(target_mode=target_mode or "files")
    counts = Counter()
    expected = []
    for row in sorted(all_rows, key=lambda row: (row["rank"], row["chunk_id"])):
        cid = row["chunk_id"]
        if cid not in scope or (target_mode and counts[groups[cid]] >= 4):
            continue
        counts[groups[cid]] += 1
        expected.append((cid, row["rank"]))
    corpus.set_search_scope(scope)
    actual = corpus.search_fts('"maintenance"', limit=2000, restricted=True, target_mode=target_mode)
    assert sorted((r["chunk_id"], r["rank"]) for r in actual) == sorted(expected)
    for limit in (1, 7):
        limited = corpus.search_fts('"maintenance"', limit=limit, restricted=True, target_mode=target_mode)
        assert len(limited) == min(limit, len(expected))
        assert all(r["chunk_id"] in scope for r in limited)
        assert [r["rank"] for r in limited] == [score for _, score in expected[:limit]]


@pytest.mark.parametrize("target_mode", [None, "files", "messages"])
def test_large_scope_does_not_push_rowid_equality_into_fts(corpus, target_mode):
    corpus.set_search_scope(set(range(1, 1001)))
    statements = []
    corpus.conn.set_trace_callback(statements.append)
    try:
        corpus.search_fts('"maintenance"', restricted=True, target_mode=target_mode)
    finally:
        corpus.conn.set_trace_callback(None)
    sql = next(s for s in statements if "FROM chunk_fts WHERE" in " ".join(s.split()))
    plan = corpus.conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    fts_steps = [r["detail"] for r in plan if "SCAN chunk_fts VIRTUAL TABLE" in r["detail"]]
    assert fts_steps
    # FTS5's '=' constraint repeats a term scan per scope row; MATCH alone does not.
    assert all("=" not in step.split("INDEX", 1)[1] for step in fts_steps)
