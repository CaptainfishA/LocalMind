"""Read-only, identity-checked discussion windows from indexed messages."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
import sqlite3
import math

from .chats import chat_source_reference, forward_items
from .models import ChatSourceReference
from .search_filters import chat_predicates, epoch, where
from .ui_formatting import display_time


class TimelineError(ValueError):
    pass


@dataclass(frozen=True)
class Discussion:
    key: tuple
    title: str
    conversation: str
    account: str
    refs: tuple[ChatSourceReference, ...]
    hit_ids: tuple[int, ...]
    anchor_id: int
    root_id: int
    sent_at: str | None
    components: tuple = ()
    reason: str = "引用 / 回复关系"


@dataclass(frozen=True)
class TimelinePage:
    rows: tuple[dict, ...]
    anchor_id: int
    previous: bool = False
    following: bool = False
    limited: bool = False
    notice: str = "已索引记录 · 未同步或解析失败的消息可能缺失"
    before_cursor: tuple | None = None
    after_cursor: tuple | None = None
    page_index: int = 0


@contextmanager
def snapshot(path, cancelled=lambda: False):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=.3)
    conn.row_factory = sqlite3.Row
    conn.create_function("localmind_epoch", 1, epoch)
    conn.set_progress_handler(lambda: int(cancelled()), 500)
    try:
        if cancelled():
            raise TimelineError("读取已取消")
        conn.execute("BEGIN")
        yield conn
    finally:
        conn.close()


def _conditions(query, *, neighbors=False):
    if query is None:
        return [], []
    chat_dates = query.scope in {"chat_files", "messages"} or query.chat_platform or query.conversation or query.sender
    return chat_predicates(replace(query, sender=None) if neighbors else query, include_dates=bool(chat_dates))


def _validate(conn, result, ref):
    conditions, params = _conditions(result.source_query)
    row = conn.execute("""SELECT m.*,m.id AS message_id,s.platform,s.account,s.source_key,s.conversation,
        a.id AS attachment_id,a.name AS attachment_name,a.origin,a.file_id,f.path
        FROM chat_messages m JOIN chat_sources s ON s.id=m.source_id
        LEFT JOIN chat_attachments a ON a.message_id=m.id AND a.id=?
        LEFT JOIN files f ON f.id=a.file_id WHERE m.id=?""" + where(conditions),
        [ref.attachment_id, ref.message_id, *params]).fetchone()
    if row is None or chat_source_reference(dict(row)).identity != ref.identity:
        raise TimelineError("聊天来源已变化，请重新搜索")
    if result.result_kind == "file" and (row["file_id"] != result.file_id or row["path"] != result.path):
        raise TimelineError("文件关联已变化，请重新搜索")
    if result.result_kind == "attachment_pending" and row["file_id"] is not None:
        raise TimelineError("附件已建立新的关联，请重新搜索")
    return dict(row)


def _root(conn, row):
    visited = {row["id"]}
    current = row
    for _ in range(16):
        if not current["reply_to"]:
            return current
        parent = conn.execute("SELECT * FROM chat_messages WHERE source_id=? AND external_id=?",
                              (row["source_id"], current["reply_to"])).fetchone()
        if parent is None:
            return current
        if parent["id"] in visited:
            return row
        visited.add(parent["id"])
        current = dict(parent)
    # A truncated ancestry is not proof of a shared discussion root.
    return row


def discussions(path, result, peers=(), *, cancelled=lambda: False):
    refs, hits = {}, set()
    selected = result.chat_source
    if selected is None:
        return (), False
    with snapshot(path, cancelled) as conn:
        _validate(conn, result, selected)
        candidates = peers if result.result_kind == "message" else (result,)
        for hit in candidates[:200]:
            if result.result_kind == "message" and hit.result_kind != "message":
                continue
            for ref in [hit.chat_source, *(p.chat_source for p in hit.supporting_evidence if p.source_type == "chat")]:
                if ref is not None:
                    try:
                        _validate(conn, hit, ref)
                    except TimelineError:
                        continue
                    refs[ref.identity] = ref
                    if hit.source_type == "chat" or any(p.chat_source == ref for p in hit.supporting_evidence):
                        hits.add(ref.message_id)
        refs.setdefault(selected.identity, selected)
        if result.source_type == "chat":
            hits.add(selected.message_id)
        more = len(peers) > 200
        if result.result_kind == "file":
            clauses, params = _conditions(result.source_query)
            rows = conn.execute("""SELECT s.platform,s.account,s.source_key,s.conversation,m.external_id,
                m.id AS message_id,m.sender,m.sent_at,a.id AS attachment_id,a.name AS attachment_name,a.origin
                FROM chat_attachments a JOIN chat_messages m ON m.id=a.message_id
                JOIN chat_sources s ON s.id=m.source_id WHERE a.file_id=?""" + where(clauses) +
                " ORDER BY m.sent_epoch DESC,m.id DESC,a.id DESC LIMIT 101", [result.file_id, *params]).fetchall()
            more = len(rows) > 100
            for row in rows[:100]:
                ref = chat_source_reference(dict(row))
                refs.setdefault(ref.identity, ref)
        groups = {}
        for ref in refs.values():
            row = conn.execute("SELECT * FROM chat_messages WHERE id=?", (ref.message_id,)).fetchone()
            if row is None:
                continue
            row = dict(row)
            root = _root(conn, row)
            key = (ref.platform, ref.account, ref.source_key, root["external_id"])
            entry = groups.setdefault(key, (root, [], []))
            entry[1].append(ref)
            if ref.message_id in hits and ref.message_id not in entry[2]:
                entry[2].append(ref.message_id)
        output = []
        for key, (root, members, matches) in groups.items():
            anchor = next((r for r in members if r == selected), members[0])
            head = conn.execute("SELECT * FROM chat_messages WHERE id=?", (anchor.message_id,)).fetchone()
            title = head["display_body"] if head["display_body"] is not None else head["body"]
            title = " ".join(title.split())[:70] or anchor.attachment_name or "非文本消息"
            output.append(Discussion(key, title, anchor.conversation, anchor.account, tuple(members),
                                     tuple(matches), anchor.message_id, root["id"], anchor.sent_at))
        return tuple(output), more


def _rows(conn, source_id, query, *, extra="", params=(), limit=301, order="ASC"):
    clauses, values = _conditions(query, neighbors=True)
    return [dict(r) for r in conn.execute("""SELECT m.* FROM chat_messages m
        JOIN chat_sources s ON s.id=m.source_id WHERE m.source_id=?""" + where(clauses) + extra +
        f" ORDER BY m.sent_epoch {order},m.id {order} LIMIT ?", [source_id, *values, *params, limit])]


def window(path, result, discussion, *, nearby=False, page=0, anchor_id=None, cursor=None, cancelled=lambda: False):
    if type(page) is not int or not -500 <= page <= 500:
        raise TimelineError("分页范围无效")
    with snapshot(path, cancelled) as conn:
        anchors = [_validate(conn, result, ref) for ref in discussion.refs]
        seed = next((r for r in anchors if r["id"] == discussion.anchor_id), None)
        roots = {g.root_id for g in discussion.components} if discussion.components else {discussion.root_id}
        if seed is None or {_root(conn, r)["id"] for r in anchors} != roots:
            raise TimelineError("讨论关系已变化，请重新搜索")
        source_id = seed["source_id"]
        if any(r["source_id"] != source_id for r in anchors):
            raise TimelineError("讨论包含不同会话的消息")
        actual_anchor = anchor_id or discussion.anchor_id
        if actual_anchor not in {r["id"] for r in anchors}:
            raise TimelineError("未知的消息锚点")
        seed = next(r for r in anchors if r["id"] == actual_anchor)
        limited = False
        if nearby and seed["sent_epoch"] is not None:
            # Seek by timestamp and identity; each subsequent page reads at most 41 rows.
            if cursor is not None:
                if not isinstance(cursor, tuple) or len(cursor) != 3 or cursor[0] not in (-1, 1):
                    raise TimelineError("分页游标无效")
                direction, stamp, mid = cursor
                if type(stamp) not in (float, int) or not math.isfinite(stamp) or type(mid) is not int or mid <= 0:
                    raise TimelineError("分页游标无效")
                values = _rows(conn, source_id, result.source_query,
                    extra=" AND (m.sent_epoch,m.id)" + ("<" if direction < 0 else ">") + "(?,?)",
                    params=(stamp, mid), limit=41, order="DESC" if direction < 0 else "ASC")
                rows = list(reversed(values[:40])) if direction < 0 else values[:40]
                previous, following = (len(values) > 40, True) if direction < 0 else (True, len(values) > 40)
            elif page:
                raise TimelineError("附近消息需要有效的分页游标")
            else:
                before = _rows(conn, source_id, result.source_query,
                    extra=" AND (m.sent_epoch,m.id)<(?,?)", params=(seed["sent_epoch"], seed["id"]),
                    limit=21, order="DESC")
                after = _rows(conn, source_id, result.source_query,
                    extra=" AND (m.sent_epoch,m.id)>=(?,?)", params=(seed["sent_epoch"], seed["id"]), limit=22)
                rows = list(reversed(before[:20])) + after[:21]
                previous, following = len(before) > 20, len(after) > 21
        elif nearby:
            rows, previous, following = [seed], False, False
        else:
            found = {r["id"]: r for r in anchors}
            root_rows = _rows(conn, source_id, result.source_query,
                             extra=" AND m.id IN (" + ",".join("?" for _ in roots) + ")", params=tuple(roots))
            found.update((r["id"], r) for r in root_rows)
            frontier = list(found.values())
            for _ in range(16):
                if not frontier or len(found) >= 300:
                    break
                children = _rows(conn, source_id, result.source_query,
                    extra=" AND m.reply_to IN (" + ",".join("?" for _ in frontier) + ")",
                    params=tuple(r["external_id"] for r in frontier), limit=301)
                frontier = [r for r in children if r["id"] not in found]
                remaining = 300 - len(found)
                limited |= len(frontier) > remaining
                found.update((r["id"], r) for r in frontier[:remaining])
            limited |= bool(frontier)
            all_rows = sorted(found.values(), key=lambda r: (r["sent_epoch"] is None, r["sent_epoch"] or 0, r["id"]))
            anchor_position = next((i for i, r in enumerate(all_rows) if r["id"] == actual_anchor), 0)
            start = (anchor_position // 40 if anchor_id is not None else max(0, page)) * 40
            rows = all_rows[start:start + 40]
            previous, following = start > 0, start + 40 < len(all_rows)
        previous_day = None
        for row in rows:
            row.update(zip(("platform", "account", "source_key"), discussion.key[:3]))
            row["hit"] = row["id"] in discussion.hit_ids
            row["source_anchor"] = row["id"] in {r["id"] for r in anchors}
            row["attachments"] = [dict(a) for a in conn.execute("""SELECT a.id,a.ordinal,a.name,a.kind,a.file_id,a.origin,
                a.context,f.path FROM chat_attachments a LEFT JOIN files f ON f.id=a.file_id
                WHERE a.message_id=? ORDER BY a.ordinal LIMIT 300""", (row["id"],))]
            row["gap_before"] = False
            row["forward_items"] = forward_items(row["forward_json"])
            day = display_time(row["sent_at"])[:10] if row["sent_at"] else "时间未知"
            row["day_heading"] = day if day != previous_day else ""
            previous_day = day
        if not nearby:
            for left, right in zip(rows, rows[1:]):
                if left["sent_epoch"] is not None and right["sent_epoch"] is not None:
                    right["gap_before"] = bool(conn.execute("""SELECT 1 FROM chat_messages WHERE source_id=?
                        AND (sent_epoch,id)>(?,?) AND (sent_epoch,id)<(?,?) LIMIT 1""",
                        (source_id,left["sent_epoch"],left["id"],right["sent_epoch"],right["id"])).fetchone())
        notice = "附近消息 · 含未命中的上下文" if nearby else "相关片段 · " + discussion.reason + "，不代表完整聊天"
        notice += "；历史可能未完整同步"
        coverage = conn.execute("SELECT * FROM chat_coverage WHERE source_id=? AND start_epoch<=? AND end_epoch>=? "
                                "ORDER BY id DESC LIMIT 1", (source_id, seed["sent_epoch"], seed["sent_epoch"])).fetchone()
        if coverage:
            notice += (f"；该时段上次同步有 {coverage['issue_count']} 项缺口" if coverage["state"] == "partial"
                       else "；该锚点所在同步范围上次已读完（不代表全部历史）")
        if nearby and seed["sent_epoch"] is None:
            notice = "发送时间未知，仅显示选中消息"
        if limited:
            notice += "；已达到讨论读取上限"
        before_cursor = (-1, rows[0]["sent_epoch"], rows[0]["id"]) if nearby and rows and previous else None
        after_cursor = (1, rows[-1]["sent_epoch"], rows[-1]["id"]) if nearby and rows and following else None
        return TimelinePage(tuple(rows), actual_anchor, previous, following, limited, notice,
                            before_cursor, after_cursor, page if nearby else start // 40)


def verify_message(path, discussion, message_id, *, expected_body=None):
    with snapshot(path) as conn:
        row = conn.execute("""SELECT m.*,s.platform,s.account,s.source_key FROM chat_messages m
            JOIN chat_sources s ON s.id=m.source_id WHERE m.id=?""", (message_id,)).fetchone()
        if row is None or tuple(row[k] for k in ("platform", "account", "source_key")) != discussion.key[:3]:
            raise TimelineError("消息已移除或来源已变化")
        if expected_body is not None and row["body"] != expected_body:
            raise TimelineError("消息已更新，请刷新后操作")
        return dict(row)
