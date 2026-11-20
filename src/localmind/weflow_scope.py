"""Account-bound, explicit conversation selections for desktop indexing."""
from __future__ import annotations

import json

from .chats import digest


CATEGORIES = {
    "private": "单人会话", "group": "群聊", "enterprise": "企业会话",
    "official": "公众号", "service": "服务会话", "system": "系统会话",
    "other": "其他", "unknown": "类型未知",
}
DEFAULT_CATEGORIES = {"private", "group"}
MAX_SCOPE_SESSIONS = 20_000


def scope_key(base_url: str, account_id: str | None) -> str:
    return "weflow_scope:" + digest(base_url + "\0" + (account_id or "legacy"))


def make_scope(base_url: str, account_id: str | None, sessions: list[dict], selected_ids) -> dict:
    rows = {row["id"]: row for row in sessions}
    ids = sorted(set(selected_ids))
    if not ids or len(rows) > MAX_SCOPE_SESSIONS or any(identity not in rows for identity in ids):
        raise ValueError("请选择至少一个当前可见会话，范围不可超过 20000 个会话")
    value = {"version": 1, "base_url": base_url, "account_id": account_id,
             "ids": ids, "types": {identity: rows[identity].get("category", "unknown") for identity in ids},
             "known_ids": sorted(rows)}
    value["fingerprint"] = scope_fingerprint(value)
    return value


def scope_fingerprint(value: dict) -> str:
    return digest(json.dumps({key: value[key] for key in ("base_url", "account_id", "ids", "types")},
                             ensure_ascii=False, sort_keys=True))


def validate_scope(value: object, base_url: str, account_id: str | None) -> dict:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("会话选择已失效，请重新选择")
    if value.get("base_url") != base_url or value.get("account_id") != account_id:
        raise ValueError("会话选择所属账号或接口已改变，请重新选择")
    ids, known, types = value.get("ids"), value.get("known_ids"), value.get("types")
    if not isinstance(ids, list) or not isinstance(known, list) or not isinstance(types, dict):
        raise ValueError("会话选择记录格式无效")
    for values in (ids, known):
        if not 0 < len(values) <= MAX_SCOPE_SESSIONS or any(not isinstance(v, str) or not v or len(v) > 1024 for v in values):
            raise ValueError("会话选择列表无效")
        if len(values) != len(set(values)):
            raise ValueError("会话选择包含重复 ID")
    if not set(ids) <= set(known) or set(types) != set(ids) or any(not isinstance(t, str) or t not in CATEGORIES for t in types.values()):
        raise ValueError("会话选择类型信息无效")
    if value.get("fingerprint") != scope_fingerprint(value):
        raise ValueError("会话选择记录已改变，请重新确认")
    return value


def load_scope(db, base_url: str, account_id: str | None) -> dict | None:
    if not account_id:
        return None  # Legacy WeFlow cannot distinguish accounts across launches.
    try:
        value = json.loads(db.get_setting(scope_key(base_url, account_id), "null"))
        return validate_scope(value, base_url, account_id)
    except (TypeError, ValueError, KeyError):
        return None


def save_scope(db, value: dict) -> None:
    validate_scope(value, value["base_url"], value["account_id"])
    if value["account_id"]:
        db.set_setting(scope_key(value["base_url"], value["account_id"]), json.dumps(value, ensure_ascii=False))


def restored_ids(sessions: list[dict], previous: dict | None) -> set[str]:
    if previous is None:
        return {r["id"] for r in sessions if r.get("category") in DEFAULT_CATEGORIES}
    types = previous["types"]
    return {r["id"] for r in sessions if r["id"] in types and types[r["id"]] == r.get("category", "unknown")}
