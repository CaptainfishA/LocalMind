"""Shared bounds for stored chat text, independent of transport page limits."""

MAX_MESSAGE_CHARS = 128_000
MAX_RAW_MESSAGE_CHARS = 1_000_000
MAX_XML_DEPTH = 64
MAX_XML_NODES = 10_000
MAX_FORWARD_ITEMS = 200
MAX_FORWARD_DEPTH = 4


def validate_body(body: object) -> None:
    if not isinstance(body, str):
        raise ValueError("消息正文必须为字符串")
    if len(body) > MAX_MESSAGE_CHARS:
        raise ValueError(f"单条规范化正文超过 {MAX_MESSAGE_CHARS} 字符安全上限")
    if "\x00" in body or any(0xD800 <= ord(char) <= 0xDFFF for char in body):
        raise ValueError("消息正文包含无效文本字符")
