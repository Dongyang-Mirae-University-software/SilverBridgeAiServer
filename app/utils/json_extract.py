from __future__ import annotations

import json
import re
from typing import Any


def _strip_markdown_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if len(lines) < 2:
        return s
    inner = "\n".join(lines[1:])
    if "```" in inner:
        inner = inner[: inner.rfind("```")].rstrip()
    return inner.strip()


def extract_json_fields_lenient(text: str, string_keys: list[str], list_keys: list[str]) -> dict[str, Any] | None:
    """max_new_tokens에 걸려 잘린 JSON에서도 채워진 필드만 건진다.

    문자열 키는 닫힌 따옴표까지, 배열 키는 지금까지 완성된 항목만 가져온다.
    하나도 못 찾으면 None.
    """
    raw = _strip_markdown_fence((text or "").strip())
    if not raw:
        return None
    result: dict[str, Any] = {}
    for key in string_keys:
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % re.escape(key), raw)
        if m:
            result[key] = m.group(1).replace('\\"', '"').strip()
    for key in list_keys:
        m = re.search(r'"%s"\s*:\s*\[' % re.escape(key), raw)
        if not m:
            continue
        rest = raw[m.end():]
        end = rest.find("]")
        chunk = rest if end == -1 else rest[:end]
        items = [item.replace('\\"', '"').strip() for item in re.findall(r'"((?:[^"\\]|\\.)*)"', chunk)]
        if items:
            result[key] = items
    m = re.search(r'"needsReservation"\s*:\s*(true|false)', raw)
    if m:
        result["needsReservation"] = m.group(1) == "true"
    return result or None


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = _strip_markdown_fence((text or "").strip())
    if not raw:
        return None
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        start = match.start()
        try:
            val, _ = decoder.raw_decode(raw[start:])
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            continue
    return None
