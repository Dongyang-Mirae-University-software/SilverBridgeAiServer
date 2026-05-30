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
