from __future__ import annotations

import re
from typing import Iterable


def normalize_history(
    history: Iterable[dict[str, str]] | None,
    keep_turns: int = 6,
    summary_max_chars: int = 500,
) -> tuple[list[dict[str, str]], str]:
    items = list(history or [])
    if keep_turns < 1:
        keep_turns = 1

    recent = items[-keep_turns:]
    older = items[:-keep_turns]
    if not older:
        return recent, ""

    chunks: list[str] = []
    for turn in older:
        role = (turn.get("role") or "unknown").strip()
        content = (turn.get("content") or "").strip()
        if content:
            chunks.append(f"{role}: {content}")

    summary = " | ".join(chunks)
    if len(summary) > summary_max_chars:
        summary = summary[: summary_max_chars - 3] + "..."
    return recent, summary


def format_recent_history(history: Iterable[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in history:
        role = (item.get("role") or "unknown").strip()
        content = (item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# 외부 LLM(Claude)으로 보내는 이력에서 가리는 개인정보 형태(2026-10-28 S-6). 판단에 필요 없는 값이라 자리표시로 바꾼다.
_RRN_RE = re.compile(r"(?<!\d)\d{6}[-\s]?[1-4]\d{6}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)0\d{1,2}[-\s.]?\d{3,4}[-\s.]?\d{4}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def mask_personal_info(text: str) -> str:
    """전화번호·주민등록번호·이메일 형태를 자리표시로 바꾼다. 주민번호를 먼저 본다(전화 패턴과 겹치지 않게)."""
    masked = _RRN_RE.sub("[주민번호]", text or "")
    masked = _PHONE_RE.sub("[전화]", masked)
    return _EMAIL_RE.sub("[이메일]", masked)


def remove_personal_info(text: str) -> str:
    """mask_personal_info 와 같은 형태를 자리표시 없이 지운다(병원에 넘기는 증상 요약용)."""
    removed = _RRN_RE.sub("", text or "")
    removed = _PHONE_RE.sub("", removed)
    return _EMAIL_RE.sub("", removed)
