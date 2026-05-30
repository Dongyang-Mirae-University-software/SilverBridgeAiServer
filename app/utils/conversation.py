from __future__ import annotations

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
