from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def make_snapshot_path(base_path: str, suffix: str = ".jpg") -> str:
    ensure_directory(base_path)
    return str(Path(base_path) / f"{uuid4().hex}{suffix}")
