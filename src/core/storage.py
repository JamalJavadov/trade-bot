from __future__ import annotations
from pathlib import Path
from typing import Optional


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)


def read_text(path: str, encoding: str = "utf-8") -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding=encoding)
