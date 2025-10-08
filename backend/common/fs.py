"""Filesystem helpers."""
from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (including parents) and return it as :class:`Path`."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
