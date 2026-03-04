"""Shared history trimming utility — avoids circular imports between agent.py and agents/."""

from __future__ import annotations

from typing import Any


def trim_history(history: list[Any], max_chars: int) -> list[Any]:
    """Drop oldest messages until total content size is within max_chars."""
    total = sum(len(str(getattr(m, "content", m))) for m in history)
    while total > max_chars and len(history) > 2:
        dropped = history.pop(0)
        total -= len(str(getattr(dropped, "content", dropped)))
    return history
