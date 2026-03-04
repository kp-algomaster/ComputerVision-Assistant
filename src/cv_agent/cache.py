"""Disk-backed cache for LLM responses and tool results.

Reduces redundant token usage by caching:
- LLM responses keyed by model + prompt hash
- Tool results (paper fetches, searches) with per-type TTLs

Zero external dependencies — uses hashlib, json, pathlib, time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_agent.config import AgentConfig

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("./output/.cache")
_singleton: CVCache | None = None


class CVCache:
    """Content-addressed disk cache with TTL support."""

    def __init__(self, cache_dir: Path, default_ttl: int = 86400) -> None:
        self._dir = cache_dir
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def make_key(self, *parts: str) -> str:
        """SHA-256 of all parts joined by null byte."""
        raw = "\x00".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        """Return cached value or None on miss / expiry."""
        path = self._entry_path(key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text())
            if data["expires_at"] < time.time():
                path.unlink(missing_ok=True)
                self._misses += 1
                return None
            self._hits += 1
            return data["value"]
        except Exception:
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

    def set(self, key: str, value: str, ttl: int | None = None, key_hint: str = "") -> None:
        """Write value to cache with given TTL (seconds)."""
        path = self._entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "value": value,
            "expires_at": time.time() + (ttl if ttl is not None else self._default_ttl),
            "key_hint": key_hint[:80],
        }
        path.write_text(json.dumps(data, ensure_ascii=False))

    def clear(self, older_than_seconds: int = 0) -> int:
        """Delete expired or old entries. Returns count of deleted files."""
        cutoff = time.time() - older_than_seconds if older_than_seconds else None
        count = 0
        for entry in self._dir.rglob("*.json"):
            try:
                data = json.loads(entry.read_text())
                expired = data["expires_at"] < time.time()
                too_old = cutoff is not None and data["expires_at"] < cutoff
                if expired or too_old:
                    entry.unlink(missing_ok=True)
                    count += 1
            except Exception:
                entry.unlink(missing_ok=True)
                count += 1
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        entries = list(self._dir.rglob("*.json"))
        total_bytes = sum(e.stat().st_size for e in entries if e.exists())
        expired = 0
        now = time.time()
        for entry in entries:
            try:
                data = json.loads(entry.read_text())
                if data["expires_at"] < now:
                    expired += 1
            except Exception:
                expired += 1
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
            "total_entries": len(entries),
            "expired_entries": expired,
            "size_bytes": total_bytes,
            "size_mb": round(total_bytes / 1_048_576, 3),
            "cache_dir": str(self._dir),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _entry_path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"


def get_cache(config: AgentConfig | None = None) -> CVCache:
    """Return the module-level CVCache singleton, initialising if needed."""
    global _singleton
    if _singleton is not None:
        return _singleton

    if config is None:
        from cv_agent.config import load_config
        config = load_config()

    if not config.cache.enabled:
        # Return a no-op cache when disabled
        _singleton = _NoOpCache()
        return _singleton

    cache_dir = Path(config.output.base_dir) / ".cache"
    _singleton = CVCache(cache_dir=cache_dir, default_ttl=config.cache.ttl_llm)
    logger.debug("CV cache initialised at %s", cache_dir)
    return _singleton


def reset_cache_singleton() -> None:
    """Reset the singleton (useful for tests or config reloads)."""
    global _singleton
    _singleton = None


class _NoOpCache(CVCache):
    """Cache that never stores anything — used when cache.enabled=False."""

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._dir = Path("/dev/null")

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str, ttl: int | None = None, key_hint: str = "") -> None:
        pass

    def clear(self, older_than_seconds: int = 0) -> int:
        return 0

    def stats(self) -> dict:
        return {"enabled": False, "hits": 0, "misses": 0}
