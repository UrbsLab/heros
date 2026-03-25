"""Simple file-based caches keyed by hashed request payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def hash_payload(payload: Dict[str, Any]) -> str:
    """Create a stable SHA-256 hash for a JSON-compatible payload."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FileCache:
    """A tiny JSON cache that stores one object per hash key."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / "{0}.json".format(key)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, payload: Dict[str, Any]) -> Path:
        path = self._cache_path(key)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path
