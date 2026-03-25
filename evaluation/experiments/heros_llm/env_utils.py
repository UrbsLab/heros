"""Helpers for loading local environment variables from .env files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import os


REPO_ROOT = Path(__file__).resolve().parents[3]


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE env file."""
    parsed: Dict[str, str] = {}
    if not path.exists():
        return parsed

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key:
            parsed[key] = value
    return parsed


def load_env_file(path: Path, override: bool = False) -> Dict[str, str]:
    """Load env vars from a file into os.environ."""
    loaded = parse_env_file(path)
    for key, value in loaded.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def discover_default_env_files(explicit_env_file: Optional[str] = None) -> Iterable[Path]:
    """Return env files to load in priority order."""
    candidates = []
    if explicit_env_file:
        candidates.append(Path(explicit_env_file).expanduser())
    else:
        cwd_env = Path.cwd() / ".env"
        repo_env = REPO_ROOT / ".env"
        if cwd_env.exists():
            candidates.append(cwd_env)
        if repo_env.exists() and repo_env != cwd_env:
            candidates.append(repo_env)
    return candidates
