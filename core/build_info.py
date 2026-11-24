"""
Build identity for the deployed app: which commit is running.

The cloud log has no deploy id, so a deployed commit cannot be told apart from a
rolled-back one. This module writes ONE line, ``app_build commit=<short>``, and it
reads the commit straight from the repository's ``.git`` directory: no
subprocess, no network, no environment variable and no secret. Anything odd
(missing ``.git``, unreadable file, unexpected content) yields ``unknown``
instead of an exception, because a missing deploy id must never break startup.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("core_build_info")

UNKNOWN = "unknown"
SHORT_LENGTH = 7
_HEX = set("0123456789abcdef")
_HASH_LENGTHS = (40, 64)  # sha1 and sha256 repositories
_LOCK = threading.Lock()
_LOGGED = False


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _git_dir(root: Path) -> Optional[Path]:
    """The repository's ``.git`` directory, following a worktree pointer file."""
    dot_git = root / ".git"
    try:
        if dot_git.is_dir():
            return dot_git
        if not dot_git.is_file():
            return None
    except OSError:
        return None

    text = _read_text(dot_git)
    if not text:
        return None
    first = text.strip().splitlines()[0].strip()
    if not first.startswith("gitdir:"):
        return None
    target = first[len("gitdir:") :].strip()
    if not target:
        return None
    path = Path(target)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        return resolved if resolved.is_dir() else None
    except OSError:
        return None


def _loose_ref(git_dir: Path, ref: str) -> Optional[str]:
    text = _read_text(git_dir / ref)
    return text.strip() if text else None


def _packed_ref(git_dir: Path, ref: str) -> Optional[str]:
    text = _read_text(git_dir / "packed-refs")
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "^")):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def _clean_hash(value: Optional[str], length: int) -> Optional[str]:
    if not value:
        return None
    text = value.strip().lower()
    if len(text) not in _HASH_LENGTHS or any(char not in _HEX for char in text):
        return None
    return text[:length]


def read_commit(
    root: Optional[Path] = None, *, length: int = SHORT_LENGTH
) -> str:
    """
    Short commit of the checked-out revision, or ``"unknown"``.

    ``root`` defaults to the repository this module lives in. Only ``HEAD`` and
    the ref it names (loose file, then ``packed-refs``) are read, so the cost is
    two small file reads.
    """
    try:
        base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
        git_dir = _git_dir(base)
        if git_dir is None:
            return UNKNOWN

        head = _loose_ref(git_dir, "HEAD")
        if not head:
            return UNKNOWN

        if head.startswith("ref:"):
            ref = head[len("ref:") :].strip()
            if not ref:
                return UNKNOWN
            value = _loose_ref(git_dir, ref) or _packed_ref(git_dir, ref)
        else:
            value = head  # detached HEAD holds the hash itself

        return _clean_hash(value, length) or UNKNOWN
    except Exception:  # a deploy id must never break startup
        return UNKNOWN


def log_build_once(*, force: bool = False) -> str:
    """Write ``app_build commit=...`` once per process (tests pass ``force``)."""
    global _LOGGED
    with _LOCK:
        if _LOGGED and not force:
            return ""
        _LOGGED = True

    commit = read_commit()
    logger.info("app_build commit=%s", commit)
    return commit
