"""
SQLite compatibility for the Linux deployment (Phase 3).

chromadb requires SQLite >= 3.35. Some Streamlit Community Cloud images still
ship an older system SQLite, and the documented workaround is the
``pysqlite3-binary`` wheel imported before chromadb. The shim is Linux-only, so
on Windows/macOS this module normally does nothing.

No silent fallback: if the system SQLite is too old and the shim is missing or
still too old, ``SQLiteVersionError`` is raised and the caller shows the fixed
technical-error screen.
"""

from __future__ import annotations

import sqlite3
import sys
from typing import Tuple

import config


class SQLiteVersionError(RuntimeError):
    """Raised when no usable SQLite version can be provided."""

    # Fix pack 13a: the code the app shows under its generic message.
    code = "DB/sqlite_version"


def parse_version(version: str) -> Tuple[int, ...]:
    parts = []
    for piece in (version or "").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_supported(version: str) -> bool:
    current = parse_version(version)
    required = config.PYTHON_MIN_SQLITE_VERSION
    return current[: len(required)] >= required


def ensure_sqlite_version() -> str:
    """Return the usable SQLite version, installing the Linux shim when needed."""
    version = sqlite3.sqlite_version
    if is_supported(version):
        return version

    if sys.platform.startswith("linux"):
        try:
            import pysqlite3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SQLiteVersionError(
                f"SQLite {version} is older than the required "
                f"{config.PYTHON_MIN_SQLITE_VERSION}; install the Linux-only "
                "'pysqlite3-binary' requirement."
            ) from exc

        if not is_supported(pysqlite3.sqlite_version):
            raise SQLiteVersionError(
                f"SQLite {pysqlite3.sqlite_version} from pysqlite3-binary is still "
                f"older than the required {config.PYTHON_MIN_SQLITE_VERSION}."
            )

        sys.modules["sqlite3"] = pysqlite3
        return pysqlite3.sqlite_version

    raise SQLiteVersionError(
        f"SQLite {version} is older than the required "
        f"{config.PYTHON_MIN_SQLITE_VERSION}."
    )
