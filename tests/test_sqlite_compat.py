"""Unit tests for the Linux SQLite shim (Phase 3)."""

import sys
import types

import pytest

import config
from core import sqlite_compat
from core.sqlite_compat import SQLiteVersionError, ensure_sqlite_version, is_supported


def test_version_parsing_ignores_suffixes():
    assert sqlite_compat.parse_version("3.45.3") == (3, 45, 3)
    assert sqlite_compat.parse_version("3.35.0-dev") == (3, 35, 0)


def test_the_minimum_version_is_the_configured_one():
    minimum = ".".join(str(part) for part in config.PYTHON_MIN_SQLITE_VERSION)
    assert is_supported(minimum) is True
    assert is_supported("3.34.9") is False


def test_a_supported_system_sqlite_is_used_directly(monkeypatch):
    monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.45.3")
    assert ensure_sqlite_version() == "3.45.3"


def test_an_old_sqlite_outside_linux_fails_loudly(monkeypatch):
    monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.31.1")
    monkeypatch.setattr(sqlite_compat.sys, "platform", "win32")
    with pytest.raises(SQLiteVersionError):
        ensure_sqlite_version()


def test_linux_swaps_in_the_shim_when_it_is_new_enough(monkeypatch):
    monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.31.1")
    monkeypatch.setattr(sqlite_compat.sys, "platform", "linux")
    shim = types.SimpleNamespace(sqlite_version="3.40.1")
    monkeypatch.setitem(sys.modules, "pysqlite3", shim)
    original = sys.modules.get("sqlite3")

    assert ensure_sqlite_version() == "3.40.1"
    assert sys.modules["sqlite3"] is shim

    if original is not None:
        sys.modules["sqlite3"] = original


def test_linux_without_the_shim_fails_loudly(monkeypatch):
    monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.31.1")
    monkeypatch.setattr(sqlite_compat.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "pysqlite3", None)
    with pytest.raises(SQLiteVersionError):
        ensure_sqlite_version()
