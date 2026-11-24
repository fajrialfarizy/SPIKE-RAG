"""
``core.build_info`` reads the deployed commit from ``.git`` without a subprocess.

Every failure mode must degrade to ``unknown``: the deploy id is diagnostic, so it
cannot be allowed to raise during startup.
"""

import logging
from pathlib import Path

import pytest

from core import build_info

HASH = "0123456789abcdef0123456789abcdef01234567"


def make_repo(tmp_path: Path, *, head: str, ref: str = "refs/heads/main") -> Path:
    """A minimal repository: ``.git/HEAD`` plus an optional loose ref."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(head + "\n", encoding="utf-8")
    if head.startswith("ref:"):
        ref_path = git_dir / ref
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(HASH + "\n", encoding="utf-8")
    return tmp_path


def test_reads_the_project_repository():
    commit = build_info.read_commit()
    assert commit == "unknown" or (len(commit) == build_info.SHORT_LENGTH)


def test_branch_ref_in_temporary_repo(tmp_path):
    root = make_repo(tmp_path, head="ref: refs/heads/main")
    assert build_info.read_commit(root) == HASH[:7]


def test_packed_refs_only(tmp_path):
    root = make_repo(tmp_path, head="ref: refs/heads/main")
    loose = root / ".git" / "refs" / "heads" / "main"
    loose.unlink()
    (root / ".git" / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{'f' * 40} refs/heads/other\n"
        f"{HASH} refs/heads/main\n",
        encoding="utf-8",
    )
    assert build_info.read_commit(root) == HASH[:7]


def test_detached_head(tmp_path):
    root = make_repo(tmp_path, head=HASH)
    assert build_info.read_commit(root) == HASH[:7]


def test_worktree_gitdir_file(tmp_path):
    real_git = tmp_path / "real" / ".git"
    (real_git / "refs" / "heads").mkdir(parents=True)
    (real_git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (real_git / "refs" / "heads" / "main").write_text(HASH + "\n", encoding="utf-8")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {real_git}\n", encoding="utf-8")
    assert build_info.read_commit(worktree) == HASH[:7]


def test_missing_git_directory_is_unknown(tmp_path):
    assert build_info.read_commit(tmp_path) == build_info.UNKNOWN


def test_unknown_ref_is_unknown(tmp_path):
    root = make_repo(tmp_path, head="ref: refs/heads/main")
    (root / ".git" / "refs" / "heads" / "main").unlink()
    assert build_info.read_commit(root) == build_info.UNKNOWN


@pytest.mark.parametrize(
    "head",
    [
        "",
        "not-a-hash",
        "ref: ",
        "0123",
        HASH[:39] + "z",
    ],
)
def test_unexpected_head_content_is_unknown(tmp_path, head):
    root = make_repo(tmp_path, head=head)
    assert build_info.read_commit(root) == build_info.UNKNOWN


def test_unreadable_git_path_is_unknown(tmp_path, monkeypatch):
    root = make_repo(tmp_path, head="ref: refs/heads/main")

    def boom(*args, **kwargs):
        raise OSError("denied")

    monkeypatch.setattr(build_info, "_read_text", boom)
    assert build_info.read_commit(root) == build_info.UNKNOWN


def test_log_line_is_one_per_process(tmp_path, caplog):
    root = make_repo(tmp_path, head="ref: refs/heads/main")
    with caplog.at_level(logging.INFO, logger="core_build_info"):
        first = build_info.log_build_once(force=True)
        second = build_info.log_build_once(force=True)
    lines = [r for r in caplog.records if r.getMessage().startswith("app_build ")]
    assert first == second
    assert len(lines) == 2  # forced twice; without force only the first logs
    assert lines[0].getMessage().startswith("app_build commit=")


def test_log_is_suppressed_after_the_first_call(tmp_path, monkeypatch, caplog):
    root = make_repo(tmp_path, head="ref: refs/heads/main")
    monkeypatch.setattr(build_info, "read_commit", lambda *a, **k: "abc1234")
    monkeypatch.setattr(build_info, "_LOGGED", False)
    with caplog.at_level(logging.INFO, logger="core_build_info"):
        assert build_info.log_build_once() == "abc1234"
        assert build_info.log_build_once() == ""
    lines = [r for r in caplog.records if r.getMessage().startswith("app_build ")]
    assert [r.getMessage() for r in lines] == ["app_build commit=abc1234"]
