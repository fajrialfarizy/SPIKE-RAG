"""
A clone of HEAD must run without the ignored evaluation data.

The phase 4 dataset (``data/eval_cache``, ``data/artifacts_eval``,
``data/eval_corpus``) is untracked and ignored on purpose, so the deploy branch has
to work from the committed tree alone. This test clones the repository into a
temporary directory and imports ``app`` there: the clone has no evaluation data, no
``data/artifacts`` and no ``chroma_db``, which is exactly what Streamlit Cloud gets.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Paths that exist locally but never travel with the branch.
IGNORED_PATHS = (
    "data/eval_cache",
    "data/artifacts_eval",
    "data/eval_corpus",
    "data/artifacts",
    "data/dummy_pdfs",
    "chroma_db",
)

REQUIRED_FILES = (
    "app.py",
    "config.py",
    "requirements.txt",
    "core/prompts.py",
    "core/build_info.py",
)


def git(*args, cwd=ROOT):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    )


def require_git_repository():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git working tree")


def test_clone_of_head_carries_no_evaluation_data(tmp_path):
    require_git_repository()
    clone = tmp_path / "clone"
    result = git("clone", "--quiet", "--no-hardlinks", str(ROOT), str(clone))
    assert result.returncode == 0, result.stderr

    for relative in IGNORED_PATHS:
        assert not (clone / relative).exists(), f"{relative} must not be in the clone"
    for relative in REQUIRED_FILES:
        assert (clone / relative).is_file(), f"{relative} is missing from the clone"


def test_clone_of_head_starts_the_app(tmp_path):
    require_git_repository()
    clone = tmp_path / "clone"
    result = git("clone", "--quiet", "--no-hardlinks", str(ROOT), str(clone))
    assert result.returncode == 0, result.stderr

    env = dict(os.environ)
    env["PYTHONPATH"] = str(clone)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    started = subprocess.run(
        [sys.executable, "-c", "import app; print('app_import_ok')"],
        cwd=str(clone),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert started.returncode == 0, started.stderr[-2000:]
    assert "app_import_ok" in started.stdout
