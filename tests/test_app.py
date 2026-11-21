"""
AppTest transcripts for the Streamlit layer (Phase 3).

The app reads settings from the environment (its documented local-development
fallback) and the runtime object from session state when a test provides one, so
no secret file, Chroma collection, model weights or network are needed.

These are the fast, essential flows; the remaining UI scenarios live in
``tests/test_app_ui.py`` and carry the ``slow`` marker.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import app as app_module
import app_texts as T
import config
from core.kb_sync import KBVersion
from core.pipeline import AnswerStatus, DebugInfo, RagResult
from core.settings import load_settings
from streamlit.testing.v1 import AppTest

PASSWORD = "kata-sandi-panjang-2026"
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def make_secrets(tmp_path):
    from core.auth import hash_password

    return {
        "auth": {
            "username": "labprestasi",
            "password_hash": hash_password(
                PASSWORD, iterations=config.AUTH_MIN_PBKDF2_ITERATIONS
            ),
        },
        "groq": {"api_key": "gsk_test_placeholder"},
        "kb": {
            "source": config.KB_SOURCE_LOCAL,
            "local_dir": str(tmp_path / "chroma_db"),
            "work_dir": str(tmp_path / "kb_work"),
        },
    }


def make_settings(tmp_path):
    return load_settings(make_secrets(tmp_path), environ={})


class FakePipeline:
    def __init__(self, answer="Jawaban uji [RPT0002].", status=AnswerStatus.ANSWERED, fail=None):
        self.answer = answer
        self.status = status
        self.fail = fail
        self.questions = []

    def run(self, question):
        self.questions.append(question)
        if self.fail is not None:
            raise self.fail
        return RagResult(
            status=self.status,
            answer=self.answer,
            cited_report_ids=["RPT0002"] if self.status is AnswerStatus.ANSWERED else [],
            debug=DebugInfo(mode="specific", route="semantic"),
        )


def make_runtime(tmp_path, pipeline=None):
    return app_module.AppRuntime(
        settings=make_settings(tmp_path),
        pipeline=pipeline or FakePipeline(),
        kb_version=KBVersion(
            persist_dir=tmp_path / "chroma_db",
            manifest={"chunk_count": 3},
            checksum="abc",
        ),
        synchronizer=SimpleNamespace(),
    )


class RuntimeFactory:
    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.runtime


def start_app(tmp_path, runtime=None, factory=None, monkeypatch=None):
    monkeypatch.setattr(config, "AUTH_FAILURE_DELAY_SECONDS", 0.0)
    secrets = make_secrets(tmp_path)
    # Settings come from the environment, which the app supports for local
    # development; only the runtime object needs a session-state override.
    monkeypatch.setenv("LAB_PRESTASI_USERNAME", secrets["auth"]["username"])
    monkeypatch.setenv("LAB_PRESTASI_PASSWORD_HASH", secrets["auth"]["password_hash"])
    monkeypatch.setenv("GROQ_API_KEY", secrets["groq"]["api_key"])
    monkeypatch.setenv("KB_SOURCE", secrets["kb"]["source"])
    monkeypatch.setenv("KB_LOCAL_DIR", secrets["kb"]["local_dir"])
    monkeypatch.setenv("KB_WORK_DIR", secrets["kb"]["work_dir"])

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    # The first run creates the session state store; the runtime override lands
    # before the second run, which is the one a test inspects.
    at.run()
    if factory is None:
        factory = RuntimeFactory(runtime or make_runtime(tmp_path))
    at.session_state[app_module.RUNTIME_OVERRIDE_KEY] = factory
    at.run()
    return at, factory


def login(at, password=PASSWORD, username="labprestasi"):
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.form_submit_button[0].click().run()
    return at


def ask(at, question):
    at.session_state["last_message_at"] = None
    at.chat_input[0].set_value(question).run()
    return at


# -- login ----------------------------------------------------------------
def test_the_login_screen_loads_nothing_heavy(tmp_path, monkeypatch):
    at, factory = start_app(tmp_path, monkeypatch=monkeypatch)

    assert at.title[0].value == T.LOGIN_TITLE
    assert not at.chat_input
    assert factory.calls == 0
    assert at.session_state["authenticated"] is False
