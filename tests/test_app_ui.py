"""
The remaining AppTest scenarios (Phase 3), marked slow.

Each AppTest run pays Streamlit's script-run startup, so this module is excluded
from the default offline suite and run with ``pytest -m slow``.
"""

import time

import pytest

import app_texts as T
import config
from core.pipeline import AnswerStatus
from streamlit.testing.v1 import AppTest

from tests.test_app import (
    APP_PATH,
    PASSWORD,
    FakePipeline,
    make_runtime,
    start_app,
)

pytestmark = pytest.mark.slow


def login(at, password=PASSWORD, username="labprestasi"):
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.form_submit_button[0].click().run()
    return at


def ask(at, question):
    at.session_state["last_message_at"] = None
    at.chat_input[0].set_value(question).run()
    return at


def test_a_wrong_password_shows_a_generic_error(tmp_path, monkeypatch):
    at, factory = start_app(tmp_path, monkeypatch=monkeypatch)

    login(at, password="salah-sekali")

    assert at.session_state["authenticated"] is False
    assert T.LOGIN_FAILED in [error.value for error in at.error]
    assert factory.calls == 0


def test_five_failures_lock_the_account_in_the_ui(tmp_path, monkeypatch):
    at, _ = start_app(tmp_path, monkeypatch=monkeypatch)

    for _ in range(config.AUTH_MAX_FAILED_ATTEMPTS):
        login(at, password="salah-sekali")

    assert any("Terlalu banyak percobaan masuk" in error.value for error in at.error)

    login(at, password=PASSWORD)
    assert at.session_state["authenticated"] is False


@pytest.mark.parametrize(
    "status,answer",
    [
        (AnswerStatus.NOT_FOUND, "Maaf, informasi belum tersedia."),
        (AnswerStatus.OUT_OF_SCOPE, "Maaf, di luar ruang lingkup."),
        (AnswerStatus.PRIVACY_REFUSED, "Maaf, tidak dapat memberikan data pribadi."),
        (AnswerStatus.SECURITY_REFUSED, "Maaf, tidak dapat menampilkan instruksi."),
        (AnswerStatus.NO_SOURCE, "Maaf, belum dapat menyusun jawaban."),
        (AnswerStatus.RATE_LIMITED, "Layanan sedang menerima terlalu banyak permintaan."),
        (AnswerStatus.TECHNICAL_ERROR, "Maaf, terjadi kendala teknis."),
    ],
)
def test_every_status_is_rendered_as_an_assistant_message(
    tmp_path, monkeypatch, status, answer
):
    at, _ = start_app(
        tmp_path,
        runtime=make_runtime(tmp_path, FakePipeline(answer=answer, status=status)),
        monkeypatch=monkeypatch,
    )
    login(at)

    ask(at, "Apa kendala tim?")

    assert at.session_state["chat_history"][-1] == {
        "role": "assistant",
        "content": answer,
    }


def test_a_pipeline_failure_shows_the_fixed_text_and_no_traceback(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, FakePipeline(fail=RuntimeError("boom raw detail")))
    at, _ = start_app(tmp_path, runtime=runtime, monkeypatch=monkeypatch)
    login(at)

    ask(at, "Apa kendala tim?")

    content = at.session_state["chat_history"][-1]["content"]
    assert content == T.TECHNICAL_ERROR
    assert "boom" not in content
    assert not at.exception


def test_the_chat_input_is_capped_at_the_configured_length(tmp_path, monkeypatch):
    at, _ = start_app(tmp_path, monkeypatch=monkeypatch)
    login(at)

    assert at.chat_input[0].proto.max_chars == config.CHAT_MAX_INPUT_CHARS


def test_the_cooldown_blocks_a_second_question_without_an_llm_call(tmp_path, monkeypatch):
    pipeline = FakePipeline()
    at, _ = start_app(tmp_path, runtime=make_runtime(tmp_path, pipeline), monkeypatch=monkeypatch)
    login(at)

    ask(at, "Apa kendala tim?")
    at.chat_input[0].set_value("Pertanyaan kedua").run()

    assert len(pipeline.questions) == 1
    assert "Tunggu" in at.session_state["chat_history"][-1]["content"]


def test_the_hourly_cap_blocks_the_next_question(tmp_path, monkeypatch):
    pipeline = FakePipeline()
    at, _ = start_app(tmp_path, runtime=make_runtime(tmp_path, pipeline), monkeypatch=monkeypatch)
    login(at)
    now = time.time()
    at.session_state["message_times"] = [now] * config.SESSION_HOURLY_MESSAGE_CAP

    at.chat_input[0].set_value("Apa kendala tim?").run()

    assert pipeline.questions == []
    assert at.session_state["chat_history"][-1]["content"] == T.HOURLY_LIMIT


def test_a_long_queue_wait_is_shown_inside_the_processing_status(tmp_path, monkeypatch):
    class SlowLimiter:
        def wait_estimate(self, tokens):
            return 12.0

    pipeline = FakePipeline(answer="Kendala utama adalah logistik [RPT0002].")
    at, _ = start_app(tmp_path, runtime=make_runtime(tmp_path, pipeline), monkeypatch=monkeypatch)
    at.session_state["_queue_limiter"] = SlowLimiter()
    login(at)

    ask(at, "Apa kendala tim?")

    rendered = [element.value for element in at.markdown if isinstance(element.value, str)]
    rendered += [
        element.value
        for status in at.status
        for element in status.markdown
        if isinstance(element.value, str)
    ]
    assert any("Sedang mengantre karena batas layanan" in value for value in rendered)
    assert pipeline.questions == ["Apa kendala tim?"]


def test_the_session_daily_cap_blocks_the_next_question(tmp_path, monkeypatch):
    pipeline = FakePipeline()
    at, _ = start_app(tmp_path, runtime=make_runtime(tmp_path, pipeline), monkeypatch=monkeypatch)
    login(at)
    now = time.time()
    at.session_state["message_times"] = [now - 7200] * config.SESSION_MAX_QUESTIONS_PER_DAY

    at.chat_input[0].set_value("Apa kendala tim?").run()

    assert pipeline.questions == []
    assert at.session_state["chat_history"][-1]["content"] == T.SESSION_DAILY_LIMIT


def test_the_three_example_buttons_run_their_question(tmp_path, monkeypatch):
    pipeline = FakePipeline()
    at, _ = start_app(tmp_path, runtime=make_runtime(tmp_path, pipeline), monkeypatch=monkeypatch)
    login(at)

    labels = [button.label for button in at.button]
    assert set(T.EXAMPLE_QUESTIONS).issubset(set(labels))

    example_index = labels.index(T.EXAMPLE_QUESTIONS[0])
    at.session_state["last_message_at"] = None
    at.button[example_index].click().run()

    assert pipeline.questions == [T.EXAMPLE_QUESTIONS[0]]


def test_a_missing_settings_section_shows_the_configuration_error(tmp_path, monkeypatch):
    backends = [
        "LAB_PRESTASI_USERNAME",
        "LAB_PRESTASI_PASSWORD_HASH",
        "GROQ_API_KEY",
    ]
    for name in backends:
        monkeypatch.delenv(name, raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_settings_override"] = {}
    at.run()

    assert T.LOGIN_CONFIG_ERROR in [error.value for error in at.error]


def test_a_correct_login_opens_the_chat_and_loads_the_runtime(tmp_path, monkeypatch):
    at, factory = start_app(tmp_path, monkeypatch=monkeypatch)

    login(at)

    assert at.session_state["authenticated"] is True
    assert factory.calls == 1
    assert at.title[0].value == T.CHAT_TITLE
    assert at.chat_input


def test_a_question_is_answered_and_kept_in_the_display_history(tmp_path, monkeypatch):
    pipeline = FakePipeline(answer="Kendala utama adalah logistik [RPT0002].")
    at, _ = start_app(tmp_path, runtime=make_runtime(tmp_path, pipeline), monkeypatch=monkeypatch)
    login(at)

    ask(at, "Apa kendala tim pada business plan?")

    assert pipeline.questions == ["Apa kendala tim pada business plan?"]
    assert [message["content"] for message in at.session_state["chat_history"]] == [
        "Apa kendala tim pada business plan?",
        "Kendala utama adalah logistik [RPT0002].",
    ]


def test_logout_clears_the_session_and_the_history(tmp_path, monkeypatch):
    at, _ = start_app(tmp_path, monkeypatch=monkeypatch)
    login(at)
    ask(at, "Apa kendala tim?")
    assert at.session_state["chat_history"]

    at.sidebar.button[0].click().run()

    assert at.session_state["authenticated"] is False
    assert at.session_state["chat_history"] == []
    assert at.chat_input == []
