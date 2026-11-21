"""
Lab Prestasi — Streamlit entry point (Phase 3).

Layer boundaries: this module reads ``st.secrets``/environment, builds a
``core.settings.Settings`` and owns the UI, the session state and the cached
resources. ``core`` never imports streamlit.

Before a successful login nothing heavy is touched: no embedder, no Chroma, no
Groq client and no Drive call. Those live behind ``get_runtime()``, which is only
reached from the authenticated chat screen.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import streamlit as st

import app_texts as T
import config
from core.sqlite_compat import SQLiteVersionError, ensure_sqlite_version

# Must happen before chromadb is imported (it needs SQLite >= 3.35).
SQLITE_VERSION = ensure_sqlite_version()

from core.auth import AuthConfigurationError, AuthManager
from core.kb_sync import DriveKBSource, KBSynchronizer, KBSyncError, KBVersion
from core.llm_client import GroqLLMClient
from core.pipeline import RagPipeline, configure_file_logging
from core.settings import Settings, SettingsError, load_settings
from core.tokens import E5TokenCounter
from core.warmup import WarmupError, warm_up_embedder
from ingestion.embedder import E5Embedder

logger = logging.getLogger("app")

SETTINGS_OVERRIDE_KEY = "_settings_override"
RUNTIME_OVERRIDE_KEY = "_runtime_factory"


# ---------------------------------------------------------------------------
# Settings and runtime resources


def read_secrets() -> Mapping[str, Any]:
    """Secrets from Streamlit, or a test/local override kept in session state."""
    override = st.session_state.get(SETTINGS_OVERRIDE_KEY)
    if override is not None:
        return override
    try:
        return dict(st.secrets)
    except Exception:  # no secrets.toml is a normal local-development state
        return {}


def load_app_settings() -> Settings:
    return load_settings(read_secrets())


@st.cache_resource(show_spinner=False)
def get_auth_manager(username: str, password_hash: str) -> AuthManager:
    """One shared AuthManager, so the lockout counter is process-wide."""
    return AuthManager(username, password_hash)


def configure_app_logging(log_dir: Path = config.LOG_DIR) -> Path:
    """Operational metadata only: never question text, answers or secrets."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = log_dir / f"{config.APP_LOG_FILE_PREFIX}_{stamp}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(config.LOG_LEVEL)
    return path


def rss_mb() -> Optional[float]:
    """Resident memory in MiB, when psutil is available."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - requirements install psutil
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


@dataclass
class AppRuntime:
    """Heavy, shared resources for the authenticated part of the app."""

    settings: Settings
    pipeline: RagPipeline
    kb_version: KBVersion
    synchronizer: KBSynchronizer
    lock: threading.Lock = field(default_factory=threading.Lock)
    load_seconds: float = 0.0
    warmup_ms: float = 0.0
    rss_startup_mb: Optional[float] = None
    first_query_ms: Optional[float] = None


def build_runtime(settings: Settings) -> AppRuntime:
    """Load the knowledge base, the embedder and the LLM client (post-login)."""
    started = time.perf_counter()
    configure_file_logging(config.LOG_DIR)
    configure_app_logging(config.LOG_DIR)

    source = None
    if settings.kb.source == config.KB_SOURCE_DRIVE:
        source = DriveKBSource(
            folder_id=settings.kb.drive_folder_id or "",
            service_account_info=settings.gcp_service_account,
            manifest_name=settings.kb.drive_manifest_name,
        )

    synchronizer = KBSynchronizer(settings.kb, source=source)
    version = synchronizer.current()
    if version.handle is None:
        raise KBSyncError("The knowledge base was not validated by its loader.")

    embedder = E5Embedder()
    warmup_ms = warm_up_embedder(embedder)

    pipeline = RagPipeline(
        knowledge_base=version.handle,
        llm=GroqLLMClient(api_key=settings.groq.api_key),
        embedder=embedder,
        token_counter=E5TokenCounter(),
    )

    load_seconds = round(time.perf_counter() - started, 2)
    runtime = AppRuntime(
        settings=settings,
        pipeline=pipeline,
        kb_version=version,
        synchronizer=synchronizer,
        load_seconds=load_seconds,
        warmup_ms=warmup_ms,
        rss_startup_mb=rss_mb(),
    )
    logger.info(
        "app_runtime_ready kb_source=%s chunks=%d load_seconds=%.2f "
        "warmup_ms=%.1f rss_mb=%s",
        settings.kb.source,
        version.chunk_count,
        load_seconds,
        warmup_ms,
        runtime.rss_startup_mb,
    )
    return runtime


@st.cache_resource(show_spinner=False, max_entries=1)
def get_runtime() -> AppRuntime:
    return build_runtime(load_app_settings())


def active_runtime() -> AppRuntime:
    """Runtime for this run; a factory override exists only for AppTest."""
    factory = st.session_state.get(RUNTIME_OVERRIDE_KEY)
    if factory is not None:
        return factory()
    return get_runtime()


# ---------------------------------------------------------------------------
# Session state helpers


def _init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("message_times", [])
    st.session_state.setdefault("last_message_at", None)


def logout() -> None:
    """Clear the whole session, including the displayed chat history."""
    st.session_state.clear()
    _init_state()


def quota_message(now: Optional[float] = None) -> Optional[str]:
    """Cooldown and hourly cap for the shared Groq allowance."""
    now = time.time() if now is None else now

    last = st.session_state.get("last_message_at")
    if last is not None:
        elapsed = now - last
        if elapsed < config.SESSION_COOLDOWN_SECONDS:
            seconds = int(config.SESSION_COOLDOWN_SECONDS - elapsed) + 1
            return T.COOLDOWN.format(seconds=seconds)

    window_start = now - 3600
    recent = [stamp for stamp in st.session_state.get("message_times", []) if stamp > window_start]
    st.session_state["message_times"] = recent
    if len(recent) >= config.SESSION_HOURLY_MESSAGE_CAP:
        return T.HOURLY_LIMIT
    return None


def _record_message(now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    st.session_state["last_message_at"] = now
    st.session_state.setdefault("message_times", []).append(now)

    history = st.session_state.get("chat_history", [])
    if len(history) > config.CHAT_HISTORY_MAX_TURNS * 2:
        st.session_state["chat_history"] = history[-config.CHAT_HISTORY_MAX_TURNS * 2 :]


# ---------------------------------------------------------------------------
# UI


def render_login(settings: Settings) -> None:
    st.title(T.LOGIN_TITLE)
    st.caption(T.LOGIN_SUBTITLE)

    with st.form("login_form"):
        username = st.text_input(T.LOGIN_USERNAME_LABEL)
        password = st.text_input(T.LOGIN_PASSWORD_LABEL, type="password")
        submitted = st.form_submit_button(T.LOGIN_BUTTON)

    if not submitted:
        return

    try:
        manager = get_auth_manager(
            settings.auth.username, settings.auth.password_hash
        )
    except AuthConfigurationError:
        logger.warning("app_auth_misconfigured")
        st.error(T.LOGIN_AUTH_ERROR)
        return

    remaining = manager.lockout_remaining()
    if remaining > 0:
        st.error(T.LOGIN_LOCKED.format(minutes=max(1, int(remaining // 60) + 1)))
        return

    if manager.authenticate(username, password):
        st.session_state["authenticated"] = True
        st.session_state["chat_history"] = []
        st.session_state["last_message_at"] = None
        st.session_state["message_times"] = []
        st.rerun()
        return

    remaining = manager.lockout_remaining()
    if remaining > 0:
        st.error(T.LOGIN_LOCKED.format(minutes=max(1, int(remaining // 60) + 1)))
    else:
        st.error(T.LOGIN_FAILED)


def _render_history() -> None:
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _append(role: str, content: str) -> None:
    st.session_state.setdefault("chat_history", []).append(
        {"role": role, "content": content}
    )


def answer_question(runtime: AppRuntime, question: str) -> str:
    """Run one pipeline call with the shared lock and turn failures into text."""
    started = time.perf_counter()
    try:
        with runtime.lock:
            result = runtime.pipeline.run(question)
        answer = result.answer
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "app_turn status=%s mode=%s chunks=%d latency_ms=%.1f",
            result.status.value,
            result.debug.mode,
            len(result.debug.retrieved),
            elapsed_ms,
        )
        if runtime.first_query_ms is None:
            runtime.first_query_ms = round(elapsed_ms, 1)
            logger.info(
                "first_query latency_ms=%.1f rss_mb=%s",
                elapsed_ms,
                rss_mb(),
            )
        return answer
    except Exception as exc:
        logger.warning("app_turn_failed error_type=%s", type(exc).__name__)
        return T.TECHNICAL_ERROR


def _handle_question(runtime: AppRuntime, question: str) -> None:
    text = (question or "").strip()
    if not text:
        st.warning(T.EMPTY_QUESTION)
        return

    _append("user", text[: config.CHAT_MAX_INPUT_CHARS])

    limit_text = quota_message()
    if limit_text:
        _append("assistant", limit_text)
        st.rerun()
        return

    with st.chat_message("user"):
        st.markdown(text[: config.CHAT_MAX_INPUT_CHARS])

    with st.chat_message("assistant"):
        with st.spinner(T.THINKING):
            answer = answer_question(runtime, text[: config.CHAT_MAX_INPUT_CHARS])
        st.markdown(answer)

    _append("assistant", answer)
    _record_message()


def _render_examples(runtime: AppRuntime) -> None:
    st.caption(T.EXAMPLES_TITLE)
    columns = st.columns(len(T.EXAMPLE_QUESTIONS))
    for column, question in zip(columns, T.EXAMPLE_QUESTIONS):
        with column:
            if st.button(question, use_container_width=True):
                _handle_question(runtime, question)
                st.rerun()


def render_chat(runtime: AppRuntime) -> None:
    with st.sidebar:
        st.caption(T.SIDEBAR_NOTE)
        if st.button(T.LOGOUT_BUTTON):
            logout()
            st.rerun()

    st.title(T.CHAT_TITLE)
    st.caption(T.INTRO)

    _render_history()

    question = st.chat_input(
        T.CHAT_PLACEHOLDER, max_chars=config.CHAT_MAX_INPUT_CHARS
    )
    if question:
        _handle_question(runtime, question)
        st.rerun()

    _render_examples(runtime)


def main() -> None:
    st.set_page_config(
        page_title=config.APP_TITLE,
        layout=config.APP_LAYOUT,
    )
    _init_state()

    try:
        settings = load_app_settings()
    except SettingsError:
        logger.warning("app_settings_incomplete")
        st.title(T.LOGIN_TITLE)
        st.error(T.LOGIN_CONFIG_ERROR)
        return

    if not st.session_state.get("authenticated"):
        render_login(settings)
        return

    try:
        runtime = active_runtime()
    except (KBSyncError, SettingsError, WarmupError, SQLiteVersionError):
        logger.warning("app_runtime_unavailable")
        st.title(T.CHAT_TITLE)
        st.error(T.TECHNICAL_ERROR)
        return

    render_chat(runtime)


if __name__ == "__main__":
    main()
