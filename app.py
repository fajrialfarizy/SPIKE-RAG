"""
SPIKE RAG — Streamlit entry point (Phase 3).

Layer boundaries: this module reads ``st.secrets``/environment, builds a
``core.settings.Settings`` and owns the UI, the session state and the cached
resources. ``core`` never imports streamlit.

Before a successful login nothing heavy is touched: no embedder, no Chroma, no
Groq client and no Drive call. Those live behind ``get_runtime()``, which is only
reached from the authenticated chat screen.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import streamlit as st

import app_texts as T
import config
from core import build_info, startup
from core.sqlite_compat import SQLiteVersionError, ensure_sqlite_version

# Fix pack 13a: one INFO handler on stdout, so the stage lines and the single
# error line reach the cloud log, which keeps stdout.
startup.configure_logging()

# ONE line naming the deployed commit, so the cloud log says which code is live.
# The module remembers it already ran, so a Streamlit rerun does not repeat it.
build_info.log_build_once()

# Must happen before chromadb is imported (it needs SQLite >= 3.35).
_SQLITE_FAILURE: Optional[BaseException] = None
SQLITE_VERSION: Optional[str] = None
try:
    with startup.stage(startup.STAGE_SQLITE):
        SQLITE_VERSION = ensure_sqlite_version()
except SQLiteVersionError as exc:
    # chromadb needs that SQLite too, so importing the core modules below would
    # abort this script before any screen exists. The failure is kept for main().
    _SQLITE_FAILURE = exc

if _SQLITE_FAILURE is None:
    from core.auth import AuthConfigurationError, AuthManager
    from core.kb_sync import DriveKBSource, KBSynchronizer, KBSyncError, KBVersion
    from core.llm_client import GroqLLMClient
    from core.pipeline import AnswerStatus, RagPipeline, configure_file_logging
    from core.rate_limiter import TokenAwareRateLimiter, get_token_limiter
    from core.settings import Settings, SettingsError, load_settings
    from core.tokens import E5TokenCounter
    from core.warmup import WarmupError, warm_up_embedder, warm_up_knowledge_base
    from ingestion.embedder import E5Embedder

logger = logging.getLogger("app")

SETTINGS_OVERRIDE_KEY = "_settings_override"
RUNTIME_OVERRIDE_KEY = "_runtime_factory"
# AppTest injects a limiter double here; the script runner re-executes app.py, so
# a patched module attribute would not reach the running script.
QUEUE_LIMITER_OVERRIDE_KEY = "_queue_limiter"


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
    kb_load_seconds: float = 0.0
    embedder_load_seconds: float = 0.0
    embedding_runtime: str = config.EMBEDDING_RUNTIME
    warmup_ms: float = 0.0
    rss_startup_mb: Optional[float] = None
    first_query_ms: Optional[float] = None


def build_runtime(settings: Settings) -> AppRuntime:
    """Load the knowledge base, the embedder and the LLM client (post-login)."""
    started = time.perf_counter()

    # Fix pack 13a: every step is bracketed so the cloud log shows stage_start and
    # stage_done (with seconds and RSS), and a killed process still names the step
    # it died in.
    with startup.stage(startup.STAGE_RUNTIME_INIT):
        configure_file_logging(config.LOG_DIR)
        configure_app_logging(config.LOG_DIR)
        startup.log_environment_once()

    with startup.stage(startup.STAGE_KB_INIT):
        source = None
        if settings.kb.source == config.KB_SOURCE_DRIVE:
            source = DriveKBSource(
                folder_id=settings.kb.drive_folder_id or "",
                service_account_info=settings.gcp_service_account,
                manifest_name=settings.kb.drive_manifest_name,
                manifest_file_id=settings.kb.drive_manifest_file_id,
                artifact_file_id=settings.kb.drive_artifact_file_id,
            )
        synchronizer = KBSynchronizer(settings.kb, source=source)

    kb_started = time.perf_counter()
    with startup.stage(startup.STAGE_KB_LOAD):
        version = synchronizer.current()
        if version.handle is None:
            raise KBSyncError("The knowledge base was not validated by its loader.")
    kb_load_seconds = round(time.perf_counter() - kb_started, 2)

    embedder = E5Embedder()
    with startup.stage(startup.STAGE_EMBEDDER_WARMUP):
        warmup_ms = warm_up_embedder(embedder)
    embedder_load_seconds = float(getattr(embedder, "load_seconds", 0.0) or 0.0)

    # Fix pack 7: the limiter counts the static prompt parts with this tokenizer.
    # Load it during startup, so a cold load can never be charged to a question's
    # deadline. The tokenizer itself is cached per model inside core.tokens.
    token_started = time.perf_counter()
    with startup.stage(startup.STAGE_TOKENIZER_LOAD):
        token_counter = E5TokenCounter()
        token_counter.count_tokens(config.EMBEDDING_WARMUP_QUERY)
    logger.info(
        "tokenizer_warmup_ms=%.1f",
        (time.perf_counter() - token_started) * 1000.0,
    )

    # Fix pack 8 (N4): the Groq client import and construction cost seconds and
    # used to happen inside the first question's (untimed) post-processing. Build
    # it here, during startup, like the embedder and the tokenizer.
    with startup.stage(startup.STAGE_LLM_CLIENT):
        llm_client = GroqLLMClient(api_key=settings.groq.api_key)
        llm_warmup_ms = llm_client.warm_up()
    logger.info("llm_client_warmup_ms=%.1f", llm_warmup_ms)

    # Fix pack 9 (O6a): the first question still paid 1,240 ms for the first
    # Chroma query and the metadata read. Exercise both here; no model call.
    with startup.stage(startup.STAGE_RETRIEVAL_WARMUP):
        retrieval_warmup = warm_up_knowledge_base(version.handle, embedder)
    logger.info("retrieval_warmup=%s", retrieval_warmup)

    with startup.stage(startup.STAGE_PIPELINE_BUILD):
        pipeline = RagPipeline(
            knowledge_base=version.handle,
            llm=llm_client,
            embedder=embedder,
            token_counter=token_counter,
        )

    load_seconds = round(time.perf_counter() - started, 2)
    runtime = AppRuntime(
        settings=settings,
        pipeline=pipeline,
        kb_version=version,
        synchronizer=synchronizer,
        load_seconds=load_seconds,
        kb_load_seconds=kb_load_seconds,
        embedder_load_seconds=embedder_load_seconds,
        warmup_ms=warmup_ms,
        rss_startup_mb=rss_mb(),
    )
    # ONE startup line (fix pack P3): enough to judge memory and cold start from
    # Streamlit Cloud logs without reading anything else.
    logger.info(
        "app_startup embedding_runtime=%s embedding_model=%s kb_version=%s "
        "kb_source=%s chunks=%d kb_load_seconds=%.2f embedder_load_seconds=%.2f "
        "embedder_warmup_ms=%.1f total_load_seconds=%.2f rss_mb=%s",
        runtime.embedding_runtime,
        config.EMBEDDING_MODEL_NAME,
        version.version,
        settings.kb.source,
        version.chunk_count,
        kb_load_seconds,
        embedder_load_seconds,
        warmup_ms,
        load_seconds,
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
    """Cooldown, hourly cap and daily cap for this browser session."""
    now = time.time() if now is None else now

    last = st.session_state.get("last_message_at")
    if last is not None:
        elapsed = now - last
        if elapsed < config.SESSION_COOLDOWN_SECONDS:
            seconds = int(config.SESSION_COOLDOWN_SECONDS - elapsed) + 1
            return T.COOLDOWN.format(seconds=seconds)

    all_times = list(st.session_state.get("message_times", []))
    window_start = now - 3600
    recent = [stamp for stamp in all_times if stamp > window_start]
    st.session_state["message_times"] = recent
    if len(recent) >= config.SESSION_HOURLY_MESSAGE_CAP:
        return T.HOURLY_LIMIT

    day_start = now - config.SESSION_DAY_WINDOW_SECONDS
    daily = [stamp for stamp in all_times if stamp > day_start]
    if len(daily) >= config.SESSION_MAX_QUESTIONS_PER_DAY:
        return T.SESSION_DAILY_LIMIT
    return None


def queue_wait_seconds(limiter: Optional[TokenAwareRateLimiter] = None) -> float:
    """
    Predicted wait before the answer call's tokens fit the minute window.

    Read-only: it never reserves tokens, so a message cannot change the quota the
    question then consumes. The estimate uses the configured token caps, which is
    the same figure the client reserves for the answer call.
    """
    try:
        limiter = limiter or st.session_state.get(QUEUE_LIMITER_OVERRIDE_KEY)
        limiter = limiter or get_token_limiter()
        return float(limiter.wait_estimate(config.QUEUE_MESSAGE_ESTIMATE_TOKENS))
    except Exception:  # a prediction must never block a question
        return 0.0


def queue_message(
    limiter: Optional[TokenAwareRateLimiter] = None,
    now: Optional[float] = None,
) -> Optional[str]:
    """The queue notice for this question, or None when the wait is short."""
    wait = queue_wait_seconds(limiter)
    if wait <= config.QUEUE_MESSAGE_MIN_WAIT_SECONDS:
        return None
    return T.QUEUE_WAIT.format(seconds=int(wait) + 1)


def _record_message(now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    st.session_state["last_message_at"] = now
    st.session_state.setdefault("message_times", []).append(now)

    history = st.session_state.get("chat_history", [])
    if len(history) > config.CHAT_HISTORY_MAX_TURNS * 2:
        st.session_state["chat_history"] = history[-config.CHAT_HISTORY_MAX_TURNS * 2 :]


# ---------------------------------------------------------------------------
# UI


LOGO_PATH = Path(__file__).resolve().parent / config.THEME_LOGO_PATH

THEME_CSS = """
<style>
  /* Inter is loaded from the web font CDN; the stack in config falls back to
     the local sans-serif faces when that request fails. */
  @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
  html, body, .stApp,
  .stApp [data-testid="stMarkdownContainer"],
  .stApp input, .stApp textarea, .stApp button,
  .stApp [data-testid="stChatInput"] textarea {
      font-family: %(font)s;
  }
  /* Logo blue: bright accent on top, navy for headings and chrome. */
  .stApp h1 {
      color: %(navy)s;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 0.15rem;
  }
  .spike-tagline {
      color: %(navy)s;
      font-size: 1.02rem;
      font-weight: 500;
      margin: 0 0 1.3rem 0;
      padding-left: 0.8rem;
      border-left: 4px solid %(primary)s;
  }
  .stApp div.stButton > button,
  .stApp div.stFormSubmitButton > button {
      background-color: %(primary)s;
      border: 1px solid %(primary)s;
      border-radius: 8px;
      color: #ffffff;
      font-weight: 600;
  }
  .stApp div.stButton > button:hover,
  .stApp div.stFormSubmitButton > button:hover {
      background-color: %(navy)s;
      border-color: %(navy)s;
      color: #ffffff;
  }
  .stApp div.stButton > button p, .stApp div.stFormSubmitButton > button p {
      color: inherit;
  }
  section[data-testid="stSidebar"] { background-color: %(navy)s; }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span { color: #ffffff; }
  section[data-testid="stSidebar"] div.stButton > button {
      background-color: #ffffff;
      border-color: #ffffff;
  }
  section[data-testid="stSidebar"] div.stButton > button p { color: %(navy)s; }
  .stApp [data-testid="stChatInput"] { border-color: %(primary)s; }
  .stApp [data-testid="stChatInputSubmitButton"] { color: %(primary)s; }
  .stApp [data-testid="stChatMessage"] { border-radius: 12px; }
</style>
"""


def apply_theme() -> None:
    """Inject the logo palette and the Inter font; config owns both."""
    st.markdown(
        THEME_CSS
        % {
            "primary": config.THEME_PRIMARY_COLOR,
            "navy": config.THEME_PRIMARY_DARK_COLOR,
            "font": config.THEME_FONT_FAMILY,
        },
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Logo beside the app name, with the smaller tagline underneath."""
    logo_column, text_column = st.columns([1, 7], vertical_alignment="center")
    with logo_column:
        st.image(str(LOGO_PATH), width=84)
    with text_column:
        st.title(T.APP_NAME)
        st.markdown(
            f'<p class="spike-tagline">{T.TAGLINE}</p>', unsafe_allow_html=True
        )


def render_login(settings: Settings) -> None:
    render_header()

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


def retry_after_text(seconds: Optional[float]) -> Optional[str]:
    """
    "Silakan coba lagi dalam sekitar N detik", rounded up to the configured step.

    Returns None when the limiter did not say how long the wait would be (fix
    pack 7): the pipeline text is then shown unchanged.
    """
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    step = config.RETRY_AFTER_ROUNDING_SECONDS
    rounded = int(math.ceil(value / step) * step)
    return T.RETRY_AFTER.format(seconds=rounded)


def answer_question(runtime: AppRuntime, question: str) -> str:
    """Run one pipeline call with the shared lock and turn failures into text."""
    started = time.perf_counter()
    try:
        with runtime.lock:
            result = runtime.pipeline.run(question)
        answer = result.answer
        if result.status is AnswerStatus.RATE_LIMITED:
            notice = retry_after_text(result.retry_after_seconds)
            if notice:
                answer = f"{answer}\n\n{notice}"
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
        with st.status(T.THINKING, expanded=False) as status:
            notice = queue_message()
            if notice:
                status.write(notice)
            answer = answer_question(runtime, text[: config.CHAT_MAX_INPUT_CHARS])
            status.update(label=T.THINKING, state="complete", expanded=False)
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

    render_header()
    st.caption(T.INTRO)

    _render_history()

    question = st.chat_input(
        T.CHAT_PLACEHOLDER, max_chars=config.CHAT_MAX_INPUT_CHARS
    )
    if question:
        # No extra rerun here: the answer and its processing status were rendered
        # in this run, and a rerun would discard the status block (fix pack J4).
        _handle_question(runtime, question)

    _render_examples(runtime)


def _render_startup_failure(exc: BaseException) -> None:
    """Log the single whitelisted error line and show its code under our text."""
    failure = startup.describe_failure(exc)
    startup.log_failure(failure)
    render_header()
    st.error(T.with_error_code(T.TECHNICAL_ERROR, failure.code))


def main() -> None:
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=str(LOGO_PATH),
        layout=config.APP_LAYOUT,
    )
    apply_theme()
    _init_state()

    if _SQLITE_FAILURE is not None:
        _render_startup_failure(_SQLITE_FAILURE)
        return

    try:
        with startup.stage(startup.STAGE_SETTINGS):
            settings = load_app_settings()
    except SettingsError as exc:
        failure = startup.describe_failure(exc)
        startup.log_failure(failure)
        render_header()
        st.error(T.with_error_code(T.LOGIN_CONFIG_ERROR, failure.code))
        return

    if not st.session_state.get("authenticated"):
        render_login(settings)
        return

    try:
        runtime = active_runtime()
    except Exception as exc:
        # Broad on purpose (fix pack 13a): any failure while building the
        # authenticated screen must name its stage in the log and its code on
        # screen, instead of dying silently behind our generic message.
        _render_startup_failure(exc)
        return

    render_chat(runtime)


if __name__ == "__main__":
    main()
