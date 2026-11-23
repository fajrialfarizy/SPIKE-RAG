"""
Startup diagnostics for the post-login runtime build (fix pack 13a).

The cloud log stopped at "Uvicorn server started" because the failure happens
while the authenticated screen is built, and only the exception class reached the
log. This module adds visibility and nothing else: no fallback, no retry and no
change to what the app does.

Three pieces, all offline and testable:

- ``configure_logging`` attaches ONE INFO handler on stdout, so every line lands
  in the cloud log.
- ``stage`` brackets each step of the runtime build with ``stage_start`` and
  ``stage_done name=<...> seconds=<...> rss_mb=<...>``. A process killed mid-build
  therefore leaves the last started stage behind.
- ``describe_failure``/``log_failure`` turn any exception into ONE line
  (``error_stage``, ``error_class``, ``http_status``, ``reason``) plus a short
  whitelisted code that the UI shows under its generic message.

Only whitelisted values are ever written: stage names, exception class names, an
HTTP status, a Google error reason and, for settings, key names. Never a value, an
id, a Drive file name, document text or a traceback.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import config

logger = logging.getLogger("core_startup")

# Stage names (W1 order). Nothing outside this list is ever written as a stage.
STAGE_SETTINGS = "settings"
STAGE_SQLITE = "sqlite"
STAGE_RUNTIME_INIT = "runtime_init"
STAGE_KB_INIT = "kb_init"
STAGE_KB_LOAD = "kb_load"
STAGE_DRIVE_AUTH = "drive_auth"
STAGE_MANIFEST_FETCH = "manifest_fetch"
STAGE_ZIP_DOWNLOAD = "zip_download"
STAGE_CHECKSUM = "checksum"
STAGE_EXTRACT = "extract"
STAGE_MANIFEST_VALIDATE = "manifest_validate"
STAGE_CHROMA_OPEN = "chroma_open"
STAGE_MODEL_VALIDATE = "model_validate"
STAGE_TOKENIZER_LOAD = "tokenizer_load"
STAGE_EMBEDDER_LOAD = "embedder_load"
STAGE_EMBEDDER_WARMUP = "embedder_warmup"
STAGE_RETRIEVAL_WARMUP = "retrieval_warmup"
STAGE_LLM_CLIENT = "llm_client"
STAGE_PIPELINE_BUILD = "pipeline_build"

STAGE_ORDER = (
    STAGE_SETTINGS,
    STAGE_SQLITE,
    STAGE_RUNTIME_INIT,
    STAGE_KB_INIT,
    STAGE_KB_LOAD,
    STAGE_DRIVE_AUTH,
    STAGE_MANIFEST_FETCH,
    STAGE_ZIP_DOWNLOAD,
    STAGE_CHECKSUM,
    STAGE_EXTRACT,
    STAGE_MANIFEST_VALIDATE,
    STAGE_CHROMA_OPEN,
    STAGE_MODEL_VALIDATE,
    STAGE_TOKENIZER_LOAD,
    STAGE_EMBEDDER_LOAD,
    STAGE_EMBEDDER_WARMUP,
    STAGE_RETRIEVAL_WARMUP,
    STAGE_LLM_CLIENT,
    STAGE_PIPELINE_BUILD,
)

# Codes the UI may show. "UNK/<ExceptionClass>" is the only open-ended one.
CODE_DRIVE_FORBIDDEN = "KB/drive_forbidden"
CODE_DRIVE_API_DISABLED = "KB/drive_api_disabled"
CODE_DRIVE_NOT_FOUND = "KB/drive_not_found"
CODE_ARTIFACT_DUPLICATE = "KB/duplicate_artifact"
CODE_ARTIFACT_MISSING = "KB/missing_artifact"
CODE_CHECKSUM_MISMATCH = "KB/checksum_mismatch"
CODE_MANIFEST_MISMATCH = "KB/manifest_mismatch"
CODE_MODEL_MISMATCH = "KB/model_mismatch"
CODE_EXTRACT_FAILED = "KB/extract_failed"
CODE_SQLITE_VERSION = "DB/sqlite_version"
CODE_CHROMA_OPEN = "DB/chroma_open"
CODE_DOWNLOAD_FAILED = "EMB/download_failed"
CODE_LOAD_FAILED = "EMB/load_failed"
CODE_WARMUP = "RT/warmup"
CODE_SETTINGS_KEY_PREFIX = "CFG/missing_key:"
CODE_UNKNOWN_PREFIX = "UNK/"

API_DISABLED_REASONS = ("accessNotConfigured", "SERVICE_DISABLED")

# A stage that raises an unclassified exception still gets the code of the work it
# was doing. Deliberately absent: download and manifest fetch, where an unexpected
# failure is reported honestly as UNK/<class>.
_STAGE_CODE_DEFAULTS = {
    STAGE_SQLITE: CODE_SQLITE_VERSION,
    STAGE_CHROMA_OPEN: CODE_CHROMA_OPEN,
    STAGE_MODEL_VALIDATE: CODE_MODEL_MISMATCH,
    STAGE_MANIFEST_VALIDATE: CODE_MANIFEST_MISMATCH,
    STAGE_CHECKSUM: CODE_CHECKSUM_MISMATCH,
    STAGE_EXTRACT: CODE_EXTRACT_FAILED,
    STAGE_EMBEDDER_LOAD: CODE_LOAD_FAILED,
    STAGE_EMBEDDER_WARMUP: CODE_WARMUP,
    STAGE_RETRIEVAL_WARMUP: CODE_WARMUP,
    STAGE_TOKENIZER_LOAD: CODE_LOAD_FAILED,
}

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_CLASS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,79}$")
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_./:<>=+-]{1,80}$")
_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_.:+/-]{1,64}$")
_REASON_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

_LOCAL = threading.local()
_LOG_LOCK = threading.Lock()
_LOGGING_READY = False
_ENVIRONMENT_LOGGED = False


# ---------------------------------------------------------------------------
# Logging


def configure_logging(
    level: int = logging.INFO, stream: Optional[Any] = None
) -> logging.Logger:
    """
    Attach one stdout handler to the root logger, once per process.

    Streamlit Cloud keeps stdout, so a line written here survives a redeploy; the
    handler flushes on every record, so the last ``stage_start`` also survives a
    process that is killed before it can write anything else.
    """
    global _LOGGING_READY
    with _LOG_LOCK:
        if _LOGGING_READY:
            return logger
        handler = logging.StreamHandler(sys.stdout if stream is None else stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        handler.setLevel(level)
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)
        _LOGGING_READY = True
    return logger


def _stack() -> List[str]:
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    return stack


def current_stage(exc: Optional[BaseException] = None) -> Optional[str]:
    """
    Innermost stage still open, else the one that was unwinding.

    With ``exc`` the recorded stage only counts when it belongs to THAT exception,
    so an earlier failure cannot mislabel a later one.
    """
    stack = _stack()
    if stack:
        return stack[-1]
    failed_stage = getattr(_LOCAL, "failed_stage", None)
    if failed_stage is None:
        return None
    if exc is None:
        return failed_stage
    return failed_stage if getattr(_LOCAL, "failed_exc", None) is exc else None


def rss_mb() -> Optional[float]:
    """Resident memory in MiB, when psutil is available."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - requirements install psutil
        return None
    try:
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:  # pragma: no cover - platform dependent
        return None


@contextmanager
def stage(name: str, *, clock: Any = time.perf_counter) -> Iterator[None]:
    """Log ``stage_start``/``stage_done`` around one step of the build."""
    if not _NAME_PATTERN.match(name):
        raise ValueError(f"Stage name {name!r} is not whitelisted.")
    stack = _stack()
    stack.append(name)
    logger.info("stage_start name=%s", name)
    started = clock()
    try:
        yield
    except BaseException as exc:
        # The innermost stage to see the exception names it, even after unwinding.
        if getattr(_LOCAL, "failed_exc", None) is not exc:
            _LOCAL.failed_exc = exc
            _LOCAL.failed_stage = name
        raise
    else:
        logger.info(
            "stage_done name=%s seconds=%.2f rss_mb=%s",
            name,
            clock() - started,
            rss_mb(),
        )
    finally:
        stack.pop()


# ---------------------------------------------------------------------------
# Failure description


@dataclass(frozen=True)
class Failure:
    """One whitelisted description of a failure; safe to log and to show."""

    stage: str
    code: str
    error_class: str
    http_status: Optional[int] = None
    reason: Optional[str] = None


def _clean(value: Any, pattern: "re.Pattern[str]", fallback: Optional[str]) -> Any:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if pattern.match(text) else fallback


def google_error_details(exc: BaseException) -> Tuple[Optional[int], Optional[str]]:
    """
    HTTP status and Google error reason, without touching the message text.

    ``http_status``/``reason`` attributes set by this project win; otherwise the
    ``googleapiclient.errors.HttpError`` shape is read (``resp.status`` and the
    ``error.errors[0].reason`` of the JSON body).
    """
    status: Any = getattr(exc, "http_status", None)
    reason = getattr(exc, "reason", None)

    if status is None:
        status = getattr(getattr(exc, "resp", None), "status", None)

    if not isinstance(reason, str):
        reason = _google_reason(exc)

    status_int: Optional[int] = None
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None
    return status_int, reason


def _google_reason(exc: BaseException) -> Optional[str]:
    details = getattr(exc, "error_details", None)
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and isinstance(item.get("reason"), str):
                return item["reason"]

    content = getattr(exc, "content", None)
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8", "replace")
        except Exception:  # pragma: no cover - defensive
            return None
    if not isinstance(content, str) or not content:
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    errors = error.get("errors")
    if isinstance(errors, list):
        for item in errors:
            if isinstance(item, dict) and isinstance(item.get("reason"), str):
                return item["reason"]
    for key in ("status", "reason"):
        value = error.get(key)
        if isinstance(value, str):
            return value
    return None


def google_drive_code(
    status: Optional[int], reason: Optional[str]
) -> Optional[str]:
    """Map a Drive HTTP status/reason to one whitelisted code."""
    if status == 403:
        if reason in API_DISABLED_REASONS:
            return CODE_DRIVE_API_DISABLED
        return CODE_DRIVE_FORBIDDEN
    if status == 401:
        return CODE_DRIVE_FORBIDDEN
    if status == 404:
        return CODE_DRIVE_NOT_FOUND
    return None


def _failure_code(exc: BaseException, stage_name: Optional[str]) -> str:
    explicit = getattr(exc, "code", None)
    if isinstance(explicit, str) and _CODE_PATTERN.match(explicit):
        return explicit

    status, reason = google_error_details(exc)
    drive_code = google_drive_code(status, reason)
    if drive_code:
        return drive_code

    default = _STAGE_CODE_DEFAULTS.get(stage_name or "")
    if default:
        return default

    error_class = _clean(type(exc).__name__, _CLASS_PATTERN, fallback="Exception")
    return f"{CODE_UNKNOWN_PREFIX}{error_class}"


def describe_failure(exc: BaseException) -> Failure:
    """Everything the log and the UI may know about this failure."""
    status, reason = google_error_details(exc)
    stage = current_stage(exc)
    return Failure(
        stage=_clean(stage, _NAME_PATTERN, fallback="unknown"),
        code=_failure_code(exc, stage),
        error_class=_clean(type(exc).__name__, _CLASS_PATTERN, fallback="Exception"),
        http_status=status,
        reason=_clean(reason, _REASON_PATTERN, fallback=None),
    )


def format_failure(failure: Failure) -> str:
    """The single ``error_*`` line. Values are already whitelisted."""
    return (
        f"error_stage={failure.stage} error_class={failure.error_class} "
        f"http_status={'-' if failure.http_status is None else failure.http_status} "
        f"reason={failure.reason or '-'} code={failure.code}"
    )


def log_failure(failure: Failure) -> None:
    logger.warning("%s", format_failure(failure))


# ---------------------------------------------------------------------------
# Environment facts


def _sqlite_facts() -> Tuple[str, bool]:
    module = sys.modules.get("sqlite3")
    version = getattr(module, "sqlite_version", None) or "unavailable"
    shim = bool(
        module is not None
        and str(getattr(module, "__name__", "")).startswith("pysqlite3")
    )
    return str(version), shim


def _virtual_memory_mb() -> Tuple[str, str]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return (
            f"{memory.total / (1024 * 1024):.0f}",
            f"{memory.available / (1024 * 1024):.0f}",
        )
    except Exception:  # pragma: no cover - platform dependent
        return "-", "-"


def _read_cgroup_bytes(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
    except OSError:
        return None
    if raw in ("", "max"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_memory_mb() -> Tuple[str, str]:
    """
    Container memory limit and usage when readable.

    cgroup v2 first, then v1; "unavailable" everywhere else (Windows, macOS, a
    host without cgroup) instead of raising.
    """
    pairs = (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    )
    for limit_path, usage_path in pairs:
        limit = _read_cgroup_bytes(limit_path)
        usage = _read_cgroup_bytes(usage_path)
        if limit is None and usage is None:
            continue
        limit_text = "unavailable" if limit is None else f"{limit / (1024 * 1024):.0f}"
        usage_text = "unavailable" if usage is None else f"{usage / (1024 * 1024):.0f}"
        return limit_text, usage_text
    return "unavailable", "unavailable"


def _temp_free_mb() -> str:
    try:
        free = shutil.disk_usage(tempfile.gettempdir()).free
        return f"{free / (1024 * 1024):.0f}"
    except Exception:  # pragma: no cover - platform dependent
        return "unavailable"


def environment_facts() -> Dict[str, str]:
    """Cheap environment facts; no secret and no environment-variable value."""
    facts: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "cpu_count": str(os.cpu_count() or "unknown"),
    }

    sqlite_version, shim = _sqlite_facts()
    facts["sqlite"] = sqlite_version
    facts["sqlite_shim"] = "true" if shim else "false"

    facts["torch"] = "unavailable"
    facts["cuda"] = "unavailable"
    try:
        import torch

        facts["torch"] = str(torch.__version__)
        facts["cuda"] = "true" if torch.cuda.is_available() else "false"
    except Exception:  # torch is optional here and must never break startup
        pass

    rss = rss_mb()
    facts["rss_mb"] = "-" if rss is None else f"{rss:.1f}"

    total, available = _virtual_memory_mb()
    facts["vm_total_mb"] = total
    facts["vm_available_mb"] = available

    limit, usage = _cgroup_memory_mb()
    facts["cgroup_limit_mb"] = limit
    facts["cgroup_used_mb"] = usage

    facts["temp_free_mb"] = _temp_free_mb()
    return facts


def log_environment_once(*, force: bool = False) -> Dict[str, str]:
    """Write ``startup_env ...`` once per process (tests pass ``force``)."""
    global _ENVIRONMENT_LOGGED
    with _LOG_LOCK:
        if _ENVIRONMENT_LOGGED and not force:
            return {}
        _ENVIRONMENT_LOGGED = True

    facts = environment_facts()
    logger.info(
        "startup_env %s",
        " ".join(
            f"{key}={_clean(value, _VALUE_PATTERN, fallback='-')}"
            for key, value in facts.items()
        ),
    )
    return facts
