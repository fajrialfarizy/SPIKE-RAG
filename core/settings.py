"""
Framework-agnostic application settings (Phase 3).

The Streamlit layer reads ``st.secrets`` and the environment, builds a plain
mapping and hands it to ``load_settings``. ``core`` never imports streamlit, so
every one of these values is testable offline and the same object works for the
CLI or a future host.

Missing required values fail loudly with ``SettingsError`` naming the key; a value
is never echoed, and the dataclasses keep secrets out of ``repr``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import config


class SettingsError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


@dataclass(frozen=True)
class AuthSettings:
    username: str
    password_hash: str = field(repr=False)


@dataclass(frozen=True)
class GroqSettings:
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class KBSettings:
    source: str
    local_dir: Path
    manifest_path: Path = config.ARTIFACT_MANIFEST_PATH
    drive_folder_id: Optional[str] = None
    # Exact file ids win over name lookup when configured (fix pack P2).
    drive_manifest_file_id: Optional[str] = None
    drive_artifact_file_id: Optional[str] = None
    drive_manifest_name: str = config.KB_DRIVE_MANIFEST_NAME
    drive_artifact_name: str = config.KB_DRIVE_ARTIFACT_NAME
    work_dir: Path = config.KB_WORK_DIR
    manifest_ttl_seconds: float = config.KB_MANIFEST_TTL_SECONDS


@dataclass(frozen=True)
class Settings:
    auth: AuthSettings
    groq: GroqSettings
    kb: KBSettings
    gcp_service_account: Mapping[str, str] = field(default_factory=dict, repr=False)


def _section(source: Mapping[str, Any], name: str) -> dict:
    if not source:
        return {}
    section = source.get(name) if hasattr(source, "get") else None
    if section is None:
        return {}
    try:
        return dict(section)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"Settings section '{name}' is not a table.") from exc


def _pick(
    section: Mapping[str, Any],
    key: str,
    environ: Mapping[str, str],
    env_key: str,
) -> Optional[str]:
    value = section.get(key)
    if value not in (None, ""):
        return str(value)
    env_value = environ.get(env_key)
    if env_value not in (None, ""):
        return str(env_value)
    return None


def _require(value: Optional[str], human_name: str) -> str:
    if value is None:
        raise SettingsError(f"Missing required setting: {human_name}.")
    return value


def load_settings(
    secrets: Mapping[str, Any],
    environ: Optional[Mapping[str, str]] = None,
) -> Settings:
    """Build validated settings from secrets with an environment fallback."""
    environ = os.environ if environ is None else environ

    auth_section = _section(secrets, "auth")
    groq_section = _section(secrets, "groq")
    kb_section = _section(secrets, "kb")
    gcp_section = _section(secrets, "gcp_service_account")

    username = _require(
        _pick(
            auth_section,
            "username",
            environ,
            config.SETTINGS_ENV_AUTH_USERNAME,
        ),
        "[auth] username (or LAB_PRESTASI_USERNAME)",
    )
    password_hash = _require(
        _pick(
            auth_section,
            "password_hash",
            environ,
            config.SETTINGS_ENV_AUTH_PASSWORD_HASH,
        ),
        "[auth] password_hash (or LAB_PRESTASI_PASSWORD_HASH)",
    )
    api_key = _require(
        _pick(groq_section, "api_key", environ, config.LLM_API_KEY_ENV_VAR),
        "[groq] api_key (or GROQ_API_KEY)",
    )

    source = (
        _pick(kb_section, "source", environ, config.SETTINGS_ENV_KB_SOURCE)
        or config.KB_SOURCE_LOCAL
    ).strip().lower()
    if source not in config.KB_SOURCES:
        raise SettingsError(
            f"[kb] source must be one of {config.KB_SOURCES}, not '{source}'."
        )

    local_dir_value = (
        _pick(kb_section, "local_dir", environ, config.SETTINGS_ENV_KB_LOCAL_DIR)
        or str(config.CHROMA_PERSIST_DIR)
    )
    work_dir_value = (
        _pick(kb_section, "work_dir", environ, config.SETTINGS_ENV_KB_WORK_DIR)
        or str(config.KB_WORK_DIR)
    )

    manifest_path_value = (
        _pick(
            kb_section,
            "manifest_path",
            environ,
            config.SETTINGS_ENV_KB_MANIFEST_PATH,
        )
        or str(config.ARTIFACT_MANIFEST_PATH)
    )

    drive_folder_id = _pick(
        kb_section,
        "drive_folder_id",
        environ,
        config.SETTINGS_ENV_KB_DRIVE_FOLDER_ID,
    )
    drive_manifest_file_id = _pick(
        kb_section,
        "manifest_file_id",
        environ,
        config.SETTINGS_ENV_KB_DRIVE_MANIFEST_FILE_ID,
    )
    drive_artifact_file_id = _pick(
        kb_section,
        "artifact_file_id",
        environ,
        config.SETTINGS_ENV_KB_DRIVE_ARTIFACT_FILE_ID,
    )
    if source == config.KB_SOURCE_DRIVE and not drive_folder_id:
        raise SettingsError(
            "[kb] drive_folder_id is required when [kb] source is 'drive'."
        )

    manifest_ttl_raw = kb_section.get("manifest_ttl_seconds")
    try:
        manifest_ttl = float(
            config.KB_MANIFEST_TTL_SECONDS if manifest_ttl_raw is None else manifest_ttl_raw
        )
    except (TypeError, ValueError) as exc:
        raise SettingsError("[kb] manifest_ttl_seconds must be a number.") from exc
    if manifest_ttl < 0:
        raise SettingsError("[kb] manifest_ttl_seconds must not be negative.")

    return Settings(
        auth=AuthSettings(username=username, password_hash=password_hash),
        groq=GroqSettings(api_key=api_key),
        kb=KBSettings(
            source=source,
            local_dir=Path(local_dir_value),
            manifest_path=Path(manifest_path_value),
            drive_folder_id=drive_folder_id,
            drive_manifest_file_id=drive_manifest_file_id,
            drive_artifact_file_id=drive_artifact_file_id,
            drive_manifest_name=str(
                kb_section.get("drive_manifest_name") or config.KB_DRIVE_MANIFEST_NAME
            ),
            drive_artifact_name=str(
                kb_section.get("drive_artifact_name") or config.KB_DRIVE_ARTIFACT_NAME
            ),
            work_dir=Path(work_dir_value),
            manifest_ttl_seconds=manifest_ttl,
        ),
        gcp_service_account=dict(gcp_section),
    )
