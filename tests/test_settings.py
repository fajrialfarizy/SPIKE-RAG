"""
Offline tests for the framework-agnostic settings object (Phase 3).
"""

from pathlib import Path

import pytest

import config
from core.settings import SettingsError, load_settings

SECRETS = {
    "auth": {"username": "labprestasi", "password_hash": "pbkdf2_sha256$600000$c2FsdA==$aGFzaA=="},
    "groq": {"api_key": "gsk_placeholder"},
    "kb": {"source": "local", "local_dir": "data/kb"},
}


def test_settings_are_built_from_the_secrets_mapping():
    settings = load_settings(SECRETS, environ={})

    assert settings.auth.username == "labprestasi"
    assert settings.auth.password_hash.startswith("pbkdf2_sha256$")
    assert settings.groq.api_key == "gsk_placeholder"
    assert settings.kb.source == config.KB_SOURCE_LOCAL
    assert settings.kb.local_dir.name == "kb"
    assert settings.kb.manifest_ttl_seconds == config.KB_MANIFEST_TTL_SECONDS


def test_environment_values_fill_missing_secrets_for_local_development():
    settings = load_settings(
        {},
        environ={
            config.SETTINGS_ENV_AUTH_USERNAME: "dev",
            config.SETTINGS_ENV_AUTH_PASSWORD_HASH: "pbkdf2_sha256$600000$c2FsdA==$aGFzaA==",
            config.LLM_API_KEY_ENV_VAR: "gsk_from_env",
            config.SETTINGS_ENV_KB_SOURCE: "local",
            config.SETTINGS_ENV_KB_LOCAL_DIR: "chroma_db",
        },
    )

    assert settings.auth.username == "dev"
    assert settings.groq.api_key == "gsk_from_env"
    assert settings.kb.local_dir.name == "chroma_db"


@pytest.mark.parametrize(
    "secrets,missing",
    [
        ({"groq": {"api_key": "k"}, "kb": {}}, "username"),
        ({"auth": {"username": "u"}, "groq": {"api_key": "k"}}, "password_hash"),
        ({"auth": {"username": "u", "password_hash": "h"}}, "api_key"),
    ],
)
def test_a_missing_required_value_names_the_key(secrets, missing):
    with pytest.raises(SettingsError) as excinfo:
        load_settings(secrets, environ={})

    assert missing in str(excinfo.value)


def test_an_unknown_kb_source_is_rejected():
    secrets = dict(SECRETS)
    secrets["kb"] = {"source": "s3"}

    with pytest.raises(SettingsError) as excinfo:
        load_settings(secrets, environ={})

    assert "source" in str(excinfo.value)


def test_drive_requires_a_folder_id():
    secrets = dict(SECRETS)
    secrets["kb"] = {"source": "drive"}

    with pytest.raises(SettingsError):
        load_settings(secrets, environ={})

    secrets["kb"] = {"source": "drive", "drive_folder_id": "folder-123"}
    settings = load_settings(secrets, environ={})
    assert settings.kb.drive_folder_id == "folder-123"


def test_secrets_never_appear_in_repr():
    settings = load_settings(SECRETS, environ={})

    text = repr(settings)
    assert "gsk_placeholder" not in text
    assert settings.auth.password_hash not in text


def test_the_service_account_section_is_carried_through():
    secrets = dict(SECRETS)
    secrets["gcp_service_account"] = {
        "client_email": "sa@example.iam.gserviceaccount.com",
        "private_key": "-----BEGIN PRIVATE KEY-----placeholder",
    }

    settings = load_settings(secrets, environ={})

    assert settings.gcp_service_account["client_email"].startswith("sa@")


def test_the_local_manifest_path_defaults_to_the_artifact_bundle():
    settings = load_settings({
        "auth": SECRETS["auth"],
        "groq": SECRETS["groq"],
        "kb": {"source": "local"},
    }, environ={})

    assert settings.kb.manifest_path == config.ARTIFACT_MANIFEST_PATH


def test_the_local_manifest_path_can_come_from_the_environment():
    settings = load_settings(
        {"auth": SECRETS["auth"], "groq": SECRETS["groq"]},
        environ={"KB_MANIFEST_PATH": "artifacts/other.json"},
    )

    assert settings.kb.manifest_path == Path("artifacts/other.json")
