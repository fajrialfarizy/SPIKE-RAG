"""
Offline tests for password hashing and the shared lockout (Phase 3).
"""

import hmac

import pytest

import config
from core import auth
from core.auth import (
    AuthConfigurationError,
    AuthError,
    AuthManager,
    hash_password,
    parse_hash,
    verify_password,
)

PASSWORD = "kata-sandi-panjang-2026"


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# Hashing once keeps the suite fast: PBKDF2 is deliberately slow, and every
# authenticate() call re-derives the digest.
ENCODED = hash_password(PASSWORD, iterations=config.AUTH_MIN_PBKDF2_ITERATIONS)


def make_manager(clock, sleeps=None, max_failures=None, lockout_seconds=None):
    return AuthManager(
        "labprestasi",
        ENCODED,
        max_failures=config.AUTH_MAX_FAILED_ATTEMPTS if max_failures is None else max_failures,
        lockout_seconds=lockout_seconds,
        clock=clock,
        sleep_fn=(sleeps.append if sleeps is not None else (lambda _: None)),
    )


# -- hashing --------------------------------------------------------------
def test_hash_roundtrip_accepts_the_right_password():
    encoded = hash_password(PASSWORD, iterations=config.AUTH_MIN_PBKDF2_ITERATIONS)

    assert encoded.startswith(f"{config.AUTH_HASH_SCHEME}$")
    assert verify_password(PASSWORD, encoded) is True
    assert verify_password(PASSWORD + "x", encoded) is False


def test_the_hash_is_self_describing():
    encoded = hash_password(PASSWORD, iterations=config.AUTH_MIN_PBKDF2_ITERATIONS)
    iterations, salt, digest = parse_hash(encoded)

    assert iterations == config.AUTH_MIN_PBKDF2_ITERATIONS
    assert len(salt) == config.AUTH_PBKDF2_SALT_BYTES
    assert digest


def test_each_hash_uses_a_fresh_salt():
    first = hash_password(PASSWORD, iterations=config.AUTH_MIN_PBKDF2_ITERATIONS)
    second = hash_password(PASSWORD, iterations=config.AUTH_MIN_PBKDF2_ITERATIONS)

    assert first != second
    assert verify_password(PASSWORD, first)
    assert verify_password(PASSWORD, second)


@pytest.mark.parametrize(
    "tampered",
    [
        "",
        "plain$text",
        "pbkdf2_sha1$600000$c2FsdA==$aGFzaA==",
        "pbkdf2_sha256$10$c2FsdA==$aGFzaA==",
        "pbkdf2_sha256$600000$not-base64!$aGFzaA==",
        "pbkdf2_sha256$abc$c2FsdA==$aGFzaA==",
    ],
)
def test_a_tampered_or_weak_hash_fails_loudly(tampered):
    with pytest.raises(AuthError):
        verify_password(PASSWORD, tampered)


def test_a_hash_below_the_iteration_floor_is_rejected():
    with pytest.raises(AuthError):
        hash_password(PASSWORD, iterations=1000)


# -- constant-time comparison --------------------------------------------
def test_username_and_password_are_compared_in_constant_time(monkeypatch):
    calls = []
    real_compare = hmac.compare_digest

    def spy(first, second):
        calls.append((first, second))
        return real_compare(first, second)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    manager = make_manager(FakeClock())

    assert manager.authenticate("labprestasi", PASSWORD) is True
    assert len(calls) >= 2


# -- lockout --------------------------------------------------------------
def test_five_failures_lock_the_shared_account():
    clock = FakeClock()
    sleeps = []
    manager = make_manager(clock, sleeps)

    for _ in range(config.AUTH_MAX_FAILED_ATTEMPTS):
        assert manager.authenticate("labprestasi", "salah") is False

    assert manager.is_locked is True
    assert manager.lockout_remaining() == pytest.approx(
        config.AUTH_LOCKOUT_DURATION_SECONDS
    )
    assert sleeps == [config.AUTH_FAILURE_DELAY_SECONDS] * config.AUTH_MAX_FAILED_ATTEMPTS


def test_a_locked_account_rejects_even_the_right_password_without_delaying():
    clock = FakeClock()
    sleeps = []
    manager = make_manager(clock, sleeps)
    for _ in range(config.AUTH_MAX_FAILED_ATTEMPTS):
        manager.authenticate("labprestasi", "salah")
    sleeps.clear()

    assert manager.authenticate("labprestasi", PASSWORD) is False
    assert sleeps == []


def test_the_lockout_expires_and_gives_a_fresh_budget():
    clock = FakeClock()
    manager = make_manager(clock, max_failures=2)

    for _ in range(2):
        manager.authenticate("labprestasi", "salah")
    clock.advance(config.AUTH_LOCKOUT_DURATION_SECONDS + 0.1)

    assert manager.is_locked is False
    assert manager.authenticate("labprestasi", PASSWORD) is True


def test_a_successful_login_clears_the_failure_counter():
    clock = FakeClock()
    manager = make_manager(clock, max_failures=3)

    for _ in range(2):
        manager.authenticate("labprestasi", "salah")
    assert manager.authenticate("labprestasi", PASSWORD) is True

    for _ in range(2):
        manager.authenticate("labprestasi", "salah")
    assert manager.is_locked is False


def test_a_wrong_username_never_authenticates():
    manager = make_manager(FakeClock())

    assert manager.authenticate("orang-lain", PASSWORD) is False


def test_a_malformed_stored_credential_is_a_configuration_error():
    with pytest.raises(AuthConfigurationError):
        AuthManager("labprestasi", "not-a-hash")
