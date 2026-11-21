"""
Shared-account authentication (Phase 3).

Password storage is stdlib PBKDF2-HMAC-SHA256 with a random per-password salt and
a self-describing string:

    pbkdf2_sha256$<iterations>$<base64 salt>$<base64 digest>

OWASP Password Storage Cheat Sheet (checked 2026-11-20) recommends at least
600,000 iterations for PBKDF2-HMAC-SHA256; the value lives in config.

The password and the digest are compared with ``hmac.compare_digest`` and are
never logged. Both the per-attempt failure counter and the lockout are shared
across sessions (the app keeps one AuthManager in ``st.cache_resource``).

Trade-off: a global lockout also lets an anonymous visitor deny access to the
shared account for a few minutes. Per-session counters would be trivially
bypassed by a new session, so the global counter is the safer default for a
single shared login; the duration is short and config-owned.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import threading
import time
from typing import Callable, Optional

import config

logger = logging.getLogger("core_auth")


class AuthError(RuntimeError):
    """Raised when a stored hash is malformed or weaker than config allows."""


class AuthConfigurationError(AuthError):
    """Raised when the stored credential cannot be used to authenticate anyone."""


def hash_password(
    password: str,
    *,
    iterations: int = config.AUTH_PBKDF2_ITERATIONS,
    salt: Optional[bytes] = None,
) -> str:
    """Return the self-describing PBKDF2 hash for one password."""
    if not isinstance(password, str) or not password:
        raise AuthError("Password must be a non-empty string.")
    if iterations < config.AUTH_MIN_PBKDF2_ITERATIONS:
        raise AuthError(
            f"PBKDF2 iterations must be at least {config.AUTH_MIN_PBKDF2_ITERATIONS}."
        )

    salt_bytes = salt if salt is not None else os.urandom(config.AUTH_PBKDF2_SALT_BYTES)
    if not salt_bytes:
        raise AuthError("Salt must not be empty.")

    digest = hashlib.pbkdf2_hmac(
        config.AUTH_PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt_bytes,
        iterations,
    )
    return "$".join(
        (
            config.AUTH_HASH_SCHEME,
            str(iterations),
            base64.b64encode(salt_bytes).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )
    )


def parse_hash(encoded: str) -> tuple:
    """Split a stored hash into (iterations, salt, digest), or fail loudly."""
    if not isinstance(encoded, str) or not encoded:
        raise AuthError("Stored password hash is empty.")

    parts = encoded.split("$")
    if len(parts) != 4:
        raise AuthError("Stored password hash is not in scheme$iterations$salt$digest form.")

    scheme, iterations_text, salt_text, digest_text = parts
    if scheme != config.AUTH_HASH_SCHEME:
        raise AuthError(f"Unsupported password hash scheme '{scheme}'.")

    try:
        iterations = int(iterations_text)
    except ValueError as exc:
        raise AuthError("Stored password hash has a non-numeric iteration count.") from exc
    if iterations < config.AUTH_MIN_PBKDF2_ITERATIONS:
        raise AuthError(
            "Stored password hash uses fewer iterations than config allows; "
            "regenerate it with scripts/make_password_hash.py."
        )

    try:
        salt = base64.b64decode(salt_text.encode("ascii"), validate=True)
        digest = base64.b64decode(digest_text.encode("ascii"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AuthError("Stored password hash is not valid base64.") from exc

    if not salt or not digest:
        raise AuthError("Stored password hash is missing its salt or digest.")
    return iterations, salt, digest


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time comparison of a candidate password against a stored hash."""
    iterations, salt, expected = parse_hash(encoded)
    candidate = hashlib.pbkdf2_hmac(
        config.AUTH_PBKDF2_ALGORITHM,
        (password or "").encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)


class AuthManager:
    """Shared lockout state around one stored credential."""

    def __init__(
        self,
        username: str,
        password_hash: str,
        *,
        max_failures: int = config.AUTH_MAX_FAILED_ATTEMPTS,
        lockout_seconds: Optional[float] = None,
        failure_delay_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if not username:
            raise AuthConfigurationError("Auth username must not be empty.")
        # Validate the stored hash now: a malformed credential must never look
        # like a wrong password.
        try:
            parse_hash(password_hash)
        except AuthError as exc:
            raise AuthConfigurationError(str(exc)) from exc

        self.username = username
        self._password_hash = password_hash
        self.max_failures = int(max_failures)
        self.lockout_seconds = float(
            config.AUTH_LOCKOUT_DURATION_SECONDS
            if lockout_seconds is None
            else lockout_seconds
        )
        self.failure_delay_seconds = float(
            config.AUTH_FAILURE_DELAY_SECONDS
            if failure_delay_seconds is None
            else failure_delay_seconds
        )
        self._clock = clock
        self._sleep = sleep_fn

        self._lock = threading.Lock()
        self._failures = 0
        self._locked_until = 0.0

    # -- lockout state -----------------------------------------------------
    def lockout_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._locked_until - self._clock())

    @property
    def is_locked(self) -> bool:
        return self.lockout_remaining() > 0.0

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._locked_until = 0.0

    # -- authentication ----------------------------------------------------
    def authenticate(self, username: str, password: str) -> bool:
        """Check one attempt. Returns False while locked, without any hashing."""
        with self._lock:
            if self._locked_until > self._clock():
                return False
            if self._locked_until and self._locked_until <= self._clock():
                self._failures = 0
                self._locked_until = 0.0

        username_ok = hmac.compare_digest(str(username or ""), self.username)
        password_ok = verify_password(password or "", self._password_hash)

        if username_ok and password_ok:
            self.reset()
            logger.info("auth_success")
            return True

        self._register_failure()
        # Fixed delay outside the lock so concurrent attempts queue fairly.
        self._sleep(self.failure_delay_seconds)
        logger.info("auth_failure")
        return False

    def _register_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.max_failures:
                self._failures = 0
                self._locked_until = self._clock() + self.lockout_seconds
                logger.warning("auth_lockout seconds=%.0f", self.lockout_seconds)
