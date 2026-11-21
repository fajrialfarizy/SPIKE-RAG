"""
Generate the shared-account password hash for .streamlit/secrets.toml.

The password is read with getpass: never as a command-line argument, never echoed
and never logged. The hash is the only thing printed.

    python scripts/make_password_hash.py
"""

from __future__ import annotations

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.auth import AuthError, hash_password


def main() -> int:
    password = getpass.getpass("Password: ")
    if len(password) < config.AUTH_MIN_PASSWORD_LENGTH:
        print(
            f"Password must be at least {config.AUTH_MIN_PASSWORD_LENGTH} characters.",
            file=sys.stderr,
        )
        return 1

    repeated = getpass.getpass("Repeat password: ")
    if password != repeated:
        print("The two passwords do not match.", file=sys.stderr)
        return 1

    try:
        encoded = hash_password(password)
    except AuthError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "\nPaste this into .streamlit/secrets.toml as [auth] password_hash "
        f"(scheme={config.AUTH_HASH_SCHEME}, "
        f"iterations={config.AUTH_PBKDF2_ITERATIONS}):\n"
    )
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
