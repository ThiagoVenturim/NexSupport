"""Where OAuth client settings and tokens live: the environment, a ``.env`` file or a vault.

Variables, per provider (``ZENDESK``, ``FRESHDESK``, ``INTERCOM``, ``JIRA_SM``)::

    SUPPORT_ENGINE_OAUTH_<PROVIDER>_CLIENT_ID
    SUPPORT_ENGINE_OAUTH_<PROVIDER>_CLIENT_SECRET
    SUPPORT_ENGINE_OAUTH_<PROVIDER>_REDIRECT_URI
    SUPPORT_ENGINE_OAUTH_<PROVIDER>_ACCESS_TOKEN      written by the login
    SUPPORT_ENGINE_OAUTH_<PROVIDER>_REFRESH_TOKEN     written by the login and by each refresh
    SUPPORT_ENGINE_OAUTH_<PROVIDER>_EXPIRES_AT        UNIX time
    SUPPORT_ENGINE_OAUTH_<PROVIDER>_GRANTED_SCOPES

``EnvFileTokenStore`` keeps tokens in a git-ignored ``.env`` file (mode 0600 where the OS allows it).
A vault (Vault, a cloud secret manager, the OS keyring) plugs in by implementing ``TokenStore``.
The environment always wins over the file, so a deployment can inject tokens without a file.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from support_engine.copilot.auth import read_env_file
from support_engine.integrations.oauth import OAuthClient, OAuthError, OAuthToken, parse_scope

PREFIX = "SUPPORT_ENGINE_OAUTH_"
_SAFE_VALUE = re.compile(r"^[\x21-\x7e]+$")
_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=")


def env_key(provider: str, name: str) -> str:
    return f"{PREFIX}{provider.upper().replace('-', '_')}_{name}"


def merged_env(environ: Mapping[str, str] | None = None, env_file: str | Path | None = ".env") -> dict[str, str]:
    """``.env`` values overridden by the environment."""
    values = read_env_file(env_file) if env_file is not None else {}
    values.update(os.environ if environ is None else environ)
    return values


def load_client(
    provider: str,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = ".env",
) -> OAuthClient:
    """OAuth application settings of ``provider``; raises ``OAuthError`` naming the missing variable."""
    values = merged_env(environ, env_file)
    found: dict[str, str] = {}
    for name in ("CLIENT_ID", "CLIENT_SECRET", "REDIRECT_URI"):
        key = env_key(provider, name)
        value = values.get(key, "").strip()
        if not value:
            raise OAuthError(f"Set {key} in the environment or in the .env file.")
        found[name] = value
    return OAuthClient(found["CLIENT_ID"], found["CLIENT_SECRET"], found["REDIRECT_URI"])


class TokenStore(Protocol):
    """Persists one token per provider. Implementations must never log token values."""

    def load(self, provider: str) -> OAuthToken | None: ...

    def save(self, provider: str, token: OAuthToken) -> None: ...


class MemoryTokenStore:
    """Tokens kept in memory only; for tests and short-lived processes."""

    def __init__(self, tokens: Mapping[str, OAuthToken] | None = None) -> None:
        self._tokens: dict[str, OAuthToken] = dict(tokens or {})

    def __repr__(self) -> str:
        return f"MemoryTokenStore(providers={sorted(self._tokens)!r})"

    def load(self, provider: str) -> OAuthToken | None:
        return self._tokens.get(provider)

    def save(self, provider: str, token: OAuthToken) -> None:
        self._tokens[provider] = token


def _check_value(value: str, what: str) -> str:
    # Tokens are opaque printable ASCII; anything else could break or inject lines into the file.
    if not _SAFE_VALUE.match(value) or "#" in value or "'" in value or '"' in value:
        raise OAuthError(f"The {what} has characters that cannot be stored in a .env file.")
    return value


class EnvFileTokenStore:
    """Tokens read from the environment or a ``.env`` file and written back to that file only."""

    def __init__(self, path: str | Path = ".env", *, environ: Mapping[str, str] | None = None) -> None:
        self.path = Path(path)
        if not self.path.name.startswith(".env") or self.path.name.endswith(".example"):
            raise OAuthError("Tokens are stored only in a git-ignored .env file.")
        self._environ = environ

    def __repr__(self) -> str:
        return f"EnvFileTokenStore(path={self.path.name!r})"

    def load(self, provider: str) -> OAuthToken | None:
        values = merged_env(self._environ, self.path)
        access = values.get(env_key(provider, "ACCESS_TOKEN"), "").strip()
        if not access:
            return None
        refresh = values.get(env_key(provider, "REFRESH_TOKEN"), "").strip() or None
        raw_expiry = values.get(env_key(provider, "EXPIRES_AT"), "").strip()
        try:
            expires_at = float(raw_expiry) if raw_expiry else None
        except ValueError:
            raise OAuthError(f"{env_key(provider, 'EXPIRES_AT')} must be a UNIX time.") from None
        scopes = parse_scope(values.get(env_key(provider, "GRANTED_SCOPES"), ""))
        return OAuthToken(access_token=access, refresh_token=refresh, expires_at=expires_at, scopes=scopes)

    def save(self, provider: str, token: OAuthToken) -> None:
        updates = {
            env_key(provider, "ACCESS_TOKEN"): _check_value(token.access_token, "access token"),
            env_key(provider, "REFRESH_TOKEN"): (
                _check_value(token.refresh_token, "refresh token") if token.refresh_token else ""
            ),
            env_key(provider, "EXPIRES_AT"): "" if token.expires_at is None else str(int(token.expires_at)),
            env_key(provider, "GRANTED_SCOPES"): ",".join(token.scopes),
        }
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.is_file() else []
        out: list[str] = []
        written: set[str] = set()
        for line in lines:
            match = _KEY.match(line.strip().removeprefix("export ").lstrip())
            key = match.group(1) if match else None
            if key in updates:
                if key not in written:
                    out.append(f"{key}={updates[key]}")
                    written.add(key)
                continue
            out.append(line)
        for key, value in updates.items():
            if key not in written:
                out.append(f"{key}={value}")
        self._write(("\n".join(out) + "\n").encode("utf-8"))

    def _write(self, data: bytes) -> None:
        directory = self.path.parent if str(self.path.parent) else Path(".")
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            try:
                os.chmod(tmp, 0o600)
            except OSError:  # pragma: no cover - file systems without POSIX modes
                pass
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


__all__ = [
    "PREFIX",
    "EnvFileTokenStore",
    "MemoryTokenStore",
    "TokenStore",
    "env_key",
    "load_client",
    "merged_env",
]
