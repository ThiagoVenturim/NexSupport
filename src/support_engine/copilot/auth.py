"""Access token of the REST API and the MCP HTTP endpoint.

The token comes from an environment variable (``SUPPORT_ENGINE_API_TOKEN`` by default) or, when
the variable is not set, from a ``.env`` file. It is never part of the code, never a default and
never written to a log. Requests send it as ``Authorization: Bearer <token>`` or ``X-API-Key``;
the comparison runs in constant time.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Mapping
from pathlib import Path


class TokenError(RuntimeError):
    """No usable access token is configured. The message never contains the token."""


def read_env_file(path: str | Path) -> dict[str, str]:
    """``KEY=value`` pairs of a dotenv file; comments, blank lines and ``export`` prefixes are handled."""
    file = Path(path)
    if not file.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_token(
    env_var: str = "SUPPORT_ENGINE_API_TOKEN",
    env_file: str | Path | None = ".env",
    *,
    environ: Mapping[str, str] | None = None,
    min_chars: int = 16,
) -> str:
    """The configured token; raises ``TokenError`` when it is missing or shorter than ``min_chars``."""
    env = os.environ if environ is None else environ
    token = (env.get(env_var) or "").strip()
    if not token and env_file is not None:
        token = read_env_file(env_file).get(env_var, "").strip()
    if not token:
        raise TokenError(f"Set {env_var} in the environment or in the .env file to enable the API.")
    if len(token) < min_chars:
        raise TokenError(f"{env_var} is too short: use at least {min_chars} characters.")
    return token


class TokenAuth:
    """Checks request credentials against one configured token."""

    def __init__(self, token: str, *, min_chars: int = 16) -> None:
        if not token or len(token) < min_chars:
            raise TokenError(f"The access token must have at least {min_chars} characters.")
        self._token = token.encode("utf-8")

    def __repr__(self) -> str:
        return "TokenAuth(token=***)"

    @staticmethod
    def credential(headers: Mapping[str, str]) -> str | None:
        """Token sent in ``Authorization: Bearer`` or ``X-API-Key``; header names are case insensitive."""
        lowered = {k.lower(): v for k, v in headers.items()}
        auth = lowered.get("authorization", "")
        if auth[:7].lower() == "bearer ":
            value = auth[7:].strip()
            if value:
                return value
        key = lowered.get("x-api-key", "").strip()
        return key or None

    def verify(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.encode("utf-8"), self._token)

    def check(self, headers: Mapping[str, str]) -> bool:
        return self.verify(self.credential(headers))
