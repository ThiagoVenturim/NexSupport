"""Who may call the Support Engine over NexLink, and which actions each caller may run.

Each calling project has its own token and its own scope list, read from the environment or a
``.env`` file, never from code or the command line::

    SUPPORT_ENGINE_NEXLINK_CLIENT_NEXHUB_TOKEN=<generated>
    SUPPORT_ENGINE_NEXLINK_CLIENT_NEXHUB_SCOPES=ingest:run,metrics:read
    SUPPORT_ENGINE_NEXLINK_CLIENT_NEXCODER_TOKEN=<generated>
    SUPPORT_ENGINE_NEXLINK_CLIENT_NEXCODER_SCOPES=metrics:read

Every request carries its token. It is checked against the token of the sender it claims, in
constant time, and then against the scope of the action, so a token only opens the operations
listed for that project. Tokens never appear in ``repr``, logs or error messages.

Outgoing calls read the token of the target the same way: ``SUPPORT_ENGINE_NEXLINK_TOKEN_<TARGET>``.
"""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from support_engine.copilot.auth import TokenError, read_env_file
from support_engine.nexlink.protocol import PROJECT_PATTERN

SCOPE_INGEST = "ingest:run"
SCOPE_SUGGEST = "copilot:suggest"
SCOPE_METRICS = "metrics:read"
SCOPES = frozenset({SCOPE_INGEST, SCOPE_SUGGEST, SCOPE_METRICS})

CLIENT_PREFIX = "SUPPORT_ENGINE_NEXLINK_CLIENT_"
OUTGOING_PREFIX = "SUPPORT_ENGINE_NEXLINK_TOKEN_"
MIN_TOKEN_CHARS = 16

_CLIENT_TOKEN = re.compile(rf"^{CLIENT_PREFIX}([A-Z][A-Z0-9_]{{0,31}})_TOKEN$")
_PROJECT = re.compile(PROJECT_PATTERN)


def _env_name(project: str) -> str:
    return project.upper().replace("-", "_")


def _merged_env(environ: Mapping[str, str] | None, env_file: str | Path | None) -> dict[str, str]:
    """``.env`` values overridden by the environment."""
    values = read_env_file(env_file) if env_file is not None else {}
    values.update(os.environ if environ is None else environ)
    return values


def parse_scopes(raw: str | Iterable[str], *, allowed: Iterable[str] = SCOPES) -> frozenset[str]:
    items = [s.strip() for s in raw.split(",")] if isinstance(raw, str) else [s.strip() for s in raw]
    scopes = frozenset(s for s in items if s)
    unknown = scopes - frozenset(allowed)
    if unknown:
        raise TokenError(f"Unknown NexLink scopes: {', '.join(sorted(unknown))}.")
    return scopes


@dataclass(frozen=True)
class Client:
    """A calling project. ``repr`` hides the token."""

    name: str
    scopes: frozenset[str]
    _token: bytes = field(repr=False)

    def __repr__(self) -> str:
        return f"Client(name={self.name!r}, scopes={sorted(self.scopes)!r}, token=***)"

    def matches(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.encode("utf-8"), self._token)

    def allows(self, scope: str | None) -> bool:
        return scope is None or scope in self.scopes


class ClientRegistry:
    """Tokens and scopes of the projects allowed to call this one."""

    def __init__(self, clients: Iterable[Client] = (), *, allowed_scopes: Iterable[str] = SCOPES) -> None:
        self.allowed_scopes = frozenset(allowed_scopes)
        self._clients: dict[str, Client] = {}
        seen: set[bytes] = set()
        for client in clients:
            if client.name in self._clients:
                raise TokenError(f"NexLink client '{client.name}' is configured twice.")
            if client._token in seen:
                raise TokenError(f"NexLink client '{client.name}' reuses the token of another client.")
            seen.add(client._token)
            self._clients[client.name] = client

    def __repr__(self) -> str:
        return f"ClientRegistry(clients={sorted(self._clients)!r})"

    def __len__(self) -> int:
        return len(self._clients)

    @property
    def names(self) -> list[str]:
        return sorted(self._clients)

    @classmethod
    def build(
        cls,
        clients: Mapping[str, tuple[str, str | Iterable[str]]],
        *,
        allowed_scopes: Iterable[str] = SCOPES,
        min_chars: int = MIN_TOKEN_CHARS,
    ) -> ClientRegistry:
        """Registry from ``{project: (token, scopes)}``; for tests and embedding apps."""
        allowed = frozenset(allowed_scopes)
        items = []
        for name, (token, scopes) in clients.items():
            if not _PROJECT.match(name):
                raise TokenError(f"Invalid NexLink client name '{name}'.")
            if not token or len(token) < min_chars:
                raise TokenError(f"The NexLink token of '{name}' must have at least {min_chars} characters.")
            items.append(Client(name, parse_scopes(scopes, allowed=allowed), token.encode("utf-8")))
        return cls(items, allowed_scopes=allowed)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: str | Path | None = ".env",
        *,
        allowed_scopes: Iterable[str] = SCOPES,
        min_chars: int = MIN_TOKEN_CHARS,
    ) -> ClientRegistry:
        """Read ``SUPPORT_ENGINE_NEXLINK_CLIENT_<NAME>_TOKEN`` and ``..._SCOPES`` pairs.

        A client without a scopes variable holds no scope and can only read the manifest.
        """
        values = _merged_env(environ, env_file)
        clients: dict[str, tuple[str, str]] = {}
        for key, token in values.items():
            match = _CLIENT_TOKEN.match(key)
            if not match or not token.strip():
                continue
            env_name = match.group(1)
            project = env_name.lower().replace("_", "-")
            scopes = values.get(f"{CLIENT_PREFIX}{env_name}_SCOPES", "")
            clients[project] = (token.strip(), scopes)
        return cls.build(clients, allowed_scopes=allowed_scopes, min_chars=min_chars)

    def authenticate(self, sender: str, candidate: str | None) -> Client | None:
        """The client when ``candidate`` is the token of ``sender``; None otherwise."""
        client = self._clients.get(sender)
        if client is None:
            # Same amount of work for unknown senders, so timing does not reveal client names.
            hmac.compare_digest(b"-" * MIN_TOKEN_CHARS, (candidate or "").encode("utf-8"))
            return None
        return client if client.matches(candidate) else None


def outgoing_token(
    target: str,
    *,
    env_var: str | None = None,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = ".env",
    min_chars: int = MIN_TOKEN_CHARS,
) -> str:
    """Token this project sends to ``target``; raises ``TokenError`` when it is missing or short."""
    name = env_var or f"{OUTGOING_PREFIX}{_env_name(target)}"
    token = _merged_env(environ, env_file).get(name, "").strip()
    if not token:
        raise TokenError(f"Set {name} in the environment or in the .env file to call {target}.")
    if len(token) < min_chars:
        raise TokenError(f"{name} is too short: use at least {min_chars} characters.")
    return token
