"""NexLink client: build an envelope, attach the token of the target and send it.

Transports:

- ``InProcessTransport(server)``: calls a ``NexLinkServer`` in the same process (tests, examples).
- ``HttpTransport(url)``: ``POST <url>/nexlink`` with ``Authorization: Bearer <token>``. The
  token travels only in the header; the JSON body never holds it.

The token of each target comes from ``SUPPORT_ENGINE_NEXLINK_TOKEN_<TARGET>`` (environment or
``.env``) unless the caller passes one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import httpx

from support_engine.nexlink.protocol import Envelope, ErrorCode, Reply
from support_engine.nexlink.security import outgoing_token
from support_engine.nexlink.server import NexLinkServer

logger = logging.getLogger(__name__)

PATH = "/nexlink"


class Transport(Protocol):
    def send(self, envelope: Envelope, token: str) -> Reply: ...


class InProcessTransport:
    def __init__(self, server: NexLinkServer) -> None:
        self.server = server

    def send(self, envelope: Envelope, token: str) -> Reply:
        return self.server.handle_sync(envelope, token)


class HttpTransport:
    """NexLink over HTTP. Pass ``client`` to reuse a pool or, in tests, an in-process ASGI client."""

    def __init__(self, base_url: str, *, client: httpx.Client | None = None, timeout: float = 30.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("The NexLink URL must start with http:// or https://")
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout, trust_env=False)

    def send(self, envelope: Envelope, token: str) -> Reply:
        body = envelope.model_dump(mode="json")
        try:
            response = self._client.post(
                f"{self.base_url}{PATH}", json=body, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            logger.warning("nexlink http send failed: %s", type(exc).__name__)
            return envelope.reply_error(envelope.target, ErrorCode.UNAVAILABLE, "The target is not reachable.")
        try:
            return Reply.model_validate(response.json())
        except ValueError:
            return envelope.reply_error(
                envelope.target, ErrorCode.UPSTREAM_ERROR, f"Unexpected answer (HTTP {response.status_code})."
            )


class NexLinkClient:
    """Sends requests and notices on behalf of ``sender``."""

    def __init__(
        self,
        sender: str,
        transport: Transport,
        *,
        tokens: Mapping[str, str] | None = None,
        environ: Mapping[str, str] | None = None,
        env_file: str | Path | None = ".env",
    ) -> None:
        self.sender = sender
        self.transport = transport
        self._tokens = dict(tokens or {})
        self._environ = environ
        self._env_file = env_file

    def __repr__(self) -> str:
        return f"NexLinkClient(sender={self.sender!r}, targets={sorted(self._tokens)!r}, tokens=***)"

    def _token(self, target: str) -> str:
        if target not in self._tokens:
            self._tokens[target] = outgoing_token(target, environ=self._environ, env_file=self._env_file)
        return self._tokens[target]

    def envelope(
        self,
        target: str,
        action: str,
        params: Mapping[str, Any] | None = None,
        *,
        kind: str = "request",
        correlation_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> Envelope:
        return Envelope(
            sender=self.sender,
            target=target,
            action=action,
            params=dict(params or {}),
            kind=kind,
            correlation_id=correlation_id,
            context=dict(context or {}),
        )

    def request(self, target: str, action: str, params: Mapping[str, Any] | None = None, **kwargs: Any) -> Reply:
        """Ask ``target`` to run ``action`` and wait for the reply."""
        envelope = self.envelope(target, action, params, **kwargs)
        return self.transport.send(envelope, self._token(target))

    def notify(self, target: str, action: str, params: Mapping[str, Any] | None = None, **kwargs: Any) -> Reply:
        """Tell ``target`` something; the reply only confirms it was accepted and carries no result."""
        return self.request(target, action, params, kind="notice", **kwargs)

    def manifest(self, target: str) -> Reply:
        return self.request(target, "nexlink.manifest")
