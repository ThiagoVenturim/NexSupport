"""Shared fakes for the integration tests: an invented OAuth app, a fake token server and a clock.

All values are synthetic. Nothing here touches the network.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx

CLIENT_ID = "synthetic-client-id"
CLIENT_SECRET = "synthetic-client-secret-0123456789"
REDIRECT = "http://127.0.0.1:8765/callback"
ACCESS_1 = "access-token-one-abcdef0123456789"
ACCESS_2 = "access-token-two-abcdef0123456789"
REFRESH_1 = "refresh-token-one-abcdef0123456789"
REFRESH_2 = "refresh-token-two-abcdef0123456789"
CODE = "synthetic-authorization-code"
SECRETS = (CLIENT_SECRET, ACCESS_1, ACCESS_2, REFRESH_1, REFRESH_2, CODE)


def env_for(provider: str, **extra: str) -> dict[str, str]:
    key = f"SUPPORT_ENGINE_OAUTH_{provider.upper()}_"
    values = {
        key + "CLIENT_ID": CLIENT_ID,
        key + "CLIENT_SECRET": CLIENT_SECRET,
        key + "REDIRECT_URI": REDIRECT,
    }
    values.update({key + k: v for k, v in extra.items()})
    return values


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TokenServer:
    """Answers the token endpoint: authorization_code gives token 1, refresh_token gives token 2."""

    def __init__(self, *, scope: str | None = "tickets:read", rotate: bool = True, expires_in: int = 3600) -> None:
        self.scope = scope
        self.rotate = rotate
        self.expires_in = expires_in
        self.requests: list[dict[str, str]] = []

    def body(self, request: httpx.Request) -> dict[str, str]:
        raw = request.content.decode("utf-8")
        if request.headers.get("content-type", "").startswith("application/json"):
            return json.loads(raw)
        return {k: v[0] for k, v in parse_qs(raw).items()}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = self.body(request)
        self.requests.append(body)
        if body.get("client_secret") != CLIENT_SECRET:
            return httpx.Response(401, json={"error": "invalid_client", "error_description": body.get("client_secret")})
        grant = body.get("grant_type")
        payload: dict[str, object] = {"token_type": "bearer", "expires_in": self.expires_in}
        if self.scope is not None:
            payload["scope"] = self.scope
        if grant == "authorization_code":
            if body.get("code") != CODE:
                return httpx.Response(400, json={"error": "invalid_grant", "error_description": body.get("code")})
            payload.update(access_token=ACCESS_1, refresh_token=REFRESH_1)
        elif grant == "refresh_token":
            if body.get("refresh_token") not in (REFRESH_1, REFRESH_2):
                return httpx.Response(400, json={"error": "invalid_grant"})
            payload["access_token"] = ACCESS_2
            if self.rotate:
                payload["refresh_token"] = REFRESH_2
        else:
            return httpx.Response(400, json={"error": "unsupported_grant_type"})
        return httpx.Response(200, json=payload)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))
