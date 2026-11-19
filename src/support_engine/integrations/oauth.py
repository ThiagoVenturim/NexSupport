"""OAuth 2.0 authorization code flow with PKCE and refresh tokens for help desk APIs.

Security rules this module follows:

- Client secrets and tokens are read from the environment, a ``.env`` file or a vault adapter
  (see ``support_engine.integrations.tokens``); nothing has a default value in code.
- ``repr`` of every object that holds a secret hides it, and error messages carry the HTTP
  status and the OAuth ``error`` code only, never a token, a secret or the response body.
- The ``state`` value is single use, expires and is compared in constant time.
- PKCE (S256) protects the authorization code; the code verifier never leaves this process.
- The token endpoint must use HTTPS. The redirect URI must use HTTPS, except a loopback address.
- The flow asks for the minimal scopes of the provider and rejects a grant that holds scopes
  that were not requested.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

logger = logging.getLogger(__name__)

BodyFormat = Literal["form", "json"]
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
STATE_TTL_SECONDS = 600.0
EXPIRY_LEEWAY_SECONDS = 60.0


class OAuthError(RuntimeError):
    """An OAuth step failed. The message never contains a token, a code or a client secret."""


def _check_https(url: str, what: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise OAuthError(f"The {what} must be an https URL.")
    return url


def check_redirect_uri(uri: str) -> str:
    """HTTPS, or plain HTTP on a loopback address for desktop and CLI logins."""
    parts = urlsplit(uri)
    if parts.fragment:
        raise OAuthError("The redirect URI must not contain a fragment.")
    if parts.scheme == "https" and parts.hostname:
        return uri
    if parts.scheme == "http" and parts.hostname in LOOPBACK_HOSTS:
        return uri
    raise OAuthError("The redirect URI must be https, or http on localhost.")


def parse_scope(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Scopes from a space or comma separated string or an iterable, without duplicates."""
    if value is None:
        return ()
    items = value.replace(",", " ").split() if isinstance(value, str) else [str(v).strip() for v in value]
    seen: dict[str, None] = {}
    for item in items:
        if item:
            seen.setdefault(item, None)
    return tuple(seen)


@dataclass(frozen=True)
class OAuthProvider:
    """Endpoints and minimal scopes of one platform."""

    name: str
    authorize_url: str
    token_url: str
    scopes: tuple[str, ...] = ()
    body_format: BodyFormat = "form"
    extra_authorize_params: Mapping[str, str] = field(default_factory=dict)
    pkce: bool = True

    def __post_init__(self) -> None:
        _check_https(self.authorize_url, f"authorize URL of {self.name}")
        _check_https(self.token_url, f"token URL of {self.name}")


@dataclass(frozen=True)
class OAuthClient:
    """Registered OAuth application. ``repr`` hides the secret."""

    client_id: str
    client_secret: str = field(repr=False)
    redirect_uri: str

    def __post_init__(self) -> None:
        if not self.client_id or not self.client_secret:
            raise OAuthError("The OAuth client id and client secret are required.")
        check_redirect_uri(self.redirect_uri)

    def __repr__(self) -> str:
        return f"OAuthClient(client_id={self.client_id!r}, client_secret=***, redirect_uri={self.redirect_uri!r})"


@dataclass(frozen=True)
class OAuthToken:
    """Access and refresh token; ``repr`` and ``str`` hide both."""

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    scopes: tuple[str, ...] = ()
    token_type: str = "Bearer"

    def __post_init__(self) -> None:
        if not self.access_token:
            raise OAuthError("The token response has no access token.")

    def __repr__(self) -> str:
        refresh = "***" if self.refresh_token else None
        return (
            f"OAuthToken(access_token=***, refresh_token={refresh}, expires_at={self.expires_at!r}, "
            f"scopes={list(self.scopes)!r})"
        )

    __str__ = __repr__

    def expired(self, now: float, leeway: float = EXPIRY_LEEWAY_SECONDS) -> bool:
        return self.expires_at is not None and now >= self.expires_at - leeway

    @property
    def can_refresh(self) -> bool:
        return bool(self.refresh_token)


@dataclass(frozen=True)
class AuthorizationRequest:
    """URL the user opens and the ``state`` to expect back. The PKCE verifier stays private."""

    url: str
    state: str
    expires_at: float
    code_verifier: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"AuthorizationRequest(url={self.url!r}, state=***, expires_at={self.expires_at!r})"


def pkce_pair() -> tuple[str, str]:
    """``(verifier, S256 challenge)`` as in RFC 7636."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def constant_time_equals(left: str | None, right: str | None) -> bool:
    """Compare two credentials without leaking where they differ."""
    if left is None or right is None:
        return False
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


class OAuthFlow:
    """Authorization code flow, code exchange and refresh for one provider and one client."""

    def __init__(
        self,
        provider: OAuthProvider,
        client: OAuthClient,
        *,
        scopes: Iterable[str] | None = None,
        http: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
        state_ttl: float = STATE_TTL_SECONDS,
        timeout: float = 30.0,
    ) -> None:
        self.provider = provider
        self.client = client
        self.scopes = parse_scope(scopes) if scopes is not None else provider.scopes
        self._http = http
        self._clock = clock
        self._state_ttl = state_ttl
        self._timeout = timeout
        self._pending: list[AuthorizationRequest] = []

    def __repr__(self) -> str:
        return f"OAuthFlow(provider={self.provider.name!r}, scopes={list(self.scopes)!r})"

    # Authorization --------------------------------------------------------------------------

    def start(self) -> AuthorizationRequest:
        """New authorization URL with a fresh ``state`` and, when supported, a PKCE challenge."""
        now = self._clock()
        self._pending = [p for p in self._pending if p.expires_at > now]
        state = secrets.token_urlsafe(32)
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.client.client_id,
            "redirect_uri": self.client.redirect_uri,
            "state": state,
        }
        if self.scopes:
            params["scope"] = " ".join(self.scopes)
        verifier: str | None = None
        if self.provider.pkce:
            verifier, challenge = pkce_pair()
            params["code_challenge"] = challenge
            params["code_challenge_method"] = "S256"
        params.update(self.provider.extra_authorize_params)
        sep = "&" if urlsplit(self.provider.authorize_url).query else "?"
        request = AuthorizationRequest(
            url=f"{self.provider.authorize_url}{sep}{urlencode(params)}",
            state=state,
            expires_at=now + self._state_ttl,
            code_verifier=verifier,
        )
        self._pending.append(request)
        return request

    def _take_pending(self, state: str | None) -> AuthorizationRequest:
        now = self._clock()
        match: AuthorizationRequest | None = None
        # Compare against every pending state so the time taken does not depend on which one matches.
        for pending in self._pending:
            if constant_time_equals(state, pending.state):
                match = pending
        if match is None:
            raise OAuthError("Unknown or reused OAuth state.")
        self._pending = [p for p in self._pending if p is not match]
        if match.expires_at <= now:
            raise OAuthError("The OAuth state expired; start the login again.")
        return match

    def finish(self, state: str | None, code: str | None, *, error: str | None = None) -> OAuthToken:
        """Check ``state`` (single use, constant time) and exchange ``code`` for a token."""
        pending = self._take_pending(state)
        if error:
            raise OAuthError(f"The authorization was denied ({_safe_code(error)}).")
        if not code:
            raise OAuthError("The redirect has no authorization code.")
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.client.redirect_uri,
        }
        if pending.code_verifier:
            data["code_verifier"] = pending.code_verifier
        token = self._token_request(data)
        logger.info("oauth %s: authorization code exchanged (scopes: %s)", self.provider.name, " ".join(token.scopes))
        return token

    def finish_from_redirect(self, redirect_url: str) -> OAuthToken:
        """``finish`` with the ``state`` and ``code`` taken from the URL the browser was sent to."""
        expected = urlsplit(self.client.redirect_uri)
        actual = urlsplit(redirect_url.strip())
        if (actual.scheme, actual.hostname, actual.port, actual.path) != (
            expected.scheme,
            expected.hostname,
            expected.port,
            expected.path,
        ):
            raise OAuthError("The redirect URL does not match the configured redirect URI.")
        query = parse_qs(actual.query)
        first = {k: v[0] for k, v in query.items() if v}
        return self.finish(first.get("state"), first.get("code"), error=first.get("error"))

    # Refresh --------------------------------------------------------------------------------

    def refresh(self, token: OAuthToken) -> OAuthToken:
        """New access token from the refresh token. Keeps the old refresh token when not rotated."""
        if not token.refresh_token:
            raise OAuthError(f"The {self.provider.name} token cannot be refreshed; log in again.")
        data = {"grant_type": "refresh_token", "refresh_token": token.refresh_token}
        fresh = self._token_request(data, previous=token)
        if fresh.refresh_token is None:
            fresh = replace(fresh, refresh_token=token.refresh_token)
        logger.info("oauth %s: access token refreshed", self.provider.name)
        return fresh

    # Token endpoint -------------------------------------------------------------------------

    def _token_request(self, data: dict[str, str], previous: OAuthToken | None = None) -> OAuthToken:
        body = {**data, "client_id": self.client.client_id, "client_secret": self.client.client_secret}
        headers = {"Accept": "application/json"}
        owns = self._http is None
        http = self._http or httpx.Client(timeout=self._timeout)
        try:
            try:
                if self.provider.body_format == "json":
                    response = http.post(self.provider.token_url, json=body, headers=headers)
                else:
                    response = http.post(self.provider.token_url, data=body, headers=headers)
            except httpx.HTTPError as exc:
                raise OAuthError(f"The {self.provider.name} token endpoint failed ({type(exc).__name__}).") from None
            payload = _json_or_empty(response)
            if response.status_code >= 400:
                raise OAuthError(
                    f"The {self.provider.name} token endpoint answered {response.status_code} "
                    f"({_safe_code(payload.get('error'))})."
                )
            return self._parse_token(payload, previous)
        finally:
            if owns:
                http.close()

    def _parse_token(self, payload: Mapping[str, Any], previous: OAuthToken | None) -> OAuthToken:
        access = payload.get("access_token") or payload.get("token")
        if not isinstance(access, str) or not access:
            raise OAuthError(f"The {self.provider.name} token response has no access token.")
        token_type = str(payload.get("token_type") or "Bearer")
        if token_type.lower() != "bearer":
            raise OAuthError(f"Unsupported token type from {self.provider.name}.")
        granted = parse_scope(payload.get("scope")) if payload.get("scope") else ()
        if not granted:
            granted = previous.scopes if previous is not None else self.scopes
        extra = sorted(set(granted) - set(self.scopes)) if self.scopes else []
        if extra:
            raise OAuthError(
                f"{self.provider.name} granted scopes that were not requested: {', '.join(extra)}. "
                "Reduce the scopes of the OAuth app."
            )
        expires_at: float | None = None
        expires_in = payload.get("expires_in")
        if expires_in not in (None, ""):
            try:
                expires_at = self._clock() + float(expires_in)
            except (TypeError, ValueError):
                raise OAuthError(f"Invalid expires_in from {self.provider.name}.") from None
        refresh = payload.get("refresh_token")
        return OAuthToken(
            access_token=access,
            refresh_token=refresh if isinstance(refresh, str) and refresh else None,
            expires_at=expires_at,
            scopes=granted,
            token_type="Bearer",
        )


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_code(value: Any) -> str:
    """OAuth error codes are short ASCII words; anything else is replaced so nothing leaks."""
    text = str(value or "unknown")
    if len(text) > 64 or not all(c.isalnum() or c in "_-." for c in text):
        return "unknown"
    return text


__all__ = [
    "AuthorizationRequest",
    "OAuthClient",
    "OAuthError",
    "OAuthFlow",
    "OAuthProvider",
    "OAuthToken",
    "check_redirect_uri",
    "constant_time_equals",
    "parse_scope",
    "pkce_pair",
]
