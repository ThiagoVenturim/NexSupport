"""HTTP plumbing for the OAuth connectors: bearer auth with refresh, host pinning and rate limits.

- ``OAuthSession`` hands out a valid access token: it refreshes shortly before expiry, saves
  the new token to the ``TokenStore`` and keeps a rotated refresh token.
- ``OAuthBearerAuth`` (an ``httpx.Auth``) adds the token only to requests for the pinned API
  hosts, so a pagination link that points elsewhere never receives it, and retries once after
  a refresh when the API answers 401.
- ``RateLimitTransport`` waits and retries on 429 (and 503 with ``Retry-After``), reading
  ``Retry-After`` (seconds or HTTP date) or ``X-RateLimit-Reset`` / ``RateLimit-Reset``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator, Iterable
from datetime import UTC
from email.utils import parsedate_to_datetime

import httpx

from support_engine.integrations.oauth import OAuthError, OAuthFlow, OAuthToken
from support_engine.integrations.tokens import TokenStore

logger = logging.getLogger(__name__)


class OAuthSession:
    """Current token of one provider, refreshed on demand and persisted after each refresh."""

    def __init__(
        self,
        provider: str,
        store: TokenStore,
        flow: OAuthFlow | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.provider = provider
        self.store = store
        self.flow = flow
        self._clock = clock
        self._token: OAuthToken | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"OAuthSession(provider={self.provider!r})"

    def _current(self) -> OAuthToken:
        if self._token is None:
            self._token = self.store.load(self.provider)
        if self._token is None:
            raise OAuthError(f"No {self.provider} token stored; run the OAuth login first.")
        return self._token

    def token(self) -> OAuthToken:
        """A token that is not about to expire; refreshes when needed."""
        with self._lock:
            token = self._current()
            if token.expired(self._clock()):
                token = self._refresh_locked(token)
            return token

    def refresh(self, stale: OAuthToken | None = None) -> OAuthToken:
        """Refresh now, unless another caller already replaced ``stale``."""
        with self._lock:
            token = self._current()
            if stale is not None and token.access_token != stale.access_token:
                return token
            return self._refresh_locked(token)

    def _refresh_locked(self, token: OAuthToken) -> OAuthToken:
        if self.flow is None or not token.can_refresh:
            raise OAuthError(f"The {self.provider} token expired and cannot be refreshed; log in again.")
        fresh = self.flow.refresh(token)
        self.store.save(self.provider, fresh)
        self._token = fresh
        return fresh

    @property
    def can_refresh(self) -> bool:
        try:
            return self.flow is not None and self._current().can_refresh
        except OAuthError:
            return False


class OAuthBearerAuth(httpx.Auth):
    """Bearer token for pinned hosts, with one refresh and retry on 401."""

    def __init__(self, session: OAuthSession, allowed_hosts: Iterable[str]) -> None:
        self.session = session
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        if not self.allowed_hosts:
            raise OAuthError("At least one API host is required for the OAuth token.")

    def __repr__(self) -> str:
        return f"OAuthBearerAuth(provider={self.session.provider!r}, hosts={sorted(self.allowed_hosts)!r})"

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        host = (request.url.host or "").lower()
        if request.url.scheme != "https" or host not in self.allowed_hosts:
            raise OAuthError("Refusing to send the OAuth token to a host outside the configured API.")
        token = self.session.token()
        request.headers["Authorization"] = f"Bearer {token.access_token}"
        response = yield request
        if response.status_code == 401 and self.session.can_refresh:
            logger.info("oauth %s: 401 from the API, refreshing the token once", self.session.provider)
            fresh = self.session.refresh(stale=token)
            request.headers["Authorization"] = f"Bearer {fresh.access_token}"
            yield request


def retry_after(headers: httpx.Headers, now: float) -> float | None:
    """Seconds to wait from the rate limit headers, or None when they say nothing usable."""
    value = headers.get("retry-after")
    if value:
        value = value.strip()
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            when = None
        if when is not None:
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            return max(0.0, when.timestamp() - now)
    for name in ("x-ratelimit-reset", "ratelimit-reset", "x-rate-limit-reset"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            number = float(raw.strip())
        except ValueError:
            continue
        # Large values are UNIX times (Intercom), small ones are seconds from now.
        return max(0.0, number - now) if number > 1e9 else max(0.0, number)
    return None


class RateLimitTransport(httpx.BaseTransport):
    """Retries rate-limited requests after the delay the platform asks for."""

    def __init__(
        self,
        inner: httpx.BaseTransport | None = None,
        *,
        max_retries: int = 3,
        max_wait: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.inner = inner or httpx.HTTPTransport()
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._sleep = sleep
        self._clock = clock

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        attempt = 0
        while True:
            response = self.inner.handle_request(request)
            if response.status_code not in (429, 503) or attempt >= self.max_retries:
                return response
            wait = retry_after(response.headers, self._clock())
            if wait is None:
                if response.status_code == 503:
                    return response
                wait = float(2**attempt)
            if wait > self.max_wait:
                logger.warning("rate limit: asked to wait %.0fs, more than %.0fs; giving up", wait, self.max_wait)
                return response
            logger.info("rate limit: %s, waiting %.1fs (retry %d)", response.status_code, wait, attempt + 1)
            response.close()
            self._sleep(wait)
            attempt += 1

    def close(self) -> None:
        self.inner.close()


__all__ = ["OAuthBearerAuth", "OAuthSession", "RateLimitTransport", "retry_after"]
