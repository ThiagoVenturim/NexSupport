"""OAuth endpoints and minimal scopes of each help desk, and connectors wired to them.

Built in:

- ``zendesk``: ``https://<subdomain>.zendesk.com/oauth/authorizations/new`` and ``/oauth/tokens``,
  scope ``tickets:read``, PKCE. Needs ``SUPPORT_ENGINE_OAUTH_ZENDESK_SUBDOMAIN``.
- ``intercom``: ``https://app.intercom.com/oauth`` and ``https://api.intercom.io/auth/eagle/token``.
  Intercom sets permissions on the app in its Developer Hub, so no scope is sent; give the app
  read access to conversations only.
- ``jira_sm``: Atlassian OAuth 2.0 (3LO) at ``https://auth.atlassian.com``, scopes
  ``read:jira-work`` and ``offline_access`` (refresh token). API calls go to
  ``https://api.atlassian.com/ex/jira/<cloud id>``; set ``SUPPORT_ENGINE_OAUTH_JIRA_SM_CLOUD_ID``.
- ``freshdesk``: the Freshdesk REST API is usually called with an API key. When the account
  issues OAuth tokens through a Freshworks app, set ``..._FRESHDESK_AUTHORIZE_URL``,
  ``..._FRESHDESK_TOKEN_URL`` and ``..._FRESHDESK_SCOPES`` from that app; there are no defaults.
  Needs ``SUPPORT_ENGINE_OAUTH_FRESHDESK_DOMAIN`` (the ``<domain>`` of ``<domain>.freshdesk.com``).

``SUPPORT_ENGINE_OAUTH_<PROVIDER>_SCOPES`` replaces the default scopes when the app needs others.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from support_engine.connectors import FreshdeskConnector, IntercomConnector, JiraSMConnector, ZendeskConnector
from support_engine.connectors.base import TicketConnector
from support_engine.integrations.http import OAuthBearerAuth, OAuthSession, RateLimitTransport
from support_engine.integrations.oauth import OAuthError, OAuthFlow, OAuthProvider, parse_scope
from support_engine.integrations.tokens import EnvFileTokenStore, TokenStore, env_key, load_client, merged_env

PROVIDERS = ("zendesk", "freshdesk", "intercom", "jira_sm")

_SUBDOMAIN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_CLOUD_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def _setting(values: Mapping[str, str], provider: str, name: str) -> str:
    key = env_key(provider, name)
    value = values.get(key, "").strip()
    if not value:
        raise OAuthError(f"Set {key} in the environment or in the .env file.")
    return value


def _subdomain(values: Mapping[str, str], provider: str, name: str) -> str:
    # Validated so a setting cannot point the token at another host.
    value = _setting(values, provider, name).lower()
    if not _SUBDOMAIN.match(value):
        raise OAuthError(f"{env_key(provider, name)} must be a bare subdomain such as 'acme'.")
    return value


def _scopes(values: Mapping[str, str], provider: str, default: tuple[str, ...]) -> tuple[str, ...]:
    override = values.get(env_key(provider, "SCOPES"), "").strip()
    return parse_scope(override) if override else default


@dataclass(frozen=True)
class ProviderSpec:
    """OAuth endpoints plus the API base URL and headers of one configured account."""

    oauth: OAuthProvider
    api_base_url: str
    api_headers: Mapping[str, str]

    @property
    def api_host(self) -> str:
        return urlsplit(self.api_base_url).hostname or ""


def provider_spec(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = ".env",
) -> ProviderSpec:
    """Endpoints of ``name`` for the account described by the environment."""
    values = merged_env(environ, env_file)
    accept = {"Accept": "application/json"}
    if name == "zendesk":
        sub = _subdomain(values, name, "SUBDOMAIN")
        base = f"https://{sub}.zendesk.com"
        oauth = OAuthProvider(
            name=name,
            authorize_url=f"{base}/oauth/authorizations/new",
            token_url=f"{base}/oauth/tokens",
            scopes=_scopes(values, name, ("tickets:read",)),
            body_format="json",
        )
        return ProviderSpec(oauth, base, accept)
    if name == "intercom":
        oauth = OAuthProvider(
            name=name,
            authorize_url="https://app.intercom.com/oauth",
            token_url="https://api.intercom.io/auth/eagle/token",
            scopes=_scopes(values, name, ()),
            body_format="form",
            pkce=False,
        )
        return ProviderSpec(oauth, "https://api.intercom.io", {**accept, "Intercom-Version": "2.11"})
    if name == "jira_sm":
        cloud_id = _setting(values, name, "CLOUD_ID")
        if not _CLOUD_ID.match(cloud_id):
            raise OAuthError(f"{env_key(name, 'CLOUD_ID')} must be the Atlassian cloud id.")
        oauth = OAuthProvider(
            name=name,
            authorize_url="https://auth.atlassian.com/authorize",
            token_url="https://auth.atlassian.com/oauth/token",
            scopes=_scopes(values, name, ("read:jira-work", "offline_access")),
            body_format="json",
            extra_authorize_params={"audience": "api.atlassian.com", "prompt": "consent"},
            pkce=False,
        )
        return ProviderSpec(oauth, f"https://api.atlassian.com/ex/jira/{cloud_id}", accept)
    if name == "freshdesk":
        domain = _subdomain(values, name, "DOMAIN")
        oauth = OAuthProvider(
            name=name,
            authorize_url=_setting(values, name, "AUTHORIZE_URL"),
            token_url=_setting(values, name, "TOKEN_URL"),
            scopes=parse_scope(_setting(values, name, "SCOPES")),
            body_format="form",
        )
        return ProviderSpec(oauth, f"https://{domain}.freshdesk.com", accept)
    raise OAuthError(f"Unknown provider '{name}'. Use one of: {', '.join(PROVIDERS)}.")


@dataclass
class Integration:
    """A connector whose HTTP client carries the OAuth token, refresh and rate limit handling."""

    name: str
    spec: ProviderSpec
    flow: OAuthFlow
    session: OAuthSession
    connector: TicketConnector
    http: httpx.Client

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> Integration:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_flow(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = ".env",
    token_http: httpx.Client | None = None,
    clock: Callable[[], float] = time.time,
) -> OAuthFlow:
    """OAuth flow of ``name`` from the environment (client id, secret, redirect URI, endpoints)."""
    spec = provider_spec(name, environ=environ, env_file=env_file)
    client = load_client(name, environ=environ, env_file=env_file)
    return OAuthFlow(spec.oauth, client, http=token_http, clock=clock)


def build_integration(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = ".env",
    store: TokenStore | None = None,
    transport: httpx.BaseTransport | None = None,
    token_http: httpx.Client | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    max_retries: int = 3,
    timeout: float = 30.0,
) -> Integration:
    """Connector for ``name`` authenticated with the stored OAuth token.

    ``transport`` and ``token_http`` let tests answer the API and the token endpoint offline.
    """
    spec = provider_spec(name, environ=environ, env_file=env_file)
    client = load_client(name, environ=environ, env_file=env_file)
    flow = OAuthFlow(spec.oauth, client, http=token_http, clock=clock)
    if store is None:
        store = EnvFileTokenStore(env_file or ".env", environ=environ)
    session = OAuthSession(name, store, flow, clock=clock)
    http = httpx.Client(
        base_url=spec.api_base_url,
        headers=dict(spec.api_headers),
        auth=OAuthBearerAuth(session, [spec.api_host]),
        transport=RateLimitTransport(transport, max_retries=max_retries, sleep=sleep, clock=clock),
        timeout=timeout,
    )
    connector: TicketConnector
    if name == "zendesk":
        connector = ZendeskConnector(base_url=spec.api_base_url, client=http)
    elif name == "freshdesk":
        connector = FreshdeskConnector(base_url=spec.api_base_url, client=http)
    elif name == "intercom":
        connector = IntercomConnector(base_url=spec.api_base_url, client=http)
    else:
        connector = JiraSMConnector(base_url=spec.api_base_url, client=http)
    return Integration(name, spec, flow, session, connector, http)


__all__ = [
    "PROVIDERS",
    "Integration",
    "ProviderSpec",
    "build_flow",
    "build_integration",
    "provider_spec",
]
