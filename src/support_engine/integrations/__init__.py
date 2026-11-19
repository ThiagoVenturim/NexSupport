"""Help desk integrations over OAuth 2.0 and MCP.

Typical use::

    from support_engine.integrations import build_flow, build_integration, sync_tickets, EnvFileTokenStore

    flow = build_flow("zendesk")                       # client id, secret, redirect URI from env/.env
    request = flow.start()                             # open request.url in a browser
    token = flow.finish_from_redirect(redirected_url)  # state checked in constant time, PKCE
    EnvFileTokenStore(".env").save("zendesk", token)   # token only in the git-ignored .env

    with build_integration("zendesk") as integration:  # bearer auth, refresh, rate limits
        result = sync_tickets(integration.connector, store=ticket_store, max_pages=5)

MCP: ``IntegrationsMCPServer(IntegrationsService(ticket_store, copilot_service)).run_stdio()``.
"""

from support_engine.integrations.http import OAuthBearerAuth, OAuthSession, RateLimitTransport, retry_after
from support_engine.integrations.mcp import (
    IntegrationsMCPServer,
    IntegrationsService,
    ListTicketsRequest,
    MCPError,
    MCPToolClient,
    MetricsRequest,
    SuggestByIdRequest,
)
from support_engine.integrations.oauth import (
    AuthorizationRequest,
    OAuthClient,
    OAuthError,
    OAuthFlow,
    OAuthProvider,
    OAuthToken,
    check_redirect_uri,
    constant_time_equals,
    parse_scope,
    pkce_pair,
)
from support_engine.integrations.providers import (
    PROVIDERS,
    Integration,
    ProviderSpec,
    build_flow,
    build_integration,
    provider_spec,
)
from support_engine.integrations.sync import sync_tickets
from support_engine.integrations.tokens import EnvFileTokenStore, MemoryTokenStore, TokenStore, env_key, load_client

__all__ = [
    "PROVIDERS",
    "AuthorizationRequest",
    "EnvFileTokenStore",
    "Integration",
    "IntegrationsMCPServer",
    "IntegrationsService",
    "ListTicketsRequest",
    "MCPError",
    "MCPToolClient",
    "MemoryTokenStore",
    "MetricsRequest",
    "OAuthBearerAuth",
    "OAuthClient",
    "OAuthError",
    "OAuthFlow",
    "OAuthProvider",
    "OAuthSession",
    "OAuthToken",
    "ProviderSpec",
    "RateLimitTransport",
    "SuggestByIdRequest",
    "TokenStore",
    "build_flow",
    "build_integration",
    "check_redirect_uri",
    "constant_time_equals",
    "env_key",
    "load_client",
    "parse_scope",
    "pkce_pair",
    "provider_spec",
    "retry_after",
    "sync_tickets",
]
