# Help desk integrations over OAuth and MCP

`support_engine.integrations` connects the Support Engine to Zendesk, Freshdesk, Intercom and
Jira Service Management with OAuth 2.0, syncs their tickets into the common ticket model with
PII masked, and exposes the main operations as MCP tools.

## OAuth flow

1. Register an OAuth app on the platform with read-only access to tickets and a redirect URI
   (https, or `http://127.0.0.1:<port>/callback` for a CLI login).
2. Put the app settings in the environment or in `.env` (see `.env.example`):

   ```
   SUPPORT_ENGINE_OAUTH_ZENDESK_SUBDOMAIN=acme
   SUPPORT_ENGINE_OAUTH_ZENDESK_CLIENT_ID=...
   SUPPORT_ENGINE_OAUTH_ZENDESK_CLIENT_SECRET=...
   SUPPORT_ENGINE_OAUTH_ZENDESK_REDIRECT_URI=http://127.0.0.1:8765/callback
   ```

3. `support-engine oauth-login zendesk` prints the authorization URL. Approve the access in the
   browser and paste back the URL you are redirected to. The command checks the `state`,
   exchanges the code (with PKCE where the platform supports it) and writes
   `..._ACCESS_TOKEN`, `..._REFRESH_TOKEN`, `..._EXPIRES_AT` and `..._GRANTED_SCOPES` into `.env`.
   The token is never printed.
4. `support-engine sync zendesk --max-pages 10` fetches the tickets, follows the pagination of the
   platform, cleans and masks them and stores them in the database (`DB_URL`).

In code:

```python
from support_engine.integrations import EnvFileTokenStore, build_flow, build_integration, sync_tickets

flow = build_flow("zendesk")
request = flow.start()                              # send the user to request.url
token = flow.finish_from_redirect(redirected_url)   # or flow.finish(state, code)
EnvFileTokenStore(".env").save("zendesk", token)

with build_integration("zendesk") as integration:
    result = sync_tickets(integration.connector, store=ticket_store, max_pages=10)
```

## Platforms

| Provider | Authorize / token endpoints | Default scopes | Refresh |
|---|---|---|---|
| `zendesk` | `https://<subdomain>.zendesk.com/oauth/authorizations/new`, `/oauth/tokens` | `tickets:read` | when the token response has a refresh token |
| `intercom` | `https://app.intercom.com/oauth`, `https://api.intercom.io/auth/eagle/token` | none sent: permissions are set on the app | tokens do not expire |
| `jira_sm` | `https://auth.atlassian.com/authorize`, `https://auth.atlassian.com/oauth/token` | `read:jira-work offline_access` | yes, rotating refresh tokens |
| `freshdesk` | from `..._FRESHDESK_AUTHORIZE_URL` and `..._FRESHDESK_TOKEN_URL` | from `..._FRESHDESK_SCOPES` | when the token response has a refresh token |

- Jira Service Management calls go to `https://api.atlassian.com/ex/jira/<cloud id>`; set
  `SUPPORT_ENGINE_OAUTH_JIRA_SM_CLOUD_ID`.
- The Freshdesk REST API is usually called with an API key (`FreshdeskConnector(api_key=...)`).
  Use OAuth only when the account issues tokens through a Freshworks app; there are no default
  endpoints for it.
- `SUPPORT_ENGINE_OAUTH_<PROVIDER>_SCOPES` replaces the default scopes.

## Security rules

- Client secrets and tokens live only in the environment, in the git-ignored `.env` file
  (written atomically, mode 0600 where the OS supports it) or in a vault: implement the
  `TokenStore` protocol (`load`, `save`) for Vault, a cloud secret manager or the OS keyring.
  The environment wins over the file. `EnvFileTokenStore` refuses any file that is not a
  `.env` file and any value that could break the file.
- No token, code, verifier or secret appears in `repr`, logs, CLI output or error messages.
  Token endpoint errors report the HTTP status and the OAuth `error` code only.
- `state` is random, single use, expires after 10 minutes and is compared in constant time
  (`hmac.compare_digest`) against every pending state.
- PKCE S256 protects the authorization code on Zendesk and on custom Freshdesk endpoints.
- Token endpoints must be https; redirect URIs must be https or loopback http, and the pasted
  redirect must match the configured one.
- The flow asks for the minimal scopes and rejects a grant that holds scopes it did not request.
- The bearer token is only sent over https to the API host of the configured account. A
  pagination link to another host fails instead of leaking the token. Subdomains and the cloud
  id are validated so a setting cannot point the token at another host.
- An access token is refreshed 60 seconds before it expires, and once more when the API answers
  401. A rotated refresh token is saved; a missing one keeps the previous refresh token. An
  expired token without a refresh token asks for a new login.

## Rate limits and pagination

`RateLimitTransport` retries 429 answers (and 503 with `Retry-After`) up to 3 times, waiting as
long as `Retry-After` (seconds or HTTP date), `X-RateLimit-Reset` (Intercom, UNIX time) or
`RateLimit-Reset` asks, at most 60 seconds per wait; without a header it backs off 1, 2, 4
seconds. Pagination follows each platform: Zendesk `links.next` / `next_page`, Freshdesk
`page`, Intercom `starting_after` cursors and Jira `startAt` offsets, bounded by `max_pages`.

## MCP tools

`IntegrationsMCPServer` serves, over the synced tickets:

| Tool | Arguments | Result |
|---|---|---|
| `list_tickets` | `status`, `source`, `limit` (1..100), `offset` | tickets in the common model (id, source, subject, status, priority, tags, dates, comment count), PII masked |
| `suggest_reply` | `ticket_id`, `namespace` | the copilot suggestion for the stored ticket |
| `support_metrics` | `max_tickets`, `top` | sentiment, trends, alerts, risk radar, quality and root causes |

- stdio: `support-engine integrations-mcp [--input history.json]` (`--input` enables `suggest_reply`).
- HTTP: `support-engine serve --input history.json --integrations` adds `POST /mcp/integrations`
  behind the same access token as the copilot API.

`MCPToolClient` consumes an MCP server over HTTP with a bearer token read from the environment,
for example the NexHub tools:

```python
from support_engine.integrations import MCPToolClient

client = MCPToolClient.from_env("NEXHUB_MCP_URL", "NEXHUB_MCP_TOKEN")
tools = client.list_tools()
data = client.call_tool("list_tickets", {"source": "zendesk", "limit": 10})
```

## Tests

`tests/test_integrations_*.py` run offline: the token endpoint and each platform API are
`httpx.MockTransport` fakes. They cover the code exchange, PKCE, state reuse and expiry, the
constant-time comparison, refresh and rotation, scope checks, host pinning, rate limit waits,
pagination, normalization to the common ticket model, PII masking and that no secret reaches
logs, reprs, errors or CLI output.
