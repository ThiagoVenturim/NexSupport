"""OAuth connectors against mocked help desk APIs: bearer auth, refresh, rate limits, pagination,
normalization to the common ticket model, PII masking and no token leaks. No network."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from email.utils import format_datetime

import httpx
import pytest
from integrations_data import (
    ACCESS_1,
    ACCESS_2,
    REFRESH_1,
    REFRESH_2,
    SECRETS,
    Clock,
    TokenServer,
    env_for,
)

from support_engine.integrations import (
    EnvFileTokenStore,
    MemoryTokenStore,
    OAuthError,
    OAuthToken,
    RateLimitTransport,
    build_integration,
    provider_spec,
    retry_after,
    sync_tickets,
)
from support_engine.models import Ticket, TicketStatus
from support_engine.storage import TicketStore

EMAIL = "maria.example@example.com"


class Api:
    """Routes requests by path and records the Authorization header of each call."""

    def __init__(self, routes, *, valid_tokens=(ACCESS_1, ACCESS_2)) -> None:
        self.routes = routes
        self.valid_tokens = set(valid_tokens)
        self.seen: list[tuple[str, str | None]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        self.seen.append((str(request.url), auth))
        self.headers.append(request.headers)
        if auth not in {f"Bearer {t}" for t in self.valid_tokens}:
            return httpx.Response(401, json={"error": "invalid_token"})
        key = request.url.path + (f"?{request.url.query.decode()}" if request.url.query else "")
        for route, answer in self.routes.items():
            if key == route or request.url.path == route:
                return answer(request) if callable(answer) else httpx.Response(200, json=answer)
        return httpx.Response(404)


def integration(provider, api, *, token=None, server=None, clock=None, sleep=None, extra=None):
    store = MemoryTokenStore({provider: token or OAuthToken(ACCESS_1, REFRESH_1, expires_at=None)})
    env = env_for(provider, **(extra or {}))
    built = build_integration(
        provider,
        environ=env,
        env_file=None,
        store=store,
        transport=httpx.MockTransport(api),
        token_http=(server or TokenServer()).client(),
        clock=clock or Clock(),
        sleep=sleep or (lambda _s: None),
    )
    return built, store


ZENDESK_PAGES = {
    "/api/v2/tickets.json?page=2": {
        "tickets": [
            {
                "id": 2,
                "subject": "Second page",
                "description": "Password reset link expired.",
                "status": "solved",
                "created_at": "2026-02-02T08:00:00Z",
                "updated_at": "2026-02-02T09:00:00Z",
            }
        ],
        "meta": {"has_more": False},
    },
    "/api/v2/tickets.json": {
        "tickets": [
            {
                "id": 1,
                "subject": "Export fails",
                "description": f"Export fails for {EMAIL}, phone +1 555 010 2030.",
                "status": "open",
                "priority": "High",
                "requester_id": 11,
                "tags": ["export"],
                "created_at": "2026-02-01T10:00:00Z",
                "comments": [
                    {"id": 5, "author_id": 11, "plain_body": f"Write to {EMAIL}", "created_at": "2026-02-01T10:05:00Z"}
                ],
            }
        ],
        "meta": {"has_more": True},
        "links": {"next": "https://acme.zendesk.com/api/v2/tickets.json?page=2"},
    },
}


def test_zendesk_sync_paginates_normalizes_and_masks(tmp_path) -> None:
    api = Api(ZENDESK_PAGES)
    built, _store = integration("zendesk", api, extra={"SUBDOMAIN": "acme"})
    with built, TicketStore(target=f"sqlite:///{tmp_path / 't.db'}") as tickets_db:
        result = sync_tickets(built.connector, store=tickets_db, max_pages=5)
        stored = tickets_db.list_tickets()
    assert [t.id for t in result.tickets] == ["1", "2"]
    assert all(isinstance(t, Ticket) and t.source == "zendesk" for t in result.tickets)
    first = result.tickets[0]
    assert first.status == TicketStatus.OPEN and first.priority == "high"
    assert EMAIL not in first.description and EMAIL not in first.comments[0].body
    assert result.stats.pii_masked.get("EMAIL", 0) >= 1
    assert {t.id for t in stored} == {"1", "2"}
    assert all(EMAIL not in t.description for t in stored)
    assert all(auth == f"Bearer {ACCESS_1}" for _url, auth in api.seen)
    assert all(url.startswith("https://acme.zendesk.com/") for url, _ in api.seen)


def test_freshdesk_sync() -> None:
    def tickets(request):
        page = request.url.params.get("page")
        rows = (
            [
                {
                    "id": 401,
                    "subject": "Rounding",
                    "description": f"<p>Mail {EMAIL}</p>",
                    "status": 4,
                    "created_at": "2026-02-03T11:00:00Z",
                }
            ]
            if page == "1"
            else []
        )
        return httpx.Response(200, json=rows)

    api = Api({"/api/v2/tickets": tickets})
    extra = {
        "DOMAIN": "acme",
        "AUTHORIZE_URL": "https://acme.myfreshworks.example/oauth/authorize",
        "TOKEN_URL": "https://acme.myfreshworks.example/oauth/token",
        "SCOPES": "freshdesk.tickets.view",
    }
    built, _ = integration("freshdesk", api, extra=extra)
    with built:
        result = sync_tickets(built.connector, per_page=30, max_pages=3)
    assert [t.id for t in result.tickets] == ["401"]
    assert result.tickets[0].status == TicketStatus.SOLVED
    assert EMAIL not in result.tickets[0].description
    assert api.seen[0][0].startswith("https://acme.freshdesk.com/api/v2/tickets")


def test_freshdesk_needs_explicit_oauth_endpoints() -> None:
    with pytest.raises(OAuthError, match="FRESHDESK_AUTHORIZE_URL"):
        provider_spec("freshdesk", environ=env_for("freshdesk", DOMAIN="acme"), env_file=None)


def test_intercom_cursor_pagination_and_version_header() -> None:
    def conversations(request):
        if request.url.params.get("starting_after") == "cursor-2":
            body = {"conversations": [{"id": "ic-2", "title": "Second", "state": "open", "created_at": 1770200000}]}
        else:
            body = {
                "conversations": [{"id": "ic-1", "title": "First", "state": "closed", "created_at": 1770100000}],
                "pages": {"next": {"starting_after": "cursor-2"}},
            }
        return httpx.Response(200, json=body)

    api = Api({"/conversations": conversations})
    built, _ = integration("intercom", api)
    with built:
        result = sync_tickets(built.connector, max_pages=5)
    assert [t.id for t in result.tickets] == ["ic-1", "ic-2"]
    assert result.tickets[0].status == TicketStatus.CLOSED
    assert all(h.get("intercom-version") == "2.11" for h in api.headers)


def test_jira_sm_uses_cloud_api_and_offset_pagination() -> None:
    def search(request):
        start = int(request.url.params.get("startAt"))
        issue = {
            "key": f"SUP-{start + 1}",
            "fields": {
                "summary": f"Issue {start + 1}",
                "description": f"Reporter {EMAIL}",
                "status": {"name": "Waiting for customer"},
                "created": "2026-02-04T09:00:00.000+0000",
            },
        }
        return httpx.Response(200, json={"startAt": start, "total": 2, "issues": [issue]})

    api = Api({"/ex/jira/cloud-1234/rest/api/3/search": search})
    built, _ = integration("jira_sm", api, extra={"CLOUD_ID": "cloud-1234"})
    with built:
        result = sync_tickets(built.connector, max_results=1, max_pages=5)
    assert [t.id for t in result.tickets] == ["SUP-1", "SUP-2"]
    assert result.tickets[0].status == TicketStatus.PENDING
    assert EMAIL not in result.tickets[0].description
    assert all(url.startswith("https://api.atlassian.com/ex/jira/cloud-1234/") for url, _ in api.seen)
    assert built.flow.provider.scopes == ("read:jira-work", "offline_access")
    assert "audience=api.atlassian.com" in built.flow.start().url


def test_minimal_default_scopes() -> None:
    zendesk = provider_spec("zendesk", environ=env_for("zendesk", SUBDOMAIN="acme"), env_file=None)
    assert zendesk.oauth.scopes == ("tickets:read",)
    assert zendesk.oauth.token_url == "https://acme.zendesk.com/oauth/tokens"
    intercom = provider_spec("intercom", environ={}, env_file=None)
    assert intercom.oauth.scopes == ()
    custom = provider_spec("zendesk", environ=env_for("zendesk", SUBDOMAIN="acme", SCOPES="read"), env_file=None)
    assert custom.oauth.scopes == ("read",)


@pytest.mark.parametrize("value", ["evil.example/x", "acme.evil", "a b", "-acme", "acme@evil"])
def test_subdomain_cannot_redirect_the_token(value) -> None:
    with pytest.raises(OAuthError, match="bare subdomain"):
        provider_spec("zendesk", environ=env_for("zendesk", SUBDOMAIN=value), env_file=None)
    if value == "-acme":
        return  # a valid cloud id shape: it only names a path segment on api.atlassian.com
    with pytest.raises(OAuthError):
        provider_spec("jira_sm", environ=env_for("jira_sm", CLOUD_ID=value), env_file=None)


def test_unknown_provider() -> None:
    with pytest.raises(OAuthError, match="Unknown provider"):
        provider_spec("helpscout", environ={}, env_file=None)


def test_missing_client_settings_name_the_variable() -> None:
    with pytest.raises(OAuthError, match="SUPPORT_ENGINE_OAUTH_ZENDESK_CLIENT_SECRET"):
        build_integration(
            "zendesk",
            environ={
                "SUPPORT_ENGINE_OAUTH_ZENDESK_SUBDOMAIN": "acme",
                "SUPPORT_ENGINE_OAUTH_ZENDESK_CLIENT_ID": "x",
            },
            env_file=None,
        )


def test_token_is_never_sent_to_another_host() -> None:
    pages = {
        "/api/v2/tickets.json": {
            "tickets": [],
            "meta": {"has_more": True},
            "links": {"next": "https://collector.example/steal"},
        }
    }
    api = Api(pages)
    built, _ = integration("zendesk", api, extra={"SUBDOMAIN": "acme"})
    with built, pytest.raises(OAuthError, match="outside the configured API"):
        built.connector.fetch_tickets(max_pages=3)
    assert all("collector.example" not in url for url, _ in api.seen)


def test_expired_token_is_refreshed_before_the_call_and_saved() -> None:
    clock = Clock()
    server = TokenServer()
    api = Api({"/conversations": {"conversations": []}}, valid_tokens=(ACCESS_2,))
    old = OAuthToken(ACCESS_1, REFRESH_1, expires_at=clock.now + 10)
    built, store = integration("intercom", api, token=old, server=server, clock=clock)
    with built:
        built.connector.fetch_tickets(max_pages=1)
    assert [r["grant_type"] for r in server.requests] == ["refresh_token"]
    assert api.seen == [(api.seen[0][0], f"Bearer {ACCESS_2}")]
    saved = store.load("intercom")
    assert (saved.access_token, saved.refresh_token) == (ACCESS_2, REFRESH_2)
    assert saved.expires_at == clock.now + 3600


def test_401_triggers_one_refresh_and_retry() -> None:
    server = TokenServer(rotate=False)
    api = Api({"/conversations": {"conversations": []}}, valid_tokens=(ACCESS_2,))
    built, store = integration("intercom", api, server=server)
    with built:
        built.connector.fetch_tickets(max_pages=1)
    assert [auth for _u, auth in api.seen] == [f"Bearer {ACCESS_1}", f"Bearer {ACCESS_2}"]
    assert store.load("intercom").refresh_token == REFRESH_1


def test_revoked_token_fails_after_one_retry() -> None:
    api = Api({"/conversations": {"conversations": []}}, valid_tokens=())
    server = TokenServer()
    built, _ = integration("intercom", api, server=server)
    with built, pytest.raises(httpx.HTTPStatusError):
        built.connector.fetch_tickets(max_pages=1)
    assert len(api.seen) == 2 and len(server.requests) == 1


def test_expired_token_without_refresh_asks_for_login() -> None:
    clock = Clock()
    api = Api({"/conversations": {"conversations": []}})
    built, _ = integration("intercom", api, token=OAuthToken(ACCESS_1, None, expires_at=clock.now - 1), clock=clock)
    with built, pytest.raises(OAuthError, match="log in again"):
        built.connector.fetch_tickets(max_pages=1)
    assert api.seen == []


def test_no_stored_token() -> None:
    built = build_integration(
        "intercom",
        environ=env_for("intercom"),
        env_file=None,
        store=MemoryTokenStore(),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    with built, pytest.raises(OAuthError, match="run the OAuth login"):
        built.connector.fetch_tickets(max_pages=1)


def test_rate_limit_waits_and_retries() -> None:
    calls = {"n": 0}

    def limited(_request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        if calls["n"] == 2:
            return httpx.Response(429, headers={"X-RateLimit-Reset": str(int(clock.now) + 5)})
        return httpx.Response(200, json={"conversations": []})

    clock = Clock()
    waits: list[float] = []
    built, _ = integration("intercom", Api({"/conversations": limited}), clock=clock, sleep=waits.append)
    with built:
        assert built.connector.fetch_tickets(max_pages=1) == []
    assert waits == [2.0, 5.0]


def test_rate_limit_gives_up_after_max_retries_or_long_waits() -> None:
    waits: list[float] = []
    transport = RateLimitTransport(
        httpx.MockTransport(lambda _r: httpx.Response(429)), max_retries=2, sleep=waits.append, clock=Clock()
    )
    with httpx.Client(transport=transport) as http:
        assert http.get("https://api.example/x").status_code == 429
    assert waits == [1.0, 2.0]

    waits.clear()
    transport = RateLimitTransport(
        httpx.MockTransport(lambda _r: httpx.Response(429, headers={"Retry-After": "3600"})),
        sleep=waits.append,
        clock=Clock(),
    )
    with httpx.Client(transport=transport) as http:
        assert http.get("https://api.example/x").status_code == 429
    assert waits == []

    transport = RateLimitTransport(httpx.MockTransport(lambda _r: httpx.Response(503)), sleep=waits.append)
    with httpx.Client(transport=transport) as http:
        assert http.get("https://api.example/x").status_code == 503
    assert waits == []


def test_retry_after_formats() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    later = format_datetime(datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC), usegmt=True)
    assert retry_after(httpx.Headers({"Retry-After": "7"}), now) == 7.0
    assert retry_after(httpx.Headers({"Retry-After": later}), now) == 30.0
    assert retry_after(httpx.Headers({"RateLimit-Reset": "12"}), now) == 12.0
    assert retry_after(httpx.Headers({"X-RateLimit-Reset": str(int(now) + 9)}), now) == 9.0
    assert retry_after(httpx.Headers({"Retry-After": "soon"}), now) is None
    assert retry_after(httpx.Headers({}), now) is None


def test_full_flow_never_logs_or_leaks_tokens(tmp_path, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    env_file = tmp_path / ".env"
    env_file.write_text("DB_URL=sqlite:///x.db\n", encoding="utf-8")
    clock = Clock()
    server = TokenServer()
    store = EnvFileTokenStore(env_file, environ={})
    api = Api(ZENDESK_PAGES, valid_tokens=(ACCESS_2,))
    built = build_integration(
        "zendesk",
        environ=env_for("zendesk", SUBDOMAIN="acme"),
        env_file=None,
        store=store,
        transport=httpx.MockTransport(api),
        token_http=server.client(),
        clock=clock,
    )
    request = built.flow.start()
    store.save("zendesk", built.flow.finish(request.state, "synthetic-authorization-code"))
    with built:
        result = sync_tickets(built.connector, max_pages=3)
    assert len(result.tickets) == 2
    texts = [caplog.text, repr(built), repr(built.session), repr(store), repr(built.flow), repr(store.load("zendesk"))]
    texts += [t.model_dump_json() for t in result.tickets]
    for text in texts:
        for secret in SECRETS:
            assert secret not in text
    # The rotated tokens live only in the .env file.
    content = env_file.read_text(encoding="utf-8")
    assert f"SUPPORT_ENGINE_OAUTH_ZENDESK_ACCESS_TOKEN={ACCESS_2}" in content
    assert f"SUPPORT_ENGINE_OAUTH_ZENDESK_REFRESH_TOKEN={REFRESH_2}" in content
    assert content.startswith("DB_URL=sqlite:///x.db\n")
