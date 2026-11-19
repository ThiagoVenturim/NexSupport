"""OAuth authorization code flow, PKCE, state checks, refresh and scope rules, all offline."""

from __future__ import annotations

import base64
import hashlib
import logging
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from integrations_data import (
    ACCESS_1,
    ACCESS_2,
    CLIENT_ID,
    CLIENT_SECRET,
    CODE,
    REDIRECT,
    REFRESH_1,
    REFRESH_2,
    SECRETS,
    Clock,
    TokenServer,
)

from support_engine.integrations import oauth as oauth_module
from support_engine.integrations.oauth import (
    OAuthClient,
    OAuthError,
    OAuthFlow,
    OAuthProvider,
    OAuthToken,
    check_redirect_uri,
    parse_scope,
)

PROVIDER = OAuthProvider(
    name="zendesk",
    authorize_url="https://synthetic.zendesk.com/oauth/authorizations/new",
    token_url="https://synthetic.zendesk.com/oauth/tokens",
    scopes=("tickets:read",),
    body_format="json",
)


def make_flow(server: TokenServer | None = None, clock: Clock | None = None, provider=PROVIDER) -> OAuthFlow:
    server = server or TokenServer()
    return OAuthFlow(
        provider,
        OAuthClient(CLIENT_ID, CLIENT_SECRET, REDIRECT),
        http=server.client(),
        clock=clock or Clock(),
    )


def query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def test_authorize_url_has_state_pkce_and_minimal_scope() -> None:
    request = make_flow().start()
    params = query(request.url)
    assert request.url.startswith(PROVIDER.authorize_url + "?")
    assert params["response_type"] == "code"
    assert params["client_id"] == CLIENT_ID
    assert params["redirect_uri"] == REDIRECT
    assert params["scope"] == "tickets:read"
    assert params["state"] == request.state and len(request.state) >= 32
    assert params["code_challenge_method"] == "S256"
    digest = hashlib.sha256(request.code_verifier.encode("ascii")).digest()
    assert params["code_challenge"] == base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert CLIENT_SECRET not in request.url
    assert request.code_verifier not in request.url
    assert request.code_verifier not in repr(request)


def test_code_exchange_sends_verifier_and_returns_token() -> None:
    server = TokenServer()
    clock = Clock()
    flow = make_flow(server, clock)
    request = flow.start()
    token = flow.finish(request.state, CODE)
    assert token.access_token == ACCESS_1
    assert token.refresh_token == REFRESH_1
    assert token.scopes == ("tickets:read",)
    assert token.expires_at == clock.now + 3600
    sent = server.requests[-1]
    assert sent["grant_type"] == "authorization_code"
    assert sent["code_verifier"] == request.code_verifier
    assert sent["redirect_uri"] == REDIRECT


def test_finish_from_redirect_url() -> None:
    flow = make_flow()
    request = flow.start()
    token = flow.finish_from_redirect(f"{REDIRECT}?code={CODE}&state={request.state}")
    assert token.access_token == ACCESS_1
    with pytest.raises(OAuthError, match="does not match"):
        flow.finish_from_redirect(f"https://attacker.example/callback?code={CODE}&state={flow.start().state}")


def test_state_is_single_use_and_checked() -> None:
    flow = make_flow()
    request = flow.start()
    with pytest.raises(OAuthError, match="Unknown"):
        flow.finish("not-the-state", CODE)
    with pytest.raises(OAuthError, match="Unknown"):
        flow.finish(None, CODE)
    flow.finish(request.state, CODE)
    with pytest.raises(OAuthError, match="reused"):
        flow.finish(request.state, CODE)


def test_state_expires() -> None:
    clock = Clock()
    flow = make_flow(clock=clock)
    request = flow.start()
    clock.now += 601
    with pytest.raises(OAuthError, match="expired"):
        flow.finish(request.state, CODE)


def test_state_is_compared_in_constant_time(monkeypatch) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = oauth_module.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(oauth_module.hmac, "compare_digest", spy)
    flow = make_flow()
    first, second = flow.start(), flow.start()
    flow.finish(second.state, CODE)
    # Every pending state is compared, not only the matching one.
    assert {b for _a, b in calls} >= {first.state.encode(), second.state.encode()}


def test_denied_authorization() -> None:
    flow = make_flow()
    request = flow.start()
    with pytest.raises(OAuthError, match="denied"):
        flow.finish(request.state, None, error="access_denied")


def test_refresh_rotates_and_keeps_refresh_token() -> None:
    flow = make_flow(TokenServer(rotate=True))
    old = OAuthToken(ACCESS_1, REFRESH_1, expires_at=0, scopes=("tickets:read",))
    fresh = flow.refresh(old)
    assert (fresh.access_token, fresh.refresh_token) == (ACCESS_2, REFRESH_2)

    flow = make_flow(TokenServer(rotate=False))
    fresh = flow.refresh(old)
    assert (fresh.access_token, fresh.refresh_token) == (ACCESS_2, REFRESH_1)

    with pytest.raises(OAuthError, match="cannot be refreshed"):
        flow.refresh(OAuthToken(ACCESS_1))


def test_scopes_beyond_the_request_are_rejected() -> None:
    flow = make_flow(TokenServer(scope="tickets:read tickets:write users:write"))
    with pytest.raises(OAuthError, match="tickets:write, users:write"):
        flow.finish(flow.start().state, CODE)


def test_missing_scope_in_answer_uses_the_requested_scopes() -> None:
    flow = make_flow(TokenServer(scope=None))
    assert flow.finish(flow.start().state, CODE).scopes == ("tickets:read",)


def test_token_endpoint_errors_do_not_leak(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    flow = OAuthFlow(
        PROVIDER,
        OAuthClient(CLIENT_ID, "wrong-secret-0123456789", REDIRECT),
        http=TokenServer().client(),
        clock=Clock(),
    )
    with pytest.raises(OAuthError) as info:
        flow.finish(flow.start().state, CODE)
    # The fake server echoes the secret in error_description; it must not reach the message.
    assert "401" in str(info.value) and "invalid_client" in str(info.value)
    assert "wrong-secret" not in str(info.value)

    flow = make_flow()
    with pytest.raises(OAuthError) as info:
        flow.finish(flow.start().state, "bad-code-with-secret-material")
    assert "bad-code-with-secret-material" not in str(info.value)
    assert "invalid_grant" in str(info.value)

    def broken(_request):
        raise httpx.ConnectError("connection refused")

    flow = OAuthFlow(
        PROVIDER,
        OAuthClient(CLIENT_ID, CLIENT_SECRET, REDIRECT),
        http=httpx.Client(transport=httpx.MockTransport(broken)),
    )
    with pytest.raises(OAuthError, match="ConnectError"):
        flow.finish(flow.start().state, CODE)
    for secret in SECRETS:
        assert secret not in caplog.text


def test_reprs_hide_secrets() -> None:
    token = OAuthToken(ACCESS_1, REFRESH_1, scopes=("tickets:read",))
    client = OAuthClient(CLIENT_ID, CLIENT_SECRET, REDIRECT)
    for text in (repr(token), str(token), repr(client), repr(make_flow())):
        for secret in SECRETS:
            assert secret not in text


def test_https_rules() -> None:
    with pytest.raises(OAuthError):
        OAuthProvider("x", "http://synthetic.example/authorize", "https://synthetic.example/token")
    with pytest.raises(OAuthError):
        OAuthProvider("x", "https://synthetic.example/authorize", "http://synthetic.example/token")
    assert check_redirect_uri("https://app.example/callback")
    assert check_redirect_uri("http://localhost:9000/cb")
    for bad in ("http://app.example/callback", "ftp://localhost/cb", "https://app.example/cb#frag"):
        with pytest.raises(OAuthError):
            check_redirect_uri(bad)
    with pytest.raises(OAuthError):
        OAuthClient(CLIENT_ID, "", REDIRECT)


def test_bad_token_answers() -> None:
    def answer(payload, status=200):
        flow = OAuthFlow(
            PROVIDER,
            OAuthClient(CLIENT_ID, CLIENT_SECRET, REDIRECT),
            http=httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(status, json=payload))),
        )
        return flow.finish(flow.start().state, CODE)

    with pytest.raises(OAuthError, match="no access token"):
        answer({"token_type": "bearer"})
    with pytest.raises(OAuthError, match="token type"):
        answer({"access_token": ACCESS_1, "token_type": "mac"})
    with pytest.raises(OAuthError, match="expires_in"):
        answer({"access_token": ACCESS_1, "expires_in": "soon"})
    # Intercom answers with "token" and no expiry.
    token = answer({"token": ACCESS_1, "access_token": ACCESS_1, "token_type": "Bearer"})
    assert token.expires_at is None and not token.can_refresh


def test_form_body_and_no_pkce() -> None:
    provider = OAuthProvider(
        "intercom", "https://synthetic.example/oauth", "https://synthetic.example/token", body_format="form", pkce=False
    )
    server = TokenServer(scope=None)
    flow = make_flow(server, provider=provider)
    request = flow.start()
    assert "code_challenge" not in query(request.url) and "scope" not in query(request.url)
    flow.finish(request.state, CODE)
    assert "code_verifier" not in server.requests[-1]


def test_parse_scope() -> None:
    assert parse_scope("a b,c  a") == ("a", "b", "c")
    assert parse_scope(None) == ()
    assert parse_scope(["x", " y ", ""]) == ("x", "y")
