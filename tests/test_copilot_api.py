"""REST API: token, endpoints, validation without echo, body limit and MCP over HTTP. Offline."""

import pytest

pytest.importorskip("fastapi")

from copilot_data import EMAIL, GOOD, TOKEN, make_copilot, open_ticket  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from support_engine.copilot import InterfaceConfig, TokenError  # noqa: E402
from support_engine.copilot.api import create_app  # noqa: E402

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def client(**kwargs) -> TestClient:
    config = kwargs.pop("config", None)
    copilot, _, _ = make_copilot(**kwargs)
    return TestClient(create_app(copilot, token=TOKEN, config=config))


def test_health_is_public() -> None:
    assert client().get("/health").json() == {"status": "ok"}


def test_requests_without_valid_token_are_refused() -> None:
    c = client()
    body = {"question": "The invoice export fails with error 500"}
    for headers in ({}, {"Authorization": "Bearer wrong-token-0123456789"}, {"X-API-Key": "nope"}):
        r = c.post("/v1/answer", json=body, headers=headers)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert c.get("/openapi.json").status_code == 401
    assert c.get("/docs").status_code == 401


def test_suggest_endpoint() -> None:
    c = client()
    r = c.post("/v1/suggest", json={"ticket": open_ticket().model_dump(mode="json")}, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == GOOD and data["citations"][0]["ticket_id"] == "T-1"
    assert data["summary"]["status"] == "open" and data["tone"]["tone"] == "empathetic"
    assert data["next_step"]["action"] == "review_reply"
    assert EMAIL not in r.text


def test_answer_summary_feedback_endpoints() -> None:
    c = client()
    headers = {"X-API-Key": TOKEN}
    r = c.post("/v1/answer", json={"question": "The invoice export fails with error 500"}, headers=headers)
    assert r.status_code == 200 and r.json()["level"] == 1
    r = c.post("/v1/summary", json={"ticket": open_ticket().model_dump(mode="json")}, headers=headers)
    assert r.status_code == 200 and "[EMAIL_1]" in r.json()["text"]
    r = c.post(
        "/v1/feedback",
        json={"ticket_ids": ["T-1"], "accepted": True, "correction": f"Mail {EMAIL} and clear the cache."},
        headers=headers,
    )
    assert r.status_code == 200 and r.json()["rating"] == 0.5 and len(r.json()["feedback"]) == 1


def test_validation_errors_do_not_echo_input() -> None:
    c = client()
    ticket = open_ticket().model_dump(mode="json")
    ticket.pop("created_at")
    ticket["subject"] = f"Contact {EMAIL}"
    r = c.post("/v1/suggest", json={"ticket": ticket, "namespace": "bad space"}, headers=AUTH)
    assert r.status_code == 422
    assert EMAIL not in r.text and "input" not in r.text
    locs = [e["loc"] for e in r.json()["detail"]]
    assert ["body", "ticket", "created_at"] in locs and ["body", "namespace"] in locs


def test_body_limit() -> None:
    c = client(config=InterfaceConfig(max_body_bytes=1000))
    r = c.post("/v1/answer", json={"question": "x" * 2000}, headers=AUTH)
    assert r.status_code == 413


def test_mcp_over_http_needs_the_token() -> None:
    c = client()
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert c.post("/mcp", json=msg).status_code == 401
    r = c.post("/mcp", json=msg, headers=AUTH)
    assert r.status_code == 200 and len(r.json()["result"]["tools"]) == 4
    note = c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=AUTH)
    assert note.status_code == 202
    bad = c.post("/mcp", content=b"{oops", headers={**AUTH, "Content-Type": "application/json"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == -32700


def test_app_reads_token_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SUPPORT_ENGINE_API_TOKEN", TOKEN)
    copilot, _, _ = make_copilot()
    c = TestClient(create_app(copilot, config=InterfaceConfig(env_file=None)))
    assert c.post("/v1/answer", json={"question": "hello there"}, headers=AUTH).status_code == 200


def test_app_refuses_to_start_without_token(monkeypatch) -> None:
    monkeypatch.delenv("SUPPORT_ENGINE_API_TOKEN", raising=False)
    copilot, _, _ = make_copilot()
    with pytest.raises(TokenError):
        create_app(copilot, config=InterfaceConfig(env_file=None))


def test_internal_errors_are_generic() -> None:
    copilot, _, _ = make_copilot()

    async def boom(*_a, **_k):
        raise RuntimeError(EMAIL)

    copilot.aanswer = boom
    c = TestClient(create_app(copilot, token=TOKEN), raise_server_exceptions=False)
    r = c.post("/v1/answer", json={"question": "hello there"}, headers=AUTH)
    assert r.status_code == 500 and EMAIL not in r.text
