"""NexLink over HTTP: header token, status codes, body limit, client transport. In-process ASGI, offline."""

import pytest

pytest.importorskip("fastapi")

from agent_data import tickets  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from support_engine.nexlink import (  # noqa: E402
    SCOPE_METRICS,
    ClientRegistry,
    ErrorCode,
    HttpTransport,
    NexLinkClient,
    SupportWorkspace,
    build_support_server,
)
from support_engine.nexlink.api import create_nexlink_app  # noqa: E402

CODER = "coder-token-0123456789abcdef"


def app_client(**kwargs) -> TestClient:
    registry = ClientRegistry.build({"nexcoder": (CODER, [SCOPE_METRICS])})
    server = build_support_server(registry, workspace=SupportWorkspace(tickets()))
    return TestClient(create_nexlink_app(server, **kwargs))


def body(action: str = "support.metrics", **extra) -> dict:
    return {"sender": "nexcoder", "target": "nexsupport", "action": action, **extra}


def test_status_codes_follow_the_reply() -> None:
    c = app_client()
    auth = {"Authorization": f"Bearer {CODER}"}
    ok = c.post("/nexlink", json=body(), headers=auth)
    assert ok.status_code == 200 and ok.json()["result"]["tickets"] == 3
    assert c.post("/nexlink", json=body(), headers={"X-API-Key": CODER}).status_code == 200
    assert c.post("/nexlink", json=body()).status_code == 401
    assert c.post("/nexlink", json=body(), headers={"Authorization": "Bearer wrong-0123456789abc"}).status_code == 401
    ingest = c.post("/nexlink", json=body("support.ingest", params={"tickets": []}), headers=auth)
    assert ingest.status_code == 403
    assert c.post("/nexlink", json=body("support.nothing"), headers=auth).status_code == 404
    assert c.post("/nexlink", json=body(params={"x": 1}), headers=auth).status_code == 400
    assert c.post("/nexlink", content=b"{not json", headers=auth).status_code == 400
    assert c.post("/nexlink", json={"sender": "nexcoder"}, headers=auth).status_code == 400


def test_body_limit() -> None:
    c = app_client(max_body_bytes=200)
    big = body(context={"note": "x" * 150, "more": "y" * 150})
    r = c.post("/nexlink", json=big, headers={"Authorization": f"Bearer {CODER}"})
    assert r.status_code == 413 and r.json()["error"]["code"] == ErrorCode.INVALID_ENVELOPE


def test_http_transport_sends_token_in_header_only() -> None:
    c = app_client()
    sent = []
    original = c.post

    def spy(url, **kwargs):
        sent.append(kwargs)
        return original(url, **kwargs)

    c.post = spy  # type: ignore[method-assign]
    client = NexLinkClient("nexcoder", HttpTransport("http://testserver", client=c), tokens={"nexsupport": CODER})
    reply = client.request("nexsupport", "support.metrics", correlation_id="chain-1")
    assert reply.ok and reply.correlation_id == "chain-1"
    assert CODER not in str(sent[0]["json"])
    assert sent[0]["headers"]["Authorization"] == f"Bearer {CODER}"
