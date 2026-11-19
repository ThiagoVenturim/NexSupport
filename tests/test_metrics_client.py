"""NexDash client over REST and MCP against an in-process fake (httpx.MockTransport): no network."""

import json
import logging

import httpx
import pytest

from support_engine.metrics import NexDashClient, NexDashError, report_request

TOKEN = "client-token-for-tests-0123456789"
REPORT = {
    "model": "support_insights",
    "errors": {},
    "data": {"panels": {"high_risk": {"kind": "kpi", "value": 3}}},
    "dashboard": {"html": "<!doctype html><html></html>"},
}


def fake_nexdash(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"error": "Unauthorized"})
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        if path == "/api/reports/models":
            return httpx.Response(200, json={"models": [{"name": "support_metrics"}, {"name": "support_insights"}]})
        if path == "/api/reports/models/support_insights":
            return httpx.Response(200, json={"name": "support_insights", "panels": []})
        if path == "/api/reports/run":
            if body["model"] != "support_insights":
                return httpx.Response(404, json={"error": "Report model not found."})
            return httpx.Response(200, json=REPORT)
        if path == "/api/reports/dashboard":
            return httpx.Response(200, text="<!doctype html><html>dash</html>", headers={"content-type": "text/html"})
        if path == "/mcp":
            if "id" not in body:
                return httpx.Response(202)
            if body["method"] == "initialize":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"capabilities": {}}})
            name = body["params"]["name"]
            if name == "nexdash_list_report_models":
                payload, error = {"models": [{"name": "support_insights"}]}, False
            elif name == "nexdash_run_report" and body["params"]["arguments"]["model"] == "support_insights":
                payload, error = REPORT, False
            else:
                payload, error = {"error": "Report model not found."}, True
            result = {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": error}
            if not error:
                result["structuredContent"] = payload
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        return httpx.Response(404, json={"error": "Not found"})

    return handler


def client(calls, transport="rest", token=TOKEN):
    http = httpx.Client(transport=httpx.MockTransport(fake_nexdash(calls)))
    return NexDashClient("http://nexdash.test", token, transport=transport, http=http)


def test_request_is_validated_before_it_leaves() -> None:
    request = report_request("support_metrics", start="2026-04-01", end="2026-06-30", team=" tier1 ")
    assert request == {
        "model": "support_metrics",
        "parameters": {"start": "2026-04-01", "end": "2026-06-30", "team": "tier1", "project": None},
        "output": "both",
        "locale": "en",
        "compare": True,
    }
    for bad in (
        {"model": "Support; DROP"},
        {"model": "support_metrics", "start": "2026-06-30", "end": "2026-04-01"},
        {"model": "support_metrics", "start": "yesterday"},
        {"model": "support_metrics", "team": "x" * 201},
        {"model": "support_metrics", "output": "sql"},
        {"model": "support_metrics", "locale": "fr"},
    ):
        model = bad.pop("model")
        with pytest.raises(ValueError):
            report_request(model, **bad)


def test_rest_calls_send_the_token_and_parse_answers(caplog) -> None:
    calls = []
    with caplog.at_level(logging.DEBUG), client(calls) as nexdash:
        assert [m["name"] for m in nexdash.list_models()] == ["support_metrics", "support_insights"]
        assert nexdash.describe_model("support_insights")["name"] == "support_insights"
        report = nexdash.run_report("support_insights", team="tier1")
        assert report["data"]["panels"]["high_risk"]["value"] == 3
        assert "dash" in nexdash.dashboard("support_insights")
        with pytest.raises(NexDashError) as exc:
            nexdash.run_report("no_such_model")
    assert exc.value.status == 404 and "not found" in str(exc.value)
    sent = json.loads(calls[2].content)
    assert sent["parameters"]["team"] == "tier1" and "sql" not in json.dumps(sent).lower()
    assert all(c.headers["authorization"] == f"Bearer {TOKEN}" for c in calls)
    assert TOKEN not in caplog.text and TOKEN not in repr(nexdash)


def test_mcp_initializes_once_and_reads_tool_results() -> None:
    calls = []
    with client(calls, transport="mcp") as nexdash:
        assert nexdash.list_models() == [{"name": "support_insights"}]
        assert nexdash.run_report("support_insights")["model"] == "support_insights"
        assert nexdash.dashboard("support_insights").startswith("<!doctype html>")
        with pytest.raises(NexDashError, match="not found"):
            nexdash.run_report("no_such_model")
    methods = [json.loads(c.content)["method"] for c in calls]
    assert methods[:3] == ["initialize", "notifications/initialized", "tools/call"]
    assert methods.count("initialize") == 1


def test_missing_token_and_unreachable_server() -> None:
    calls = []
    with client(calls, token=None) as nexdash, pytest.raises(NexDashError) as exc:
        nexdash.list_models()
    assert exc.value.status == 401 and "authorization" not in calls[0].headers

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(transport=httpx.MockTransport(refuse))
    with pytest.raises(NexDashError, match="ConnectError"):
        NexDashClient("http://nexdash.test", TOKEN, http=http).list_models()


def test_url_and_environment(monkeypatch) -> None:
    for bad in ("ftp://host", "http://", "http://user:pw@host", "http://host/?token=x"):
        with pytest.raises(ValueError):
            NexDashClient(bad)
    with pytest.raises(ValueError):
        NexDashClient(transport="grpc")
    monkeypatch.setenv("NEXDASH_URL", "https://nexdash.test/")
    monkeypatch.setenv("NEXDASH_ENGINE_TOKEN", TOKEN)
    nexdash = NexDashClient.from_env(transport="mcp")
    assert nexdash.url == "https://nexdash.test" and nexdash.transport == "mcp"
    assert "token=set" in repr(nexdash) and TOKEN not in repr(nexdash)
    nexdash.close()
