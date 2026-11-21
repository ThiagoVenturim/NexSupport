"""nexdash client: request shape, bearer token from env, error mapping, real engine when installed. Offline."""

import json

import httpx
import pytest
from nexlink_data import FakeReportEngine, report_answer
from pydantic import ValidationError

from support_engine.copilot import TokenError
from support_engine.nexlink import EngineReportTransport, NexDashClient, ReportError, RestReportTransport

TOKEN = "nexdash-token-0123456789abcdef"
URL = "http://nexdash.test"


def mock_client(calls: list, status: int = 200, body=None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if status >= 400:
            return httpx.Response(status, json=body or {"error": "parameters: Value error"})
        if request.url.path == "/api/reports/models":
            return httpx.Response(200, json={"models": [{"name": "support_metrics"}]})
        return httpx.Response(200, json=report_answer(json.loads(request.content)))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_rest_transport_sends_bearer_and_validated_request() -> None:
    calls: list[httpx.Request] = []
    client = NexDashClient(RestReportTransport(URL, TOKEN, client=mock_client(calls)))
    assert client.list_models() == ["support_metrics"]
    answer = client.support_dashboard(start="2026-04-01", end="2026-06-30", team="tier1")
    assert answer["model"] == "support_metrics"
    run = calls[-1]
    assert run.method == "POST" and run.url.path == "/api/reports/run"
    assert run.headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(run.content) == {
        "model": "support_metrics",
        "parameters": {"start": "2026-04-01", "end": "2026-06-30", "team": "tier1", "project": None},
        "output": "both",
        "locale": "en",
        "compare": True,
    }


def test_token_comes_from_the_environment_and_stays_hidden(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"NEXDASH_ENGINE_TOKEN={TOKEN}\n", encoding="utf-8")
    transport = RestReportTransport(URL, client=mock_client([]), environ={}, env_file=env_file)
    assert TOKEN not in repr(transport)
    with pytest.raises(TokenError) as missing:
        RestReportTransport(URL, environ={}, env_file=None)
    assert "NEXDASH_ENGINE_TOKEN" in str(missing.value)
    with pytest.raises(ValueError):
        RestReportTransport("file:///etc", TOKEN)


def test_requests_are_validated_before_sending() -> None:
    calls: list[httpx.Request] = []
    client = NexDashClient(RestReportTransport(URL, TOKEN, client=mock_client(calls)))
    with pytest.raises(ValidationError):
        client.support_dashboard(start="2026-06-30", end="2026-01-01")
    with pytest.raises(ValidationError):
        client.run_report({"model": "support_metrics", "parameters": {"sql": "SELECT 1"}})
    with pytest.raises(ValidationError):
        client.run_report(model="DROP TABLE")
    assert calls == []


def test_errors_map_to_report_error() -> None:
    client = NexDashClient(RestReportTransport(URL, TOKEN, client=mock_client([], status=403)))
    with pytest.raises(ReportError) as refused:
        client.support_dashboard()
    assert refused.value.status == 403 and TOKEN not in str(refused.value)

    def unreachable(_request):
        raise httpx.ConnectError("refused")

    offline = httpx.Client(transport=httpx.MockTransport(unreachable))
    down = NexDashClient(RestReportTransport(URL, TOKEN, client=offline))
    with pytest.raises(ReportError) as err:
        down.support_dashboard()
    assert err.value.status == 503

    failing = NexDashClient(EngineReportTransport(FakeReportEngine(fail_with=404)))
    with pytest.raises(ReportError) as missing:
        failing.run_report(model="unknown_model")
    assert missing.value.status == 404


def test_real_nexdash_engine_in_process() -> None:
    pytest.importorskip("nexdash.engine")
    client = NexDashClient(EngineReportTransport.demo())
    assert "support_metrics" in client.list_models()
    answer = client.support_dashboard(start="2026-04-01", end="2026-06-30", locale="pt-BR")
    assert answer["errors"] == {}
    assert {"volume", "csat", "resolution_rate"} <= set(answer["data"]["panels"])
    html = answer["dashboard"]["html"]
    assert html.lower().startswith("<!doctype html>") and "<script" not in html.lower()
    with pytest.raises(ReportError) as unknown:
        client.run_report(model="no_such_model")
    assert unknown.value.status == 404
