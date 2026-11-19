"""Support Engine NexLink server: auth, scopes, actions, no echo and no PII in replies or logs. Offline."""

import json
import logging

from agent_data import tickets
from copilot_data import EMAIL, GOOD, make_copilot, open_ticket
from nexlink_data import FakeReportEngine

from support_engine.copilot import CopilotService
from support_engine.nexlink import (
    SCOPE_INGEST,
    SCOPE_METRICS,
    SCOPE_SUGGEST,
    ClientRegistry,
    EngineReportTransport,
    ErrorCode,
    InProcessTransport,
    NexDashClient,
    NexLinkClient,
    SupportWorkspace,
    build_support_server,
)

HUB = "hub-token-0123456789abcdef"
CODER = "coder-token-0123456789abcdef"


def registry() -> ClientRegistry:
    return ClientRegistry.build(
        {"nexhub": (HUB, [SCOPE_INGEST, SCOPE_METRICS, SCOPE_SUGGEST]), "nexcoder": (CODER, [SCOPE_METRICS])}
    )


def envelope(action: str, params=None, sender: str = "nexhub", **kwargs) -> dict:
    body = {"sender": sender, "target": "nexsupport", "action": action, "params": params or {}}
    body.update(kwargs)
    return body


def server(**kwargs):
    return build_support_server(registry(), **kwargs)


def test_manifest_lists_actions_and_scopes() -> None:
    reply = server().handle_sync(envelope("nexlink.manifest"), HUB)
    assert reply.ok
    actions = {a["name"]: a["scope"] for a in reply.result["actions"]}
    assert actions == {"nexlink.manifest": None, "support.ingest": SCOPE_INGEST, "support.metrics": SCOPE_METRICS}
    copilot, _, _ = make_copilot()
    with_copilot = server(copilot_service=CopilotService(copilot)).manifest()
    assert with_copilot.action("support.suggest_reply").scope == SCOPE_SUGGEST
    assert with_copilot.action("support.ingest").params_schema["properties"]["tickets"]


def test_token_must_belong_to_the_sender() -> None:
    srv = server()
    for sender, token in (("nexhub", None), ("nexhub", "wrong-token-0123456789"), ("nexhub", CODER), ("other", HUB)):
        reply = srv.handle_sync(envelope("support.metrics", sender=sender), token)
        assert reply.error.code == ErrorCode.UNAUTHORIZED
        assert HUB not in reply.model_dump_json() and CODER not in reply.model_dump_json()


def test_token_inside_the_envelope_is_accepted() -> None:
    reply = server().handle_sync(envelope("support.metrics", token=HUB))
    assert reply.ok


def test_scope_per_action() -> None:
    srv = server(workspace=SupportWorkspace(tickets()))
    batch = [t.model_dump(mode="json") for t in tickets()]
    refused = srv.handle_sync(envelope("support.ingest", {"tickets": batch}, sender="nexcoder"), CODER)
    assert refused.error.code == ErrorCode.FORBIDDEN
    assert srv.handle_sync(envelope("support.metrics", sender="nexcoder"), CODER).ok


def test_errors_are_generic() -> None:
    srv = server()
    assert srv.handle_sync(envelope("support.delete"), HUB).error.code == ErrorCode.UNKNOWN_ACTION
    wrong = srv.handle_sync({**envelope("support.metrics"), "target": "nexdash"}, HUB)
    assert wrong.error.code == ErrorCode.WRONG_TARGET
    broken = srv.handle_sync({"sender": "nexhub", "params": {"secret": EMAIL}}, HUB)
    assert broken.error.code == ErrorCode.INVALID_ENVELOPE and broken.in_reply_to is None
    assert EMAIL not in broken.model_dump_json()


def test_invalid_params_do_not_echo_values() -> None:
    bad = {"tickets": [{"id": "X", "subject": f"mail {EMAIL}"}]}
    reply = server().handle_sync(envelope("support.ingest", bad), HUB)
    assert reply.error.code == ErrorCode.INVALID_PARAMS
    assert EMAIL not in reply.model_dump_json()
    assert reply.error.details and all("loc" in d and "type" in d for d in reply.error.details)
    extra = server().handle_sync(envelope("support.metrics", {"sql": "SELECT 1"}), HUB)
    assert extra.error.code == ErrorCode.INVALID_PARAMS
    assert "SELECT" not in extra.model_dump_json()


def test_ingest_answers_counters_and_masks_tickets(caplog) -> None:
    saved = []
    workspace = SupportWorkspace(sink=saved.extend)
    srv = server(workspace=workspace)
    batch = [open_ticket().model_dump(mode="json"), *[t.model_dump(mode="json") for t in tickets()]]
    with caplog.at_level(logging.DEBUG):
        reply = srv.handle_sync(envelope("support.ingest", {"tickets": batch}, correlation_id="run-7"), HUB)
    assert reply.ok and reply.correlation_id == "run-7"
    assert reply.result["ingested"] == 4 and reply.result["known_tickets"] == 4
    assert reply.result["stats"]["pii_masked"].get("EMAIL", 0) >= 1
    dumped = reply.model_dump_json()
    assert EMAIL not in dumped and "invoice export" not in dumped.lower()
    assert len(saved) == 4 and all(EMAIL not in t.thread_text() for t in saved)
    assert all(EMAIL not in t.thread_text() for t in workspace.tickets.values())
    assert EMAIL not in caplog.text and HUB not in caplog.text
    assert "action=support.ingest outcome=ok correlation=run-7" in caplog.text


def test_metrics_numbers_only_and_dashboard_from_nexdash() -> None:
    engine = FakeReportEngine()
    srv = server(workspace=SupportWorkspace(tickets()), nexdash=NexDashClient(EngineReportTransport(engine)))
    plain = srv.handle_sync(envelope("support.metrics", sender="nexcoder"), CODER)
    assert plain.ok and plain.result["tickets"] == 3 and "report" not in plain.result
    assert plain.result["by_status"] == {"open": 1, "solved": 2}
    assert 0 < plain.result["quality"]["mean_score"] <= 1
    assert engine.requests == []

    params = {"dashboard": True, "start": "2026-04-01", "end": "2026-06-30", "team": "tier1", "locale": "pt-BR"}
    reply = srv.handle_sync(envelope("support.metrics", params, sender="nexcoder"), CODER)
    assert reply.ok
    assert reply.result["report"]["model"] == "support_metrics"
    assert reply.result["report"]["dashboard"]["html"].startswith("<!doctype html>")
    assert engine.requests == [
        {
            "model": "support_metrics",
            "parameters": {"start": "2026-04-01", "end": "2026-06-30", "team": "tier1", "project": None},
            "output": "both",
            "locale": "pt-BR",
            "compare": True,
        }
    ]


def test_metrics_dashboard_failures() -> None:
    no_dash = server().handle_sync(envelope("support.metrics", {"dashboard": True}), HUB)
    assert no_dash.error.code == ErrorCode.UNAVAILABLE
    engine = FakeReportEngine(fail_with=500)
    srv = server(nexdash=NexDashClient(EngineReportTransport(engine)))
    reply = srv.handle_sync(envelope("support.metrics", {"dashboard": True}), HUB)
    assert reply.error.code == ErrorCode.UPSTREAM_ERROR and "500" in reply.error.message


def test_suggest_reply_through_nexlink() -> None:
    copilot, _, _ = make_copilot()
    srv = server(copilot_service=CopilotService(copilot))
    reply = srv.handle_sync(envelope("support.suggest_reply", {"ticket": open_ticket().model_dump(mode="json")}), HUB)
    assert reply.ok
    assert reply.result["reply"].endswith(GOOD) or GOOD in reply.result["reply"]
    assert EMAIL not in json.dumps(reply.result)
    coder = srv.handle_sync(
        envelope("support.suggest_reply", {"ticket": open_ticket().model_dump(mode="json")}, sender="nexcoder"), CODER
    )
    assert coder.error.code == ErrorCode.FORBIDDEN


def test_notice_runs_without_result_and_failures_are_logged_by_type(caplog) -> None:
    srv = server()
    notice = srv.handle_sync(envelope("support.metrics", kind="notice"), HUB)
    assert notice.ok and notice.result is None

    def boom(_params, _env):
        raise RuntimeError(f"leak {EMAIL}")

    srv.actions["support.metrics"] = srv.actions["support.metrics"].__class__(
        "support.metrics", "x", SCOPE_METRICS, srv.actions["support.metrics"].params, boom
    )
    with caplog.at_level(logging.DEBUG):
        reply = srv.handle_sync(envelope("support.metrics"), HUB)
    assert reply.error.code == ErrorCode.FAILED and EMAIL not in reply.model_dump_json()
    assert "RuntimeError" in caplog.text and EMAIL not in caplog.text


def test_client_round_trip_in_process() -> None:
    srv = server(workspace=SupportWorkspace(tickets()))
    client = NexLinkClient(
        "nexcoder",
        InProcessTransport(srv),
        environ={"SUPPORT_ENGINE_NEXLINK_TOKEN_NEXSUPPORT": CODER},
        env_file=None,
    )
    reply = client.request("nexsupport", "support.metrics", context={"run": "r1"})
    assert reply.ok and reply.target == "nexcoder" and reply.sender == "nexsupport"
    assert client.manifest("nexsupport").ok
    assert CODER not in repr(client)
