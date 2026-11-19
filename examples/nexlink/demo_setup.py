"""Shared setup of the NexLink examples. Offline: synthetic tickets, in-process transports.

- ``synthetic_tickets``: invented tickets (the e-mail addresses use example.com).
- ``report_transport``: the real nexdash report engine in process when nexdash is installed,
  otherwise ``FakeReportEngine``, a stub that answers in the same format.
- ``Demo``: a Support Engine NexLink server plus one client per calling project, each with its
  own freshly generated token and scopes. A real deployment reads the tokens from
  ``SUPPORT_ENGINE_NEXLINK_CLIENT_<NAME>_TOKEN`` and ``..._SCOPES`` instead.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from support_engine.models import Comment, Ticket, TicketStatus
from support_engine.nexlink import (
    SCOPE_INGEST,
    SCOPE_METRICS,
    ClientRegistry,
    EngineReportTransport,
    InProcessTransport,
    NexDashClient,
    NexLinkClient,
    NexLinkServer,
    SupportWorkspace,
    build_support_server,
)

T0 = datetime(2026, 5, 4, 9, 0, tzinfo=UTC)

# Which project may run which Support Engine action.
CLIENT_SCOPES = {
    "nexhub": [SCOPE_INGEST, SCOPE_METRICS],
    "nexcoder": [SCOPE_METRICS],
}


def _ticket(tid: str, subject: str, problem: str, answer: str | None, **kwargs: Any) -> Ticket:
    comments = [Comment(id=f"{tid}-c1", author_role="customer", body=problem, created_at=T0)]
    if answer:
        comments.append(Comment(id=f"{tid}-c2", author_role="agent", body=answer, created_at=T0 + timedelta(hours=2)))
    base: dict[str, Any] = {
        "id": tid,
        "source": "json",
        "subject": subject,
        "status": TicketStatus.SOLVED if answer else TicketStatus.OPEN,
        "csat": 0.9 if answer else None,
        "created_at": T0,
        "comments": comments,
    }
    base.update(kwargs)
    return Ticket(**base)


def synthetic_tickets() -> list[Ticket]:
    return [
        _ticket(
            "D-1",
            "Invoice export fails with error 500",
            "Hello, this is ana.example@example.com. The invoice export fails with error 500.",
            "Clear the export cache in Settings, then run the invoice export again.",
        ),
        _ticket(
            "D-2",
            "SSO login redirect loop",
            "After the SAML assertion users go back to the login screen. Call me at +1 555 0100.",
            "Enable the clock drift tolerance on the SAML verifier, then sign in again.",
        ),
        _ticket(
            "D-3",
            "Password reset email never arrives",
            "The password reset email never arrives for bruno.example@example.com.",
            None,
        ),
    ]


class EngineFailure(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class FakeReportEngine:
    """Stub with the ``run`` and ``list_models`` contract of ``nexdash.engine.ReportEngine``."""

    PANELS = ("volume", "first_response", "resolution_rate", "csat")
    FIELDS = frozenset({"model", "parameters", "output", "locale", "compare"})

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def list_models(self) -> list[dict[str, Any]]:
        return [{"name": "support_metrics", "title": "Support metrics"}]

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if set(request) - self.FIELDS:
            raise EngineFailure("Unknown field in the request", 400)
        if request.get("model") != "support_metrics":
            raise EngineFailure("Unknown report model", 404)
        title = "Support metrics"
        answer: dict[str, Any] = {
            "engine_version": "1",
            "model": "support_metrics",
            "title": title,
            "parameters": request.get("parameters", {}),
            "filters": "synthetic",
            "generated_at": "2026-07-01 12:00 UTC",
            "errors": {},
        }
        output = request.get("output", "both")
        if output in ("data", "both"):
            answer["data"] = {
                "panels": {
                    pid: {"kind": "kpi", "title": pid, "rows": [{"value": 1}], "value": 1, "error": None}
                    for pid in self.PANELS
                }
            }
        if output in ("dashboard", "both"):
            answer["dashboard"] = {
                "format": "html",
                "content_type": "text/html; charset=utf-8",
                "html": f"<!doctype html><html><head><title>{title}</title></head>"
                f"<body><main><svg></svg></main></body></html>",
            }
        return answer


def report_transport(prefer_real: bool = True) -> tuple[EngineReportTransport, str]:
    """The nexdash engine in process (synthetic data) when installed, the stub otherwise."""
    if prefer_real:
        try:
            return EngineReportTransport.demo(), "nexdash engine (in process, synthetic data)"
        except (ImportError, RuntimeError):
            pass
    return EngineReportTransport(FakeReportEngine()), "nexdash stub"


@dataclass
class Demo:
    server: NexLinkServer
    workspace: SupportWorkspace
    nexdash: NexDashClient
    nexdash_label: str
    clients: dict[str, NexLinkClient]


def build_demo(*, prefer_real_nexdash: bool = True, tickets: list[Ticket] | None = None) -> Demo:
    tokens = {name: secrets.token_urlsafe(32) for name in CLIENT_SCOPES}
    registry = ClientRegistry.build({name: (tokens[name], scopes) for name, scopes in CLIENT_SCOPES.items()})
    transport, label = report_transport(prefer_real_nexdash)
    nexdash = NexDashClient(transport)
    workspace = SupportWorkspace(tickets if tickets is not None else synthetic_tickets())
    server = build_support_server(registry, workspace=workspace, nexdash=nexdash)
    channel = InProcessTransport(server)
    clients = {
        name: NexLinkClient(name, channel, tokens={"nexsupport": tokens[name]}, env_file=None) for name in tokens
    }
    return Demo(server, workspace, nexdash, label, clients)
