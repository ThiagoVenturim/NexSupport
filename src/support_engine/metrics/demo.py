"""Reproducible NexSupport -> NexDash round trip, offline, on synthetic tickets.

``run_demo`` does what a deployment does, on 127.0.0.1 only:

1. ingest and analyse synthetic tickets and agent outcomes (``build_support_metrics``);
2. start NexDash with the support report models over a SQLite file and load the
   ``support_conversations`` and ``support_insights`` tables into it;
3. give the client a freshly generated token that holds only ``reports:list`` and
   ``reports:run`` (a deployment sets ``NEXDASH_CLIENT_NEXSUPPORT_TOKEN`` instead);
4. ask NexDash for the support dashboards over REST and over MCP, check the pages and
   that the made-up personal data of the tickets is nowhere in the answers.

Requires the ``nexdash`` package. Nothing leaves the machine.
"""

from __future__ import annotations

import json
import secrets
import tempfile
import threading
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from support_engine.metrics.client import NexDashClient, NexDashError
from support_engine.metrics.rows import build_support_metrics
from support_engine.metrics.synthetic import FAKE_EMAIL, FAKE_PHONE, synthetic_outcomes, synthetic_tickets

CLIENT_NAME = "nexsupport"
CLIENT_SCOPES = "reports:list,reports:run"
MODELS = ("support_metrics", "support_insights")
PERIOD = {"start": "2026-04-01", "end": "2026-06-30"}
PII = (FAKE_EMAIL, FAKE_PHONE, "maria.lima", "91234-5678")


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags[tag] = self.tags.get(tag, 0) + 1


def check_dashboard(html: str) -> list[str]:
    """Problems of a NexDash dashboard page; an empty list means it is valid."""
    problems = []
    if not html.lower().startswith("<!doctype html>"):
        problems.append("missing <!doctype html>")
    page = _Page()
    page.feed(html)
    page.close()
    for tag in ("html", "head", "body", "main"):
        if page.tags.get(tag) != 1:
            problems.append(f"expected exactly one <{tag}>")
    if page.tags.get("script"):
        problems.append("dashboard contains <script>")
    if not page.tags.get("svg"):
        problems.append("dashboard has no chart")
    return problems


def _leaks(texts: Iterable[str]) -> list[str]:
    joined = "\n".join(texts)
    return [value for value in PII if value in joined]


def _round_trip(client: NexDashClient, team: str | None) -> dict[str, Any]:
    names = {m["name"] for m in client.list_models()}
    missing = [m for m in MODELS if m not in names]
    reports = {m: client.run_report(m, team=team, **PERIOD) for m in MODELS if m in names}
    problems = [f"model {m} is not served" for m in missing]
    for name, report in reports.items():
        problems += [f"{name}: {p}" for p in check_dashboard(report["dashboard"]["html"])]
        problems += [f"{name}: panel {p} failed" for p in report["errors"]]
    try:
        client.run_report("support_insights", start="2026-06-30", end="2026-04-01")
        problems.append("a period that ends before it starts was accepted")
    except ValueError:
        pass
    try:
        client.run_report("no_such_model")
        problems.append("an unknown model was accepted")
    except NexDashError:
        pass
    problems += [f"personal data in the answer: {v}" for v in _leaks(json.dumps(r) for r in reports.values())]
    return {"reports": reports, "problems": problems}


def run_demo(out_dir: str | Path | None = None, *, team: str | None = None) -> dict[str, Any]:
    """Run the round trip and return KPIs per model, files written and problems found (none on success)."""
    from nexdash.auth import TokenAuth

    from support_engine.metrics.askdash import create_server

    tickets = synthetic_tickets()
    metrics = build_support_metrics(tickets, synthetic_outcomes(tickets))
    token = secrets.token_urlsafe(32)
    auth = TokenAuth(None, clients={CLIENT_NAME: (token, CLIENT_SCOPES)})
    summary: dict[str, Any] = {"tickets": len(metrics.tickets), "transports": {}, "files": [], "problems": []}

    with tempfile.TemporaryDirectory() as tmp:
        db_url = f"sqlite:///{Path(tmp, 'nexdash.db').as_posix()}"
        server = create_server("127.0.0.1", 0, db_url, auth=auth, metrics=metrics)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            for transport in ("rest", "mcp"):
                with NexDashClient(url, token, transport=transport) as client:
                    outcome = _round_trip(client, team)
                summary["problems"] += [f"{transport}: {p}" for p in outcome["problems"]]
                summary["transports"][transport] = {
                    name: {
                        pid: panel["value"] for pid, panel in report["data"]["panels"].items() if panel["kind"] == "kpi"
                    }
                    for name, report in outcome["reports"].items()
                }
                if transport == "rest" and out_dir is not None:
                    target = Path(out_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    for name, report in outcome["reports"].items():
                        path = target / f"{name}.html"
                        path.write_text(report["dashboard"]["html"], encoding="utf-8")
                        summary["files"].append(str(path))
            with NexDashClient(url, None) as anonymous:
                try:
                    anonymous.list_models()
                    summary["problems"].append("a request without a token was accepted")
                except NexDashError as exc:
                    if exc.status != 401:
                        summary["problems"].append(f"a request without a token answered {exc.status}")
        finally:
            server.shutdown()
            server.server_close()
            server.executor.close()
    if summary["transports"].get("rest") != summary["transports"].get("mcp"):
        summary["problems"].append("REST and MCP returned different KPIs")
    return summary


__all__ = ["check_dashboard", "run_demo"]
