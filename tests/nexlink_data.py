"""Offline stand-ins for nexdash used by the NexLink tests. All data is invented."""

from __future__ import annotations

from typing import Any


class EngineFailure(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def report_answer(request: dict[str, Any]) -> dict[str, Any]:
    """An answer in the format of the nexdash report engine (``docs/REPORT_ENGINE.md``)."""
    answer: dict[str, Any] = {
        "engine_version": "1",
        "model": request["model"],
        "title": "Support metrics",
        "parameters": request.get("parameters", {}),
        "filters": "test",
        "generated_at": "2026-07-01 12:00 UTC",
        "errors": {},
    }
    if request.get("output", "both") in ("data", "both"):
        answer["data"] = {"panels": {"volume": {"kind": "kpi", "rows": [{"ticket_count": 3}], "value": 3}}}
    if request.get("output", "both") in ("dashboard", "both"):
        answer["dashboard"] = {"format": "html", "html": "<!doctype html><html><body><main></main></body></html>"}
    return answer


class FakeReportEngine:
    """Records requests and answers like ``nexdash.engine.ReportEngine``."""

    def __init__(self, fail_with: int | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_with = fail_with

    def list_models(self) -> list[dict[str, Any]]:
        return [{"name": "support_metrics"}]

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if self.fail_with:
            raise EngineFailure("Report failed", self.fail_with)
        return report_answer(request)
