"""The Support Engine actions on NexLink.

=========================  =================  ==================================================
Action                     Scope              What it does
=========================  =================  ==================================================
``support.ingest``         ``ingest:run``     Clean, mask and chunk tickets; answers counters only
``support.suggest_reply``  ``copilot:suggest`` Suggested reply, summary, next step and tone
``support.metrics``        ``metrics:read``   Quality metrics; optionally the nexdash dashboard
``nexlink.manifest``       none               The list above, with the JSON schema of each action
=========================  =================  ==================================================

``support.suggest_reply`` is listed only when the server has a copilot, and the dashboard of
``support.metrics`` needs a ``NexDashClient``. Ticket text never leaves through ``ingest`` or
``metrics``: their answers are numbers. Tickets are masked before they are kept.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from support_engine.copilot.service import CopilotService, SuggestRequest
from support_engine.curation import CurationPipeline, QualityMetrics
from support_engine.ingestion import Chunk, IngestionPipeline, IngestionStats
from support_engine.models import Ticket
from support_engine.nexlink.nexdash import NexDashClient, ReportError, ReportParameters, ReportRequest
from support_engine.nexlink.protocol import Envelope, ErrorCode
from support_engine.nexlink.security import SCOPE_INGEST, SCOPE_METRICS, SCOPE_SUGGEST, ClientRegistry
from support_engine.nexlink.server import ActionError, NexLinkServer

PROJECT = "nexsupport"
MAX_TICKETS_PER_CALL = 500


class IngestParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickets: list[Ticket] = Field(min_length=1, max_length=MAX_TICKETS_PER_CALL)


class MetricsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dashboard: bool = Field(default=False, description="also ask nexdash for the support_metrics dashboard")
    output: Literal["data", "dashboard", "both"] = "both"
    start: dt.date | None = None
    end: dt.date | None = None
    team: str | None = Field(default=None, max_length=200)
    locale: Literal["en", "pt-BR"] = "en"


class SupportWorkspace:
    """Tickets the engine knows, already cleaned and masked, and their chunks.

    ``sink`` receives every ingested batch (for example ``TicketStore.save_tickets``).
    """

    def __init__(
        self,
        tickets: Iterable[Ticket] = (),
        *,
        pipeline: IngestionPipeline | None = None,
        curation: CurationPipeline | None = None,
        sink: Callable[[list[Ticket]], Any] | None = None,
    ) -> None:
        self.pipeline = pipeline or IngestionPipeline()
        self.curation = curation or CurationPipeline()
        self.sink = sink
        self.tickets: dict[str, Ticket] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        initial = list(tickets)
        if initial:
            self.ingest(initial)

    def ingest(self, tickets: Iterable[Ticket]) -> IngestionStats:
        result = self.pipeline.process(list(tickets))
        by_ticket: dict[str, list[Chunk]] = {}
        for chunk in result.chunks:
            by_ticket.setdefault(chunk.ticket_id, []).append(chunk)
        for ticket in result.tickets:
            self.tickets[ticket.id] = ticket
            self.chunks[ticket.id] = by_ticket.get(ticket.id, [])
        if self.sink is not None and result.tickets:
            self.sink(result.tickets)
        return result.stats

    def metrics(self) -> dict[str, Any]:
        tickets = list(self.tickets.values())
        chunks = [c for items in self.chunks.values() for c in items]
        quality: QualityMetrics = self.curation.curate(tickets, chunks).metrics if tickets else QualityMetrics()
        by_status: dict[str, int] = {}
        for ticket in tickets:
            by_status[ticket.status.value] = by_status.get(ticket.status.value, 0) + 1
        return {
            "tickets": len(tickets),
            "chunks": len(chunks),
            "by_status": dict(sorted(by_status.items())),
            "quality": quality.model_dump(mode="json"),
        }


class SupportActions:
    """Handlers of the Support Engine actions."""

    def __init__(
        self,
        workspace: SupportWorkspace,
        *,
        copilot_service: CopilotService | None = None,
        nexdash: NexDashClient | None = None,
    ) -> None:
        self.workspace = workspace
        self.copilot_service = copilot_service
        self.nexdash = nexdash

    def ingest(self, params: IngestParams, _envelope: Envelope) -> dict[str, Any]:
        stats = self.workspace.ingest(params.tickets)
        counters = stats.model_dump(mode="json", exclude={"duplicate_of"})
        return {"ingested": stats.tickets, "stats": counters, "known_tickets": len(self.workspace.tickets)}

    async def suggest_reply(self, params: SuggestRequest, _envelope: Envelope) -> dict[str, Any]:
        if self.copilot_service is None:
            raise ActionError(ErrorCode.UNAVAILABLE, "The copilot is not configured.")
        return await self.copilot_service.suggest(params)

    async def metrics(self, params: MetricsParams, _envelope: Envelope) -> dict[str, Any]:
        result = await asyncio.to_thread(self.workspace.metrics)
        if params.dashboard:
            if self.nexdash is None:
                raise ActionError(ErrorCode.UNAVAILABLE, "nexdash is not configured.")
            request = ReportRequest(
                parameters=ReportParameters(start=params.start, end=params.end, team=params.team),
                output=params.output,
                locale=params.locale,
            )
            try:
                result["report"] = await asyncio.to_thread(self.nexdash.run_report, request)
            except ReportError as exc:
                raise ActionError(ErrorCode.UPSTREAM_ERROR, f"nexdash answered {exc.status}.") from exc
        return result


def build_support_server(
    registry: ClientRegistry | None = None,
    *,
    workspace: SupportWorkspace | None = None,
    copilot_service: CopilotService | None = None,
    nexdash: NexDashClient | None = None,
    project: str = PROJECT,
    version: str = "0.1.0",
) -> NexLinkServer:
    """NexLink server with the Support Engine catalog. Without ``registry`` clients come from the env."""
    server = NexLinkServer(project, registry if registry is not None else ClientRegistry.from_env(), version=version)
    actions = SupportActions(workspace or SupportWorkspace(), copilot_service=copilot_service, nexdash=nexdash)
    server.register(
        "support.ingest",
        "Clean, mask, deduplicate and chunk support tickets. Answers counters only, never ticket text.",
        SCOPE_INGEST,
        IngestParams,
        actions.ingest,
    )
    if copilot_service is not None:
        server.register(
            "support.suggest_reply",
            "Suggested reply, summary, next step and tone for an open ticket, grounded in curated answers.",
            SCOPE_SUGGEST,
            SuggestRequest,
            actions.suggest_reply,
        )
    server.register(
        "support.metrics",
        "Ticket counts and answer quality metrics. With dashboard=true the Support Engine also asks nexdash "
        "for the support_metrics report and returns it.",
        SCOPE_METRICS,
        MetricsParams,
        actions.metrics,
    )
    server.support_actions = actions  # type: ignore[attr-defined]
    return server
