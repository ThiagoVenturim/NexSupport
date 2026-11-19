"""Per-ticket metric rows for the askdash (NexDash) report engine. No askdash import here.

``SupportMetrics`` joins masked tickets, their ``AnalyticsReport`` and the support agent
outcomes, and derives two record sets:

- ``conversation_records``: tickets in the shape the NexDash ``support_metrics`` model
  reads (``support_rows_from_tickets``), with sentiment, team and deflection filled in
  by the analytics. Comments keep only their role, visibility and time: no message text.
- ``insight_rows``: one row per ticket for the ``support_insights`` model: sentiment,
  risk level and score, SLA breach, reopen, trend topic and alert, root cause candidate,
  deflection and the ROI of the agent on that ticket.

Rows hold ids, labels, numbers, folded topic tokens and feature names, never message
text, subjects or personal data. The fields used as team, queue and channel come from
``Ticket.extra``; do not point them at keys that hold personal data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from support_engine.analytics import (
    AnalyticsConfig,
    AnalyticsPipeline,
    AnalyticsReport,
    Outcome,
    compute_roi,
    first_response_hours,
    latest_timestamp,
    resolution_hours,
    sla_breached,
)
from support_engine.models import Ticket

logger = logging.getLogger(__name__)

UNASSIGNED = "unassigned"
NONE = "none"
MAX_LABEL_CHARS = 200
SENTIMENTS: tuple[str, ...] = ("negative", "neutral", "positive", "unknown")
RISK_LEVELS: tuple[str, ...] = ("high", "medium", "low", "not_rated")
TOPIC_KEYWORDS = 3

TeamOf = Mapping[str, str] | Callable[[str], str | None]


def _label(value: Any, default: str = UNASSIGNED) -> str:
    text = "" if value is None else str(value).strip()
    return text[:MAX_LABEL_CHARS] if text else default


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    moment = _utc(value)
    return moment.isoformat() if moment is not None else None


def _outcome(item: Any) -> Outcome:
    return Outcome.of(dict(item) if isinstance(item, Mapping) else item)


def _ticket_id(item: Any) -> str | None:
    value = item.get("ticket_id") if isinstance(item, Mapping) else getattr(item, "ticket_id", None)
    return str(value) if value else None


def split_outcomes(agent_results: Iterable[Any] | Mapping[str, Any] | None) -> tuple[list[Outcome], dict[str, Outcome]]:
    """Normalise agent results into all outcomes plus the ones tied to a ticket id.

    Accepts a mapping ``ticket_id -> outcome`` or an iterable of ``Outcome``, ``AgentResult``
    or dicts. Dicts and objects tied to a ticket carry ``ticket_id``; outcomes without one
    still count in the overall ROI but not in the per-ticket rows.
    """
    if agent_results is None:
        return [], {}
    by_ticket: dict[str, Outcome] = {}
    everything: list[Outcome] = []
    if isinstance(agent_results, Mapping):
        for ticket_id, item in agent_results.items():
            outcome = _outcome(item)
            by_ticket[str(ticket_id)] = outcome
            everything.append(outcome)
        return everything, by_ticket
    for item in agent_results:
        outcome = _outcome(item)
        everything.append(outcome)
        ticket_id = _ticket_id(item)
        if ticket_id:
            by_ticket[ticket_id] = outcome
    return everything, by_ticket


@dataclass
class SupportMetrics:
    """Masked tickets, their analytics report and the agent outcomes per ticket."""

    tickets: list[Ticket]
    report: AnalyticsReport
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    config: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    team_key: str = "team"
    queue_key: str = "queue"
    channel_key: str = "channel"
    team_of: TeamOf | None = None

    # -- shared fields -----------------------------------------------------

    def team(self, ticket: Ticket) -> str:
        """Team from ``extra[team_key]``, then the queue, then ``team_of(product)``."""
        extra = ticket.extra or {}
        team = extra.get(self.team_key) or extra.get(self.queue_key)
        if not team and self.team_of is not None and ticket.product:
            team = self.team_of(ticket.product) if callable(self.team_of) else self.team_of.get(ticket.product)
        return _label(team)

    def queue(self, ticket: Ticket) -> str:
        extra = ticket.extra or {}
        return _label(extra.get(self.queue_key) or extra.get(self.team_key))

    def channel(self, ticket: Ticket) -> str:
        return _label((ticket.extra or {}).get(self.channel_key) or ticket.source)

    def sentiment(self, ticket: Ticket) -> tuple[str, float | None, float | None]:
        """Label, score and first-to-last delta from the analytics; ``unknown`` without a reading."""
        found = self.report.ticket_sentiment.get(ticket.id)
        if found is None:
            return "unknown", None, None
        label = found.label if found.label in SENTIMENTS else "unknown"
        return label, round(found.score, 4), round(found.delta, 4)

    def deflected(self, ticket: Ticket) -> bool | None:
        """True for a level 1 answer of the agent, False for other outcomes, None without one."""
        outcome = self.outcomes.get(ticket.id)
        if outcome is None:
            flag = (ticket.extra or {}).get("deflected")
            return flag if isinstance(flag, bool) else None
        return outcome.action == "answer" and outcome.level == 1

    @property
    def now(self) -> datetime | None:
        return _utc(self.report.generated_at) or latest_timestamp(self.tickets)

    # -- NexDash support_metrics -------------------------------------------

    def conversation_records(self) -> list[dict[str, Any]]:
        """Tickets for ``nexdash.report_models.support_rows_from_tickets``, without any text."""
        records = []
        for ticket in self.tickets:
            label, score, _delta = self.sentiment(ticket)
            extra: dict[str, Any] = {"team": self.team(ticket), "channel": self.channel(ticket), "sentiment": label}
            if score is not None:
                extra["sentiment_score"] = score
            deflected = self.deflected(ticket)
            if deflected is not None:
                extra["deflected"] = deflected
            records.append(
                {
                    "id": ticket.id,
                    "source": ticket.source,
                    "status": ticket.status.value,
                    "priority": ticket.priority,
                    "product": ticket.product,
                    "tags": list(ticket.tags),
                    "created_at": _iso(ticket.created_at),
                    "solved_at": _iso(ticket.solved_at),
                    "csat": ticket.csat,
                    "comments": [
                        {"author_role": c.author_role, "public": c.public, "created_at": _iso(c.created_at)}
                        for c in ticket.comments
                    ],
                    "extra": extra,
                }
            )
        return records

    # -- support_insights --------------------------------------------------

    def insight_rows(self) -> list[dict[str, Any]]:
        """One ``support_insights`` row per ticket."""
        cfg = self.config
        now = self.now
        risk = {a.ticket_id: a for a in self.report.risk}
        trends: dict[str, tuple[str, str, bool]] = {}
        for trend in sorted(self.report.trends, key=lambda t: (not t.alert, -t.z_score)):
            topic = _label(" ".join(trend.keywords[:TOPIC_KEYWORDS]) or trend.cluster_id, NONE)
            for ticket_id in trend.ticket_ids:
                trends.setdefault(ticket_id, (topic, trend.status, trend.alert))
        causes: dict[str, tuple[float, str]] = {}
        for group in self.report.root_causes:
            for cause in group.causes:
                for ticket_id in cause.ticket_ids:
                    if ticket_id not in causes or cause.score > causes[ticket_id][0]:
                        causes[ticket_id] = (cause.score, cause.feature)
        roi_config = cfg.roi.model_copy(update={"fixed_cost": 0.0})
        agent_roles = cfg.risk.agent_roles

        rows = []
        for ticket in self.tickets:
            created = _utc(ticket.created_at)
            label, score, delta = self.sentiment(ticket)
            assessment = risk.get(ticket.id)
            level = assessment.level if assessment is not None else "not_rated"
            topic, trend_status, alert = trends.get(ticket.id, (NONE, NONE, False))
            outcome = self.outcomes.get(ticket.id)
            roi = compute_roi([outcome], roi_config) if outcome is not None else None
            breached = bool(now is not None and sla_breached(ticket, now, cfg.risk.sla, agent_roles))
            first = first_response_hours(ticket, agent_roles)
            solved = resolution_hours(ticket)
            rows.append(
                {
                    "ticket_id": ticket.id,
                    "source": _label(ticket.source),
                    "project": _label(ticket.product),
                    "team": self.team(ticket),
                    "queue": self.queue(ticket),
                    "channel": self.channel(ticket),
                    "priority": _label(ticket.priority, NONE),
                    "status": ticket.status.value,
                    "opened_date": created.date().isoformat() if created else None,
                    "opened_month": created.strftime("%Y-%m") if created else None,
                    "resolved_flag": int(ticket.is_resolved),
                    "reopened_flag": int(ticket.reopen_count > 0),
                    "sla_breached": "yes" if breached else "no",
                    "sla_breached_flag": int(breached),
                    "first_response_hours": None if first is None else round(first, 2),
                    "resolution_hours": None if solved is None else round(solved, 2),
                    "sentiment": label,
                    "sentiment_rank": SENTIMENTS.index(label) + 1,
                    "sentiment_score": score,
                    "negative_flag": int(label == "negative"),
                    "worsening_flag": int(delta is not None and delta < 0),
                    "risk_level": level,
                    "risk_rank": RISK_LEVELS.index(level) + 1,
                    "risk_score": None if assessment is None else round(assessment.score, 4),
                    "high_risk_flag": int(level == "high"),
                    "risk_reason": _label(assessment.reasons[0], NONE) if assessment and assessment.reasons else NONE,
                    "trend_topic": topic,
                    "trend_status": trend_status,
                    "trend_alert": "yes" if alert else "no",
                    "trend_alert_flag": int(alert),
                    "root_cause": _label(causes[ticket.id][1], NONE) if ticket.id in causes else NONE,
                    "agent_action": _label(outcome.action, NONE) if outcome is not None else NONE,
                    "deflected_flag": int(bool(roi and roi.answered)),
                    "assisted_flag": int(bool(roi and roi.assisted)),
                    "hours_saved": roi.hours_saved if roi else 0.0,
                    "gross_savings": roi.gross_savings if roi else 0.0,
                    "llm_cost": roi.llm_cost if roi else 0.0,
                    "net_savings": roi.net_savings if roi else 0.0,
                    "csat": ticket.csat,
                }
            )
        return rows


def build_support_metrics(
    tickets: Iterable[Ticket],
    agent_results: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    config: AnalyticsConfig | None = None,
    now: datetime | None = None,
    ingest: bool = True,
    team_key: str = "team",
    queue_key: str = "queue",
    channel_key: str = "channel",
    team_of: TeamOf | None = None,
) -> SupportMetrics:
    """Ingest (clean and mask) the tickets, run the analytics and collect the metric inputs.

    Root cause candidates keep every ticket id here (``max_ticket_ids`` is raised to the
    number of tickets) so the dashboard counts each ticket under its strongest cause.
    """
    rows = list(tickets)
    if ingest:
        from support_engine.ingestion import IngestionPipeline

        rows = IngestionPipeline().process(rows).tickets
    cfg = (config or AnalyticsConfig()).model_copy(deep=True)
    cfg.root_cause.max_ticket_ids = max(cfg.root_cause.max_ticket_ids, len(rows))
    everything, by_ticket = split_outcomes(agent_results)
    report = AnalyticsPipeline(cfg).run(rows, everything if agent_results is not None else None, now=now)
    logger.info("metrics: %d tickets, %d outcomes tied to a ticket", len(rows), len(by_ticket))
    return SupportMetrics(
        tickets=rows,
        report=report,
        outcomes=by_ticket,
        config=cfg,
        team_key=team_key,
        queue_key=queue_key,
        channel_key=channel_key,
        team_of=team_of,
    )


__all__ = [
    "RISK_LEVELS",
    "SENTIMENTS",
    "SupportMetrics",
    "build_support_metrics",
    "split_outcomes",
]
