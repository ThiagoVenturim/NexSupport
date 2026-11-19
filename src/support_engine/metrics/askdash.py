"""Support metrics served by the askdash (NexDash) report engine.

Needs the ``nexdash`` package (``pip install -e <askdash checkout>``). Two report models
answer support questions:

- ``support_metrics`` (shipped with NexDash): volume, first response, resolution,
  deflection, CSAT and sentiment per period, team and channel;
- ``support_insights`` (defined here): the Support Engine analytics per period, team and
  queue: risk radar, SLA breaches, trend alerts, root cause candidates, sentiment and the
  ROI of the support agent.

The Support Engine writes both tables into the database NexDash reads
(``load_into_nexdash``); clients then ask NexDash for reports over MCP or REST
(``support_engine.metrics.client``). Report requests never carry SQL: panels are fixed
``QueryPlan`` objects compiled by the NexDash guarded compiler.
"""

from __future__ import annotations

import logging
from typing import Any

import sqlalchemy as sa
from nexdash.cube import Cube
from nexdash.engine import ReportEngine
from nexdash.executor import DatabaseExecutor, SQLiteExecutor, get_executor
from nexdash.mcp_server import MCPServer
from nexdash.plan import PlanFilter, PlanOrderBy, QueryPlan
from nexdash.report_models import REPORT_MODELS, Panel, ReportModel, support_rows_from_tickets

from support_engine.metrics.rows import SupportMetrics

logger = logging.getLogger(__name__)

SUPPORT_METRICS = REPORT_MODELS["support_metrics"]
INSIGHTS_TABLE_NAME = "support_insights"

_metadata = sa.MetaData()

INSIGHTS_TABLE = sa.Table(
    INSIGHTS_TABLE_NAME,
    _metadata,
    sa.Column("ticket_id", sa.String(200)),
    sa.Column("source", sa.String(64)),
    sa.Column("project", sa.String(200)),
    sa.Column("team", sa.String(200)),
    sa.Column("queue", sa.String(200)),
    sa.Column("channel", sa.String(200)),
    sa.Column("priority", sa.String(200)),
    sa.Column("status", sa.String(32)),
    sa.Column("opened_date", sa.String(10)),
    sa.Column("opened_month", sa.String(7)),
    sa.Column("resolved_flag", sa.Integer),
    sa.Column("reopened_flag", sa.Integer),
    sa.Column("sla_breached", sa.String(3)),
    sa.Column("sla_breached_flag", sa.Integer),
    sa.Column("first_response_hours", sa.Float),
    sa.Column("resolution_hours", sa.Float),
    sa.Column("sentiment", sa.String(16)),
    sa.Column("sentiment_rank", sa.Integer),
    sa.Column("sentiment_score", sa.Float),
    sa.Column("negative_flag", sa.Integer),
    sa.Column("worsening_flag", sa.Integer),
    sa.Column("risk_level", sa.String(16)),
    sa.Column("risk_rank", sa.Integer),
    sa.Column("risk_score", sa.Float),
    sa.Column("high_risk_flag", sa.Integer),
    sa.Column("risk_reason", sa.String(200)),
    sa.Column("trend_topic", sa.String(200)),
    sa.Column("trend_status", sa.String(16)),
    sa.Column("trend_alert", sa.String(3)),
    sa.Column("trend_alert_flag", sa.Integer),
    sa.Column("root_cause", sa.String(200)),
    sa.Column("agent_action", sa.String(200)),
    sa.Column("deflected_flag", sa.Integer),
    sa.Column("assisted_flag", sa.Integer),
    sa.Column("hours_saved", sa.Float),
    sa.Column("gross_savings", sa.Float),
    sa.Column("llm_cost", sa.Float),
    sa.Column("net_savings", sa.Float),
    sa.Column("csat", sa.Float),
)


def _metric(name: str, aggregation: str, column: str | None, description: str) -> dict[str, Any]:
    metric: dict[str, Any] = {"name": name, "aggregation": aggregation, "description": description}
    if column:
        metric["column"] = column
    return metric


INSIGHTS_CUBE = Cube.model_validate(
    {
        "name": "support_insights",
        "source": INSIGHTS_TABLE_NAME,
        "description": "Support Engine analytics per ticket: risk, SLA, trends, root cause, sentiment and agent ROI",
        "dimensions": [
            {"name": "ticket_id", "description": "Ticket identifier in the help desk"},
            {"name": "source", "description": "Help desk connector"},
            {"name": "project", "description": "Product or project the ticket is about"},
            {"name": "team", "description": "Support team (the queue when no team is set)"},
            {"name": "queue", "description": "Support queue (the team when no queue is set)"},
            {"name": "channel", "description": "Contact channel"},
            {"name": "priority", "description": "Ticket priority"},
            {"name": "status", "description": "Ticket status"},
            {"name": "opened_date", "type": "date", "description": "Date the ticket was opened (YYYY-MM-DD)"},
            {"name": "opened_month", "description": "Month the ticket was opened (YYYY-MM)"},
            {"name": "sla_breached", "description": "yes when a first response or resolution target was missed"},
            {"name": "sentiment", "description": "negative, neutral, positive or unknown"},
            {
                "name": "sentiment_rank",
                "type": "integer",
                "description": "1 negative, 2 neutral, 3 positive, 4 unknown",
            },
            {"name": "risk_level", "description": "high, medium, low or not_rated (resolved tickets)"},
            {"name": "risk_rank", "type": "integer", "description": "1 high, 2 medium, 3 low, 4 not_rated"},
            {"name": "risk_reason", "description": "Strongest risk signal, or none"},
            {"name": "trend_topic", "description": "Topic keywords of the current trend window, or none"},
            {"name": "trend_status", "description": "new, spike, rising, falling, stable or none"},
            {"name": "trend_alert", "description": "yes when the topic of the ticket raised a trend alert"},
            {"name": "root_cause", "description": "Strongest root cause candidate that covers the ticket, or none"},
            {"name": "agent_action", "description": "What the support agent did: answer, review, escalate, refuse"},
        ],
        "metrics": [
            _metric("ticket_count", "count", None, "Number of tickets"),
            _metric("high_risk_tickets", "sum", "high_risk_flag", "Open tickets with high churn or escalation risk"),
            _metric("avg_risk_score", "avg", "risk_score", "Average risk score of rated tickets (0 to 1)"),
            _metric("negative_share", "avg", "negative_flag", "Share of tickets with negative sentiment"),
            _metric("worsening_share", "avg", "worsening_flag", "Share of tickets whose customer got more negative"),
            _metric("avg_sentiment_score", "avg", "sentiment_score", "Average sentiment score (-1 to 1)"),
            _metric("sla_breach_rate", "avg", "sla_breached_flag", "Share of tickets past an SLA target"),
            _metric("reopen_rate", "avg", "reopened_flag", "Share of tickets reopened at least once"),
            _metric("resolution_rate", "avg", "resolved_flag", "Share of tickets solved or closed"),
            _metric("avg_first_response_hours", "avg", "first_response_hours", "Average hours to the first reply"),
            _metric("avg_resolution_hours", "avg", "resolution_hours", "Average hours to resolve"),
            _metric("trend_alert_tickets", "sum", "trend_alert_flag", "Tickets in topics that raised a trend alert"),
            _metric("deflected_tickets", "sum", "deflected_flag", "Tickets answered by the agent at level 1"),
            _metric("deflection_rate", "avg", "deflected_flag", "Share of tickets answered by the agent at level 1"),
            _metric("hours_saved", "sum", "hours_saved", "Human hours saved by the agent"),
            _metric("gross_savings", "sum", "gross_savings", "Value of the hours saved"),
            _metric("llm_cost", "sum", "llm_cost", "Model cost of the agent"),
            _metric("net_savings", "sum", "net_savings", "Gross savings minus model cost"),
            _metric("avg_csat", "avg", "csat", "Average satisfaction from 0 to 1"),
        ],
    }
)


def _plan(**kwargs: Any) -> QueryPlan:
    return QueryPlan(cube=INSIGHTS_CUBE.name, **kwargs)


def _monthly(metric: str) -> QueryPlan:
    return _plan(metrics=[metric], dimensions=["opened_month"], order_by=[PlanOrderBy(field="opened_month")])


def _top(metric: str, dimension: str, *, exclude_none: bool = False, limit: int = 10) -> QueryPlan:
    filters = [PlanFilter(field=dimension, operator="neq", value="none")] if exclude_none else []
    return _plan(
        metrics=[metric],
        dimensions=[dimension],
        filters=filters,
        order_by=[PlanOrderBy(field=metric, direction="desc")],
        limit=limit,
    )


INSIGHTS_PANELS: tuple[Panel, ...] = (
    Panel(
        id="high_risk",
        kind="kpi",
        title="Open tickets at high risk",
        plan=_plan(metrics=["high_risk_tickets"]),
        y="high_risk_tickets",
        better="lower",
        description="Churn, escalation and SLA risk from the risk radar",
    ),
    Panel(
        id="negative_share",
        kind="kpi",
        title="Tickets with negative sentiment",
        plan=_plan(metrics=["negative_share"]),
        y="negative_share",
        unit="percent",
        better="lower",
    ),
    Panel(
        id="sla_breach_rate",
        kind="kpi",
        title="Tickets past an SLA target",
        plan=_plan(metrics=["sla_breach_rate"]),
        y="sla_breach_rate",
        unit="percent",
        better="lower",
    ),
    Panel(
        id="trend_alert_tickets",
        kind="kpi",
        title="Tickets in alerting topics",
        plan=_plan(metrics=["trend_alert_tickets"]),
        y="trend_alert_tickets",
        better="lower",
        description="Topics growing faster than their own baseline",
    ),
    Panel(
        id="hours_saved",
        kind="kpi",
        title="Agent hours saved by the support agent",
        plan=_plan(metrics=["hours_saved"]),
        y="hours_saved",
        unit="hours",
    ),
    Panel(
        id="sentiment_trend",
        kind="line",
        title="Average customer sentiment per month",
        plan=_monthly("avg_sentiment_score"),
        x="opened_month",
        y="avg_sentiment_score",
        unit="score",
    ),
    Panel(
        id="risk_trend",
        kind="line",
        title="Average risk score per month",
        plan=_monthly("avg_risk_score"),
        x="opened_month",
        y="avg_risk_score",
        unit="score",
        better="lower",
    ),
    Panel(
        id="risk_levels",
        kind="bar",
        title="Tickets by risk level",
        plan=_plan(
            metrics=["ticket_count"],
            dimensions=["risk_rank", "risk_level"],
            order_by=[PlanOrderBy(field="risk_rank")],
        ),
        x="risk_level",
        y="ticket_count",
        keep_order=True,
        highlight="high",
    ),
    Panel(
        id="risk_reasons",
        kind="bar",
        title="Strongest risk signal",
        plan=_top("ticket_count", "risk_reason", exclude_none=True),
        x="risk_reason",
        y="ticket_count",
    ),
    Panel(
        id="trend_topics",
        kind="bar",
        title="Tickets per topic in the current trend window",
        plan=_top("ticket_count", "trend_topic", exclude_none=True),
        x="trend_topic",
        y="ticket_count",
        better="none",
    ),
    Panel(
        id="root_causes",
        kind="bar",
        title="Root cause candidates of negative, high risk and trending tickets",
        plan=_top("ticket_count", "root_cause", exclude_none=True),
        x="root_cause",
        y="ticket_count",
        better="none",
    ),
    Panel(
        id="sentiment_by_queue",
        kind="bar",
        title="Average customer sentiment per queue",
        plan=_plan(
            metrics=["avg_sentiment_score"],
            dimensions=["queue"],
            order_by=[PlanOrderBy(field="avg_sentiment_score")],
            limit=10,
        ),
        x="queue",
        y="avg_sentiment_score",
        unit="score",
    ),
    Panel(
        id="sla_by_queue",
        kind="bar",
        title="Share of tickets past an SLA target per queue",
        plan=_top("sla_breach_rate", "queue"),
        x="queue",
        y="sla_breach_rate",
        unit="percent",
        better="lower",
    ),
    Panel(
        id="net_savings_by_team",
        kind="bar",
        title="Net savings of the support agent per team (agent cost minus model cost)",
        plan=_top("net_savings", "team"),
        x="team",
        y="net_savings",
        unit="score",
    ),
)

INSIGHTS_MODEL = ReportModel(
    name="support_insights",
    title="Support insights",
    description="Risk radar, SLA breaches, trend alerts, root causes, sentiment and agent ROI per period, team, queue.",
    cube=INSIGHTS_CUBE,
    table=INSIGHTS_TABLE,
    panels=INSIGHTS_PANELS,
    period_field="opened_date",
    source_note="Support Engine analytics over masked help desk tickets",
)

SUPPORT_MODELS: dict[str, ReportModel] = {SUPPORT_METRICS.name: SUPPORT_METRICS, INSIGHTS_MODEL.name: INSIGHTS_MODEL}


def report_models() -> dict[str, ReportModel]:
    """Every NexDash report model plus ``support_insights``."""
    return {**REPORT_MODELS, INSIGHTS_MODEL.name: INSIGHTS_MODEL}


def conversation_rows(metrics: SupportMetrics) -> list[dict[str, Any]]:
    """``support_conversations`` rows (the NexDash ``support_metrics`` table)."""
    return support_rows_from_tickets(metrics.conversation_records())


def load_into_nexdash(
    executor: DatabaseExecutor | SQLiteExecutor,
    metrics: SupportMetrics,
    *,
    replace: bool = True,
) -> dict[str, int]:
    """Write ``support_conversations`` and ``support_insights`` into the NexDash database.

    With ``replace=True`` (the default) both tables are rebuilt, so a refresh does not
    duplicate tickets. Returns the number of rows written per model.
    """
    counts = {
        SUPPORT_METRICS.name: SUPPORT_METRICS.load_rows(executor, conversation_rows(metrics), replace=replace),
        INSIGHTS_MODEL.name: INSIGHTS_MODEL.load_rows(executor, metrics.insight_rows(), replace=replace),
    }
    logger.info("metrics: loaded %s", ", ".join(f"{name}={n}" for name, n in counts.items()))
    return counts


def build_engine(executor: DatabaseExecutor | SQLiteExecutor, **kwargs: Any) -> ReportEngine:
    """A NexDash ``ReportEngine`` that serves every NexDash model plus ``support_insights``."""
    return ReportEngine(executor, report_models(), **kwargs)


def _register_cube(service: Any) -> None:
    registry = getattr(service, "registry", None)
    if registry is not None and INSIGHTS_CUBE.name not in registry:
        registry.register(INSIGHTS_CUBE)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    db_url: str | None = None,
    *,
    auth: Any = None,
    metrics: SupportMetrics | None = None,
) -> Any:
    """The NexDash HTTP server (REST and MCP) with ``support_insights`` registered.

    Tokens come from the environment as in NexDash (``NEXDASH_API_TOKEN`` and
    ``NEXDASH_CLIENT_<NAME>_TOKEN`` / ``_SCOPES``) unless ``auth`` is given. Call
    ``serve_forever`` to start it and ``shutdown`` to stop it. ``metrics``, when given,
    is loaded into the server database first.
    """
    from nexdash.server import run_server

    server = run_server(host, port, db_url, report_models=True, auth=auth)
    if metrics is not None:
        load_into_nexdash(server.executor, metrics)
    server.engine = build_engine(server.executor)
    _register_cube(server.service)
    server.mcp = MCPServer(server.service, server.engine)
    return server


def serve_mcp_stdio(db_url: str | None = None, *, metrics: SupportMetrics | None = None) -> None:
    """Serve the NexDash MCP tools, ``support_insights`` included, over stdin and stdout."""
    from nexdash.db import get_db_url
    from nexdash.server import build_service

    executor = get_executor(db_url or get_db_url())
    try:
        if metrics is not None:
            load_into_nexdash(executor, metrics)
        service = build_service(executor, report_models=True)
        _register_cube(service)
        MCPServer(service, build_engine(executor)).serve_stdio()
    finally:
        executor.close()


__all__ = [
    "INSIGHTS_CUBE",
    "INSIGHTS_MODEL",
    "INSIGHTS_TABLE",
    "SUPPORT_METRICS",
    "SUPPORT_MODELS",
    "build_engine",
    "conversation_rows",
    "create_server",
    "load_into_nexdash",
    "report_models",
    "serve_mcp_stdio",
]
