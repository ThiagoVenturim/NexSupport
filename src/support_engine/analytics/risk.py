"""Risk radar and service quality.

Each open ticket gets a risk score in 0..1 from signals that each lie in 0..1:

- ``negative_sentiment``: customer sentiment below 0, full at ``-negative_score``;
- ``sentiment_drop``: the last customer message is worse than the first, full at ``sentiment_drop``;
- ``churn_terms``: the customer mentions cancelling, refunds, lawyers and alike (en, pt, es);
- ``reopens``: ``reopen_count / max_reopens``;
- ``sla_breach``: 1 past the first response or resolution target, rising from half the target;
- ``waiting``: hours since the last unanswered customer message over ``waiting_hours``;
- ``priority``: ``priority_levels[priority]``;
- ``low_csat``: CSAT at or below ``low_csat``.

Signals combine as a noisy OR, ``risk = 1 - prod(1 - weight * signal)``, so one strong
signal is enough to raise the risk and several weak ones add up. The report keeps numbers,
signal names and the matched churn terms from the configured list, never ticket text.
``now`` defaults to the latest timestamp found in the tickets, which keeps reports
reproducible; pass the real clock for live use.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, Field

from support_engine.analytics.config import QualityConfig, RiskConfig, SLAConfig
from support_engine.analytics.sentiment import TicketSentiment
from support_engine.analytics.text import customer_messages, fold, ticket_field
from support_engine.models import Ticket

RiskLevel = Literal["low", "medium", "high"]


class RiskAssessment(BaseModel):
    """Risk of one ticket. Numbers, signal names and configured terms only."""

    ticket_id: str
    score: float = Field(ge=0.0, le=1.0)
    level: RiskLevel
    signals: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list, description="signals at or above 0.5, strongest first")
    churn_terms: list[str] = Field(default_factory=list)
    status: str
    priority: str | None = None
    product: str | None = None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _hours(start: datetime, end: datetime) -> float:
    return max(0.0, (_utc(end) - _utc(start)).total_seconds() / 3600)


def latest_timestamp(tickets: Iterable[Ticket]) -> datetime | None:
    """Most recent creation, update, solve or comment time in ``tickets``."""
    moments = []
    for t in tickets:
        moments += [t.created_at, *(m for m in (t.updated_at, t.solved_at) if m), *(c.created_at for c in t.comments)]
    return max((_utc(m) for m in moments), default=None)


def first_response_hours(ticket: Ticket, agent_roles: Iterable[str] = ("agent",)) -> float | None:
    """Hours from creation to the first public agent reply; None without a reply."""
    roles = set(agent_roles)
    replies = [c.created_at for c in ticket.comments if c.public and c.author_role in roles]
    return _hours(ticket.created_at, min(replies, key=_utc)) if replies else None


def resolution_hours(ticket: Ticket) -> float | None:
    if not ticket.is_resolved or ticket.solved_at is None:
        return None
    return _hours(ticket.created_at, ticket.solved_at)


def sla_signal(ticket: Ticket, now: datetime, sla: SLAConfig, agent_roles: Iterable[str] = ("agent",)) -> float:
    """1 when a target is missed; rises linearly from half the target for an open ticket."""
    first = first_response_hours(ticket, agent_roles)
    first_target = sla.first_response_target(ticket.priority)
    resolution_target = sla.resolution_target(ticket.priority)
    if first is not None and first > first_target:
        return 1.0
    if ticket.is_resolved:
        spent = resolution_hours(ticket)
        return 1.0 if spent is not None and spent > resolution_target else 0.0
    elapsed = _hours(ticket.created_at, now)
    ratios = [elapsed / resolution_target if resolution_target else 0.0]
    if first is None:
        ratios.append(elapsed / first_target if first_target else 0.0)
    ratio = max(ratios)
    if ratio >= 1.0:
        return 1.0
    return round(max(0.0, (ratio - 0.5) / 0.5), 4)


def sla_breached(ticket: Ticket, now: datetime, sla: SLAConfig, agent_roles: Iterable[str] = ("agent",)) -> bool:
    return sla_signal(ticket, now, sla, agent_roles) >= 1.0


def waiting_hours(ticket: Ticket, now: datetime, agent_roles: Iterable[str] = ("agent",)) -> float:
    """Hours the customer has been waiting since their last public message without an agent reply."""
    if ticket.is_resolved:
        return 0.0
    roles = set(agent_roles)
    public = sorted((c for c in ticket.comments if c.public), key=lambda c: _utc(c.created_at))
    if not public:
        return _hours(ticket.created_at, now)
    last = public[-1]
    if last.author_role in roles:
        return 0.0
    return _hours(last.created_at, now)


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    """Configured terms present in ``text`` as whole words, accents and case ignored."""
    folded = fold(text)
    found = []
    for term in terms:
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in fold(term).split()) + r"\b"
        if re.search(pattern, folded) and term not in found:
            found.append(term)
    return found


class RiskRadar:
    """Scores tickets for churn, escalation and SLA risk."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def level_for(self, score: float) -> RiskLevel:
        if score >= self.config.high:
            return "high"
        if score >= self.config.medium:
            return "medium"
        return "low"

    def signals(self, ticket: Ticket, sentiment: TicketSentiment | None, now: datetime) -> tuple[dict, list[str]]:
        cfg = self.config
        out: dict[str, float] = {}
        if sentiment is not None:
            out["negative_sentiment"] = min(1.0, max(0.0, -sentiment.score) / cfg.negative_score)
            drop = -sentiment.delta if sentiment.messages > 1 else 0.0
            out["sentiment_drop"] = min(1.0, max(0.0, drop) / cfg.sentiment_drop)
        terms = find_terms("\n".join(customer_messages(ticket)), cfg.churn_terms)
        out["churn_terms"] = 1.0 if terms else 0.0
        out["reopens"] = min(1.0, ticket.reopen_count / cfg.max_reopens)
        out["sla_breach"] = sla_signal(ticket, now, cfg.sla, cfg.agent_roles)
        out["waiting"] = min(1.0, waiting_hours(ticket, now, cfg.agent_roles) / cfg.waiting_hours)
        out["priority"] = cfg.priority_levels.get((ticket.priority or "").lower(), 0.0)
        out["low_csat"] = 1.0 if ticket.csat is not None and ticket.csat <= cfg.low_csat else 0.0
        return {k: round(v, 4) for k, v in out.items()}, terms

    def assess(self, ticket: Ticket, sentiment: TicketSentiment | None = None, now: datetime | None = None):
        moment = _utc(now) if now else (latest_timestamp([ticket]) or _utc(ticket.created_at))
        signals, terms = self.signals(ticket, sentiment, moment)
        weights = self.config.weights.model_dump()
        keep = 1.0
        for name, value in signals.items():
            keep *= 1.0 - weights.get(name, 0.0) * value
        score = round(1.0 - keep, 4)
        reasons = sorted((n for n, v in signals.items() if v >= 0.5), key=lambda n: -weights.get(n, 0.0) * signals[n])
        return RiskAssessment(
            ticket_id=ticket.id,
            score=score,
            level=self.level_for(score),
            signals=signals,
            reasons=reasons,
            churn_terms=terms,
            status=ticket.status.value,
            priority=ticket.priority,
            product=ticket.product,
        )

    def radar(
        self,
        tickets: Iterable[Ticket],
        sentiments: Mapping[str, TicketSentiment] | None = None,
        now: datetime | None = None,
    ) -> list[RiskAssessment]:
        """Assessments of the open tickets (all with ``include_resolved``), riskiest first."""
        rows = list(tickets)
        moment = _utc(now) if now else latest_timestamp(rows)
        sentiments = sentiments or {}
        out = [
            self.assess(t, sentiments.get(t.id), moment)
            for t in rows
            if self.config.include_resolved or not t.is_resolved
        ]
        return sorted(out, key=lambda a: (-a.score, a.ticket_id))


class GroupQuality(BaseModel):
    """Service quality numbers of one group of tickets."""

    key: str
    tickets: int = 0
    resolved: int = 0
    resolved_rate: float = 0.0
    reopen_rate: float = 0.0
    mean_first_response_hours: float | None = None
    median_resolution_hours: float | None = None
    sla_breach_rate: float = 0.0
    mean_csat: float | None = None
    mean_sentiment: float | None = None
    negative_share: float | None = None
    high_risk: int = 0


class QualityReport(BaseModel):
    group_by: str
    overall: GroupQuality
    groups: list[GroupQuality] = Field(default_factory=list)


def _group_quality(
    key: str,
    tickets: list[Ticket],
    now: datetime,
    risk: RiskConfig,
    sentiments: Mapping[str, TicketSentiment],
    high_risk: set[str],
) -> GroupQuality:
    g = GroupQuality(key=key, tickets=len(tickets))
    if not tickets:
        return g
    n = len(tickets)
    g.resolved = sum(t.is_resolved for t in tickets)
    g.resolved_rate = round(g.resolved / n, 4)
    g.reopen_rate = round(sum(t.reopen_count > 0 for t in tickets) / n, 4)
    firsts = [h for h in (first_response_hours(t, risk.agent_roles) for t in tickets) if h is not None]
    if firsts:
        g.mean_first_response_hours = round(mean(firsts), 4)
    spent = [h for h in (resolution_hours(t) for t in tickets) if h is not None]
    if spent:
        g.median_resolution_hours = round(median(spent), 4)
    g.sla_breach_rate = round(sum(sla_breached(t, now, risk.sla, risk.agent_roles) for t in tickets) / n, 4)
    csat = [t.csat for t in tickets if t.csat is not None]
    if csat:
        g.mean_csat = round(mean(csat), 4)
    scored = [sentiments[t.id] for t in tickets if t.id in sentiments]
    if scored:
        g.mean_sentiment = round(mean(s.score for s in scored), 4)
        g.negative_share = round(sum(s.label == "negative" for s in scored) / len(scored), 4)
    g.high_risk = sum(t.id in high_risk for t in tickets)
    return g


def service_quality(
    tickets: Iterable[Ticket],
    *,
    config: QualityConfig | None = None,
    risk: RiskConfig | None = None,
    sentiments: Mapping[str, TicketSentiment] | None = None,
    assessments: Iterable[RiskAssessment] = (),
    now: datetime | None = None,
) -> QualityReport:
    """Resolution, reopen, response time, SLA, CSAT, sentiment and risk numbers, overall and per group."""
    cfg = config or QualityConfig()
    risk_cfg = risk or RiskConfig()
    rows = list(tickets)
    moment = _utc(now) if now else (latest_timestamp(rows) or datetime.now(UTC))
    scores = dict(sentiments or {})
    high = {a.ticket_id for a in assessments if a.level == "high"}
    groups: dict[str, list[Ticket]] = {}
    for t in rows:
        for key in ticket_field(t, cfg.group_by) or ["unknown"]:
            groups.setdefault(key, []).append(t)
    return QualityReport(
        group_by=cfg.group_by,
        overall=_group_quality("all", rows, moment, risk_cfg, scores, high),
        groups=[_group_quality(k, v, moment, risk_cfg, scores, high) for k, v in sorted(groups.items())],
    )
