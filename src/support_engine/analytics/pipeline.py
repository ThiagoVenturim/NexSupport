"""One call for every analytic: sentiment, trends, risk radar, service quality, root cause and ROI."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from support_engine.analytics.config import AnalyticsConfig
from support_engine.analytics.risk import QualityReport, RiskAssessment, RiskRadar, latest_timestamp, service_quality
from support_engine.analytics.roi import ROIReport, compute_roi
from support_engine.analytics.rootcause import RootCauseAnalyzer, RootCauseReport
from support_engine.analytics.sentiment import (
    LexiconSentiment,
    LLMSentiment,
    SentimentAnalyzer,
    SentimentSummary,
    TicketSentiment,
    summarize_sentiment,
    ticket_sentiment,
)
from support_engine.analytics.trends import Trend, TrendDetector
from support_engine.models import Ticket
from support_engine.pii import Anonymizer, mask_ticket

logger = logging.getLogger(__name__)


class AnalyticsReport(BaseModel):
    """Everything the analytics computed. Numbers, ids, configured terms and masked tokens only."""

    generated_at: datetime | None = Field(default=None, description="the 'now' the report was computed for")
    tickets: int = 0
    sentiment: SentimentSummary = Field(default_factory=SentimentSummary)
    ticket_sentiment: dict[str, TicketSentiment] = Field(default_factory=dict)
    trends: list[Trend] = Field(default_factory=list)
    alerts: list[Trend] = Field(default_factory=list)
    risk: list[RiskAssessment] = Field(default_factory=list)
    quality: QualityReport | None = None
    root_causes: list[RootCauseReport] = Field(default_factory=list)
    roi: ROIReport | None = None

    def high_risk(self) -> list[RiskAssessment]:
        return [r for r in self.risk if r.level == "high"]


class AnalyticsPipeline:
    """Runs every analytic over (masked) tickets and, optionally, the outcomes of the support agent."""

    def __init__(
        self,
        config: AnalyticsConfig | None = None,
        *,
        sentiment: SentimentAnalyzer | None = None,
        llm: Any = None,
    ) -> None:
        self.config = config or AnalyticsConfig()
        cfg = self.config
        if sentiment is not None:
            self.sentiment = sentiment
        elif cfg.sentiment_backend == "llm":
            self.sentiment = LLMSentiment(llm, tier=cfg.llm_tier, config=cfg.sentiment)
        else:
            self.sentiment = LexiconSentiment(cfg.sentiment)
        self.radar = RiskRadar(cfg.risk)
        self.root_cause = RootCauseAnalyzer(cfg.root_cause)

    def run(
        self,
        tickets: Iterable[Ticket],
        agent_results: Iterable[Any] | None = None,
        *,
        now: datetime | None = None,
        trends: TrendDetector | None = None,
    ) -> AnalyticsReport:
        """Analyse ``tickets``. Pass a long lived ``trends`` detector to keep live trends across runs."""
        cfg = self.config
        rows = [_mask(t) for t in tickets] if cfg.mask_pii else list(tickets)
        moment = now or latest_timestamp(rows)
        report = AnalyticsReport(generated_at=moment, tickets=len(rows))

        sentiments = {t.id: ticket_sentiment(t, self.sentiment, cfg.sentiment) for t in rows}
        report.ticket_sentiment = sentiments
        report.sentiment = summarize_sentiment(sentiments.values())

        detector = trends if trends is not None else TrendDetector(cfg.trends)
        detector.observe_many(rows)
        report.trends = detector.report(moment)
        report.alerts = [t for t in report.trends if t.alert]

        report.risk = self.radar.radar(rows, sentiments, moment)
        report.quality = service_quality(
            rows, config=cfg.quality, risk=cfg.risk, sentiments=sentiments, assessments=report.risk, now=moment
        )

        rc = cfg.root_cause
        if rc.negative_target:
            risky = {a.ticket_id for a in report.risk if a.level == "high"}
            target = [t for t in rows if sentiments[t.id].label == "negative" or t.id in risky]
            if target:
                report.root_causes.append(self.root_cause.analyze(target, rows, name="negative_or_high_risk"))
        if rc.trend_targets:
            by_id = {t.id: t for t in rows}
            for alert in report.alerts:
                target = [by_id[i] for i in alert.ticket_ids if i in by_id]
                if target:
                    report.root_causes.append(self.root_cause.analyze(target, rows, name=f"trend:{alert.cluster_id}"))

        if agent_results is not None:
            report.roi = compute_roi(agent_results, cfg.roi)

        # Counts only: the logs never carry ticket text.
        logger.info(
            "analytics: %d tickets, %d negative, %d alerts, %d high risk",
            report.tickets,
            report.sentiment.negative,
            len(report.alerts),
            len(report.high_risk()),
        )
        return report


def _mask(ticket: Ticket) -> Ticket:
    """Mask text, personal names and `extra` values; a no-op for tickets that went through ingestion."""
    anonymizer = Anonymizer(mask_names=True)
    masked = mask_ticket(ticket, anonymizer)
    return masked.model_copy(update={"extra": anonymizer.anonymize_record(masked.extra)})


def analyze(tickets: Iterable[Ticket], agent_results: Iterable[Any] | None = None, **kwargs: Any) -> AnalyticsReport:
    """Run ``AnalyticsPipeline`` with the default configuration."""
    now = kwargs.pop("now", None)
    return AnalyticsPipeline(**kwargs).run(tickets, agent_results, now=now)
