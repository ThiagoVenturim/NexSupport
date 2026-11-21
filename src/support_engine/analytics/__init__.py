"""Support analytics: multilingual sentiment, live trends, risk radar, service quality, root cause and ROI.

Typical use, after ingestion (and, for ROI, with the outcomes of the support agent)::

    from support_engine.analytics import AnalyticsPipeline
    from support_engine.ingestion import IngestionPipeline

    ingested = IngestionPipeline().process(tickets)
    report = AnalyticsPipeline().run(ingested.tickets, agent_results)
    report.sentiment     # mean score, negative share, worsening tickets, per language
    report.alerts        # topics growing faster than their baseline
    report.high_risk()   # tickets most likely to churn, escalate or breach the SLA
    report.quality       # resolution, reopen, response time, SLA, CSAT per product
    report.root_causes   # features over-represented in negative and trending tickets
    report.roi           # deflection, hours saved, cost and ROI

Every module runs offline; ``LLMSentiment`` is the only one that calls a model.
"""

from support_engine.analytics.config import (
    DEFAULT_CHURN_TERMS,
    AnalyticsConfig,
    QualityConfig,
    RiskConfig,
    RiskWeights,
    ROIConfig,
    RootCauseConfig,
    SentimentConfig,
    SLAConfig,
    TrendConfig,
)
from support_engine.analytics.pipeline import AnalyticsPipeline, AnalyticsReport, analyze
from support_engine.analytics.risk import (
    GroupQuality,
    QualityReport,
    RiskAssessment,
    RiskRadar,
    find_terms,
    first_response_hours,
    latest_timestamp,
    resolution_hours,
    service_quality,
    sla_breached,
    sla_signal,
    waiting_hours,
)
from support_engine.analytics.roi import Outcome, ROIReport, compute_roi
from support_engine.analytics.rootcause import RootCause, RootCauseAnalyzer, RootCauseReport, change_features
from support_engine.analytics.sentiment import (
    LexiconSentiment,
    LLMSentiment,
    LLMSentimentVerdict,
    SentimentAnalyzer,
    SentimentResult,
    SentimentSummary,
    TicketSentiment,
    detect_language,
    label_for,
    summarize_sentiment,
    ticket_sentiment,
)
from support_engine.analytics.trends import TopicCluster, Trend, TrendDetector, detect_trends

__all__ = [
    "DEFAULT_CHURN_TERMS",
    "AnalyticsConfig",
    "AnalyticsPipeline",
    "AnalyticsReport",
    "GroupQuality",
    "LLMSentiment",
    "LLMSentimentVerdict",
    "LexiconSentiment",
    "Outcome",
    "QualityConfig",
    "QualityReport",
    "ROIConfig",
    "ROIReport",
    "RiskAssessment",
    "RiskConfig",
    "RiskRadar",
    "RiskWeights",
    "RootCause",
    "RootCauseAnalyzer",
    "RootCauseConfig",
    "RootCauseReport",
    "SLAConfig",
    "SentimentAnalyzer",
    "SentimentConfig",
    "SentimentResult",
    "SentimentSummary",
    "TicketSentiment",
    "TopicCluster",
    "Trend",
    "TrendConfig",
    "TrendDetector",
    "analyze",
    "change_features",
    "compute_roi",
    "detect_language",
    "detect_trends",
    "find_terms",
    "first_response_hours",
    "label_for",
    "latest_timestamp",
    "resolution_hours",
    "service_quality",
    "sla_breached",
    "sla_signal",
    "summarize_sentiment",
    "ticket_sentiment",
    "waiting_hours",
]
