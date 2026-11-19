"""Parameters of support analytics: sentiment, trends, risk radar, service quality, root cause and ROI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_CHURN_TERMS: tuple[str, ...] = (
    # English
    "cancel", "cancellation", "refund", "chargeback", "lawyer", "legal action", "lawsuit", "switch provider",
    "competitor", "unacceptable", "complaint", "terminate the contract",
    # Portuguese (accents folded)
    "cancelar", "cancelamento", "reembolso", "estorno", "advogado", "processo", "procon", "reclame aqui",
    "concorrente", "inaceitavel", "rescindir", "rescisao",
    # Spanish (accents folded)
    "reembolsar", "abogado", "demanda", "competencia", "inaceptable", "rescindir el contrato", "darme de baja",
)


class SentimentConfig(BaseModel):
    """How message and ticket sentiment are computed."""

    roles: list[str] = Field(default_factory=lambda: ["customer"], description="author roles whose messages count")
    negative_threshold: float = Field(default=-0.2, ge=-1.0, le=0.0, description="score at or below is negative")
    positive_threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="score at or above is positive")
    negation_window: int = Field(default=3, ge=0, description="words after a negator whose polarity flips")
    intensifier: float = Field(default=0.5, ge=0.0, description="extra weight added by words such as 'very'")
    normalisation: float = Field(default=4.0, gt=0.0, description="alpha in score = s / sqrt(s^2 + alpha)")
    recency_weight: float = Field(
        default=1.5, ge=1.0, description="weight of the last message relative to the first in the ticket mean"
    )
    max_chars: int = Field(default=4000, ge=1, description="characters sent to an LLM analyzer per message")


class TrendConfig(BaseModel):
    """Live trends: topic clusters counted per time window and compared with their baseline."""

    window_hours: int = Field(default=24, ge=1, description="size of one counting window")
    baseline_windows: int = Field(default=7, ge=1, description="previous windows that form the baseline")
    similarity: float = Field(default=0.3, gt=0.0, le=1.0, description="token Jaccard to join a topic cluster")
    min_count: int = Field(default=3, ge=1, description="tickets in the current window needed to alert")
    z_threshold: float = Field(default=2.0, gt=0.0, description="z-score at or above is a spike")
    ratio_threshold: float = Field(default=2.0, gt=1.0, description="current / baseline mean at or above is a spike")
    min_std: float = Field(default=1.0, gt=0.0, description="floor of the baseline standard deviation")
    min_token_chars: int = Field(default=3, ge=1)
    max_keywords: int = Field(default=5, ge=1)
    use_tags: bool = Field(default=True, description="add ticket tags and product to the cluster tokens")


class SLAConfig(BaseModel):
    """Service level targets in hours, per priority."""

    first_response_hours: dict[str, float] = Field(
        default_factory=lambda: {"urgent": 1.0, "high": 4.0, "normal": 8.0, "low": 24.0}
    )
    resolution_hours: dict[str, float] = Field(
        default_factory=lambda: {"urgent": 8.0, "high": 24.0, "normal": 72.0, "low": 168.0}
    )
    default_priority: str = "normal"

    def first_response_target(self, priority: str | None) -> float:
        return self._target(self.first_response_hours, priority)

    def resolution_target(self, priority: str | None) -> float:
        return self._target(self.resolution_hours, priority)

    def _target(self, table: dict[str, float], priority: str | None) -> float:
        key = (priority or self.default_priority).lower()
        if key in table:
            return table[key]
        return table.get(self.default_priority, max(table.values()) if table else 24.0)


class RiskWeights(BaseModel):
    """Largest contribution of each signal to the risk score (0..1). Signals combine as a noisy OR."""

    negative_sentiment: float = Field(default=0.5, ge=0.0, le=1.0)
    sentiment_drop: float = Field(default=0.3, ge=0.0, le=1.0)
    churn_terms: float = Field(default=0.6, ge=0.0, le=1.0)
    reopens: float = Field(default=0.4, ge=0.0, le=1.0)
    sla_breach: float = Field(default=0.5, ge=0.0, le=1.0)
    waiting: float = Field(default=0.3, ge=0.0, le=1.0)
    priority: float = Field(default=0.2, ge=0.0, le=1.0)
    low_csat: float = Field(default=0.5, ge=0.0, le=1.0)


class RiskConfig(BaseModel):
    """Risk radar: which tickets may churn, escalate or breach their SLA."""

    weights: RiskWeights = Field(default_factory=RiskWeights)
    sla: SLAConfig = Field(default_factory=SLAConfig)
    high: float = Field(default=0.6, ge=0.0, le=1.0, description="score at or above is high risk")
    medium: float = Field(default=0.35, ge=0.0, le=1.0, description="score at or above is medium risk")
    include_resolved: bool = Field(default=False, description="also rate solved and closed tickets")
    churn_terms: list[str] = Field(default_factory=lambda: list(DEFAULT_CHURN_TERMS))
    max_reopens: int = Field(default=2, ge=1, description="reopens that give the full reopen signal")
    waiting_hours: float = Field(default=24.0, gt=0.0, description="customer wait that gives the full signal")
    sentiment_drop: float = Field(default=0.5, gt=0.0, description="first-to-last fall that gives the full signal")
    negative_score: float = Field(default=0.6, gt=0.0, le=1.0, description="negative score that gives the full signal")
    low_csat: float = Field(default=0.4, ge=0.0, le=1.0, description="CSAT at or below counts as low")
    priority_levels: dict[str, float] = Field(
        default_factory=lambda: {"urgent": 1.0, "high": 0.6, "normal": 0.0, "low": 0.0}
    )
    agent_roles: list[str] = Field(default_factory=lambda: ["agent"])

    @model_validator(mode="after")
    def _ordered(self) -> RiskConfig:
        if self.medium > self.high:
            raise ValueError("medium risk threshold must not exceed the high threshold")
        return self


class QualityConfig(BaseModel):
    """Service quality report: which ticket field groups the numbers."""

    group_by: str = Field(default="product", description="product, source, priority, language, status or an extra key")


class RootCauseConfig(BaseModel):
    """Root cause: features over-represented in a target group of tickets against a baseline."""

    min_support: int = Field(default=2, ge=1, description="target tickets that must share a feature")
    min_lift: float = Field(default=1.5, gt=0.0, description="target share / baseline share needed")
    smoothing: float = Field(default=1.0, ge=0.0, description="additive smoothing of both shares")
    max_causes: int = Field(default=10, ge=1)
    max_ticket_ids: int = Field(default=10, ge=0)
    min_token_chars: int = Field(default=3, ge=1)
    use_terms: bool = True
    use_bigrams: bool = True
    fields: list[str] = Field(
        default_factory=lambda: ["tag", "product", "priority", "source", "language"],
        description="ticket fields used as features",
    )
    extra_keys: list[str] = Field(
        default_factory=lambda: ["environment", "version", "channel", "plan"],
        description="keys of Ticket.extra used as features",
    )
    detect_changes: bool = Field(default=True, description="look for 'after the update', 'since version 2.3' and alike")
    negative_target: bool = Field(default=True, description="pipeline: analyse negative or high risk tickets")
    trend_targets: bool = Field(default=True, description="pipeline: analyse the tickets of each trend alert")


class ROIConfig(BaseModel):
    """Deflection and ROI of the support agent. Money values share one currency."""

    currency: str = "USD"
    agent_hourly_cost: float = Field(default=30.0, ge=0.0, description="loaded cost of one support agent hour")
    handle_minutes: float = Field(default=12.0, ge=0.0, description="human time to handle one question")
    review_minutes: float = Field(default=4.0, ge=0.0, description="human time to review a level 2 draft")
    llm_cost_per_call: float = Field(default=0.002, ge=0.0)
    llm_calls_per_attempt: float = Field(default=2.0, ge=0.0, description="generator plus judge")
    fixed_cost: float = Field(default=0.0, ge=0.0, description="platform cost of the period analysed")
    confirmed_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="share of level 1 answers that really resolve the question"
    )
    deflect_cached: bool = Field(default=True, description="cache hits at level 1 count as deflected")


class AnalyticsConfig(BaseModel):
    """All analytics parameters."""

    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    trends: TrendConfig = Field(default_factory=TrendConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    root_cause: RootCauseConfig = Field(default_factory=RootCauseConfig)
    roi: ROIConfig = Field(default_factory=ROIConfig)
    mask_pii: bool = Field(default=True, description="mask tickets again before analysis (a no-op after ingestion)")
    sentiment_backend: Literal["lexicon", "llm"] = "lexicon"
    llm_tier: str = "small"
