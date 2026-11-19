"""Parameters of answer curation: score weights, feedback loop, gap detection and metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ScoreWeights(BaseModel):
    """Relative weight of each resolution signal. Missing signals are left out and the rest renormalised."""

    solved: float = Field(default=0.25, ge=0.0)
    no_reopen: float = Field(default=0.15, ge=0.0)
    csat: float = Field(default=0.2, ge=0.0)
    judge: float = Field(default=0.15, ge=0.0)
    feedback: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def _not_all_zero(self) -> ScoreWeights:
        if self.total() <= 0:
            raise ValueError("at least one score weight must be positive")
        return self

    def total(self) -> float:
        return self.solved + self.no_reopen + self.csat + self.judge + self.feedback


class ScoringConfig(BaseModel):
    """How the resolution score of an answer is computed."""

    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    unresolved_value: float = Field(default=0.0, ge=0.0, le=1.0, description="solved signal for open tickets")
    pending_value: float = Field(default=0.3, ge=0.0, le=1.0, description="solved signal for pending tickets")
    reopen_penalty: float = Field(default=0.5, ge=0.0, le=1.0, description="no_reopen signal lost per reopen")
    feedback_prior_mean: float = Field(default=0.5, ge=0.0, le=1.0)
    feedback_prior_weight: float = Field(
        default=2.0, ge=0.0, description="pseudo-votes at the prior mean; higher values need more votes to move"
    )
    require_answer: bool = Field(default=True, description="score 0 when the ticket has no public agent reply")
    answer_roles: list[str] = Field(default_factory=lambda: ["agent"])


class FeedbackConfig(BaseModel):
    """How human feedback flows back into the index."""

    apply_corrections: bool = True
    correction_score: float = Field(default=1.0, ge=0.0, le=1.0, description="score of a human-written correction")
    superseded_policy: Literal["drop", "demote", "keep"] = Field(
        default="demote", description="what happens to an answer chunk replaced by a correction"
    )
    demote_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    helpful_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="ratings at or above count as helpful")


class GapConfig(BaseModel):
    """When an unanswered question counts as a knowledge gap and when a gap asks for an article."""

    enabled: bool = True
    gap_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="best answer score below this is a gap")
    similarity: float = Field(default=0.5, gt=0.0, le=1.0, description="token Jaccard to join a gap cluster")
    min_occurrences: int = Field(default=3, ge=1, description="repeats needed before an article is suggested")
    min_token_chars: int = Field(default=3, ge=1)
    max_samples: int = Field(default=3, ge=0, description="masked sample questions kept per suggestion")
    max_keywords: int = Field(default=5, ge=1)


class CurationConfig(BaseModel):
    """All curation parameters."""

    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    gaps: GapConfig = Field(default_factory=GapConfig)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0, description="drop chunks scored below this")
    high_score: float = Field(default=0.7, ge=0.0, le=1.0, description="metrics: share of answers at or above")
    low_score: float = Field(default=0.4, ge=0.0, le=1.0, description="metrics: share of answers below")
    relevance_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="rerank: weight of retrieval relevance against curation score"
    )
