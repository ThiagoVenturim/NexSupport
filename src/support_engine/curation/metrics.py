"""Quality metrics of the curated knowledge base. Numbers only, no ticket text."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean, median

from pydantic import BaseModel, Field

from support_engine.curation.feedback import Feedback
from support_engine.curation.gaps import ArticleSuggestion, GapDetector
from support_engine.curation.scoring import ResolutionScore

BUCKETS: tuple[str, ...] = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")


class QualityMetrics(BaseModel):
    answers: int = 0
    answered: int = Field(default=0, description="tickets that have an agent answer")
    mean_score: float = 0.0
    median_score: float = 0.0
    mean_confidence: float = 0.0
    share_high: float = 0.0
    share_low: float = 0.0
    distribution: dict[str, int] = Field(default_factory=lambda: dict.fromkeys(BUCKETS, 0))
    signal_coverage: dict[str, float] = Field(default_factory=dict, description="share of answers with each signal")
    feedback_count: int = 0
    helpful_rate: float | None = None
    corrections: int = 0
    gaps: int = 0
    gap_clusters: int = 0
    suggested_articles: int = 0


def _bucket(score: float) -> str:
    return BUCKETS[min(len(BUCKETS) - 1, int(score * len(BUCKETS)))]


def compute_metrics(
    scores: Iterable[ResolutionScore],
    feedback: Iterable[Feedback] = (),
    gaps: GapDetector | None = None,
    suggestions: Iterable[ArticleSuggestion] | None = None,
    *,
    high_score: float = 0.7,
    low_score: float = 0.4,
    helpful_threshold: float = 0.5,
) -> QualityMetrics:
    """Aggregate scores, feedback and gaps into quality metrics."""
    all_scores = list(scores)
    scored = [s for s in all_scores if s.has_answer]
    m = QualityMetrics(answers=len(all_scores), answered=len(scored))
    if scored:
        values = [s.score for s in scored]
        m.mean_score = round(mean(values), 4)
        m.median_score = round(median(values), 4)
        m.mean_confidence = round(mean(s.confidence for s in scored), 4)
        m.share_high = round(sum(v >= high_score for v in values) / len(values), 4)
        m.share_low = round(sum(v < low_score for v in values) / len(values), 4)
        for v in values:
            m.distribution[_bucket(v)] += 1
        names = sorted({k for s in scored for k in s.components})
        m.signal_coverage = {n: round(sum(n in s.components for s in scored) / len(scored), 4) for n in names}

    items = list(feedback)
    m.feedback_count = len(items)
    if items:
        m.helpful_rate = round(sum(f.rating >= helpful_threshold for f in items) / len(items), 4)
    m.corrections = sum(1 for f in items if f.correction)
    if gaps is not None:
        m.gaps = len(gaps)
        m.gap_clusters = len(gaps.clusters)
        m.suggested_articles = len(list(suggestions) if suggestions is not None else gaps.suggestions())
    elif suggestions is not None:
        m.suggested_articles = len(list(suggestions))
    return m
