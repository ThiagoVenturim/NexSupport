"""Answer curation: resolution score, human feedback loop, knowledge gaps and quality metrics.

Typical use, after ingestion::

    from support_engine.curation import CurationPipeline, Feedback
    from support_engine.ingestion import IngestionPipeline

    ingested = IngestionPipeline().process(tickets)
    feedback = [Feedback.vote("T-1", helpful=False, correction="Reset the token, then sign in again.")]
    result = CurationPipeline().curate(ingested.tickets, ingested.chunks, feedback)
    result.chunks       # scored chunks plus correction chunks, ready to index
    result.suggestions  # articles to write for repeated gaps
    result.metrics      # numbers only
"""

from support_engine.curation.config import CurationConfig, FeedbackConfig, GapConfig, ScoreWeights, ScoringConfig
from support_engine.curation.feedback import Feedback, bayesian_rating, group_by_ticket
from support_engine.curation.gaps import ArticleSuggestion, GapCluster, GapDetector, query_tokens
from support_engine.curation.judge import Judge, JudgeVerdict, LLMJudge
from support_engine.curation.metrics import QualityMetrics, compute_metrics
from support_engine.curation.pipeline import (
    CurationPipeline,
    CurationResult,
    CurationStats,
    chunk_priority,
    curate,
)
from support_engine.curation.scoring import ResolutionScore, answer_comment, problem_text, score_ticket
from support_engine.curation.store import FeedbackStore

__all__ = [
    "ArticleSuggestion",
    "CurationConfig",
    "CurationPipeline",
    "CurationResult",
    "CurationStats",
    "Feedback",
    "FeedbackConfig",
    "FeedbackStore",
    "GapCluster",
    "GapConfig",
    "GapDetector",
    "Judge",
    "JudgeVerdict",
    "LLMJudge",
    "QualityMetrics",
    "ResolutionScore",
    "ScoreWeights",
    "ScoringConfig",
    "answer_comment",
    "bayesian_rating",
    "chunk_priority",
    "compute_metrics",
    "curate",
    "group_by_ticket",
    "problem_text",
    "query_tokens",
    "score_ticket",
]
