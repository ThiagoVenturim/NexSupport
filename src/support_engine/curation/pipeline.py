"""Curation pipeline: score answers, fold human feedback back into the index, track gaps.

Runs on the output of ``support_engine.ingestion`` (masked tickets and their chunks):

1. every ticket gets a ``ResolutionScore`` (status, reopens, CSAT, optional judge, feedback);
2. every chunk of the ticket gets ``resolution_score``, ``score_confidence`` and
   ``curation_priority`` in its metadata; a chunk rated on its own blends that rating in;
3. human corrections become ``correction`` chunks and the answer they replace is
   demoted, dropped or kept (``FeedbackConfig.superseded_policy``);
4. chunks below ``min_score`` are dropped when it is set;
5. unhelpful ratings without a correction and weak query results feed the ``GapDetector``,
   whose repeated clusters become article suggestions;
6. ``QualityMetrics`` summarise the run.

Log lines carry counts only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from support_engine.curation.config import CurationConfig
from support_engine.curation.feedback import Feedback, bayesian_rating, group_by_ticket
from support_engine.curation.gaps import ArticleSuggestion, GapDetector
from support_engine.curation.judge import Judge
from support_engine.curation.metrics import QualityMetrics, compute_metrics
from support_engine.curation.scoring import ResolutionScore, answer_comment, score_ticket
from support_engine.ingestion.chunking import Chunk
from support_engine.models import Ticket

logger = logging.getLogger(__name__)

NEUTRAL_PRIORITY = 0.5


class CurationStats(BaseModel):
    """Counters of one run. Holds no ticket text."""

    tickets: int = 0
    chunks_in: int = 0
    chunks_out: int = 0
    chunks_scored: int = 0
    chunks_below_min: int = 0
    chunks_superseded: int = 0
    corrections_added: int = 0
    feedback_used: int = 0
    feedback_unknown_ticket: int = 0


class CurationResult(BaseModel):
    chunks: list[Chunk] = Field(default_factory=list)
    scores: dict[str, ResolutionScore] = Field(default_factory=dict)
    suggestions: list[ArticleSuggestion] = Field(default_factory=list)
    metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    stats: CurationStats = Field(default_factory=CurationStats)


def chunk_priority(chunk: Chunk) -> float:
    """Curation priority of a chunk, neutral when it was never scored."""
    value: Any = chunk.metadata.get("curation_priority", chunk.metadata.get("resolution_score"))
    return float(value) if isinstance(value, int | float) else NEUTRAL_PRIORITY


class CurationPipeline:
    """Applies resolution scores and the feedback loop to ingested chunks."""

    def __init__(
        self,
        config: CurationConfig | None = None,
        *,
        judge: Judge | None = None,
        gaps: GapDetector | None = None,
    ) -> None:
        self.config = config or CurationConfig()
        self.judge = judge
        self.gaps = gaps or GapDetector(self.config.gaps)

    # Scoring -------------------------------------------------------------------------------

    def score(self, tickets: Iterable[Ticket], feedback: Iterable[Feedback] = ()) -> dict[str, ResolutionScore]:
        """Resolution score per ticket id."""
        by_ticket = group_by_ticket(feedback)
        return {
            t.id: score_ticket(t, self.config.scoring, judge=self.judge, feedback=by_ticket.get(t.id, ()))
            for t in tickets
        }

    def _chunk_score(self, base: ResolutionScore, ratings: list[float]) -> float:
        if not ratings:
            return base.score
        cfg = self.config.scoring
        rating = bayesian_rating(ratings, cfg.feedback_prior_mean, cfg.feedback_prior_weight)
        share = cfg.weights.feedback / cfg.weights.total()
        return round((1 - share) * base.score + share * (rating or 0.0), 6)

    # Feedback loop -------------------------------------------------------------------------

    def _superseded_ids(self, ticket: Ticket | None, chunks: Sequence[Chunk], item: Feedback) -> set[str]:
        if item.chunk_id:
            return {item.chunk_id}
        ids = {c.id for c in chunks if c.ticket_id == item.ticket_id and c.kind == "summary"}
        answer = answer_comment(ticket, self.config.scoring.answer_roles) if ticket else None
        if answer is not None:
            ids |= {
                c.id
                for c in chunks
                if c.ticket_id == item.ticket_id
                and c.kind == "interaction"
                and c.metadata.get("comment_id") == answer.id
            }
        return ids

    def correction_chunk(self, item: Feedback, ticket: Ticket | None, corrects: Iterable[str]) -> Chunk:
        """Index entry for a human correction (already masked by ``Feedback``)."""
        subject = ticket.subject if ticket else ""
        metadata: dict[str, Any] = {
            "source": ticket.source if ticket else None,
            "product": ticket.product if ticket else None,
            "tags": list(ticket.tags) if ticket else [],
            "language": ticket.language if ticket else None,
            "subject": subject,
            "author_role": item.reviewer_role,
            "feedback_id": item.id,
            "corrects": sorted(corrects),
            "created_at": item.created_at.isoformat(),
            "resolution_score": self.config.feedback.correction_score,
            "score_confidence": 1.0,
            "curation_priority": self.config.feedback.correction_score,
        }
        return Chunk(
            id=f"{item.ticket_id}:correction:{item.id}",
            ticket_id=item.ticket_id,
            kind="correction",
            text=item.correction or "",
            context=f"{subject}\ncorrection:" if subject else "",
            position=-2,
            metadata=metadata,
        )

    # Whole run -----------------------------------------------------------------------------

    def curate(
        self,
        tickets: Iterable[Ticket],
        chunks: Iterable[Chunk],
        feedback: Iterable[Feedback] = (),
    ) -> CurationResult:
        """Score, apply feedback, filter and measure. Inputs are not modified."""
        cfg = self.config
        ticket_list = list(tickets)
        tickets_by_id = {t.id: t for t in ticket_list}
        items = list(feedback)
        known = [f for f in items if f.ticket_id in tickets_by_id]
        stats = CurationStats(
            tickets=len(ticket_list), feedback_used=len(known), feedback_unknown_ticket=len(items) - len(known)
        )
        chunk_list = list(chunks)
        stats.chunks_in = len(chunk_list)
        scores = self.score(ticket_list, known)

        chunk_ratings: dict[str, list[float]] = {}
        for item in known:
            if item.chunk_id:
                chunk_ratings.setdefault(item.chunk_id, []).append(item.rating)

        superseded: dict[str, list[str]] = {}
        corrections: list[Chunk] = []
        if cfg.feedback.apply_corrections:
            for item in known:
                if not item.correction:
                    continue
                ticket = tickets_by_id.get(item.ticket_id)
                replaced = self._superseded_ids(ticket, chunk_list, item)
                correction = self.correction_chunk(item, ticket, replaced)
                corrections.append(correction)
                for cid in replaced:
                    superseded.setdefault(cid, []).append(correction.id)

        out: list[Chunk] = []
        for chunk in chunk_list:
            base = scores.get(chunk.ticket_id)
            metadata = dict(chunk.metadata)
            if base is not None:
                value = self._chunk_score(base, chunk_ratings.get(chunk.id, []))
                metadata.update(resolution_score=value, score_confidence=base.confidence, curation_priority=value)
                stats.chunks_scored += 1
            if chunk.id in superseded:
                policy = cfg.feedback.superseded_policy
                stats.chunks_superseded += 1
                if policy == "drop":
                    continue
                metadata["superseded_by"] = superseded[chunk.id]
                if policy == "demote":
                    current = float(metadata.get("curation_priority", NEUTRAL_PRIORITY))
                    metadata["curation_priority"] = round(current * cfg.feedback.demote_factor, 6)
            curated = chunk.model_copy(update={"metadata": metadata})
            if cfg.min_score is not None and chunk_priority(curated) < cfg.min_score:
                stats.chunks_below_min += 1
                continue
            out.append(curated)

        stats.corrections_added = len(corrections)
        out.extend(corrections)
        stats.chunks_out = len(out)

        self.gaps.observe_feedback(known, cfg.feedback.helpful_threshold)
        suggestions = self.gaps.suggestions()
        metrics = compute_metrics(
            scores.values(),
            known,
            self.gaps,
            suggestions,
            high_score=cfg.high_score,
            low_score=cfg.low_score,
            helpful_threshold=cfg.feedback.helpful_threshold,
        )
        logger.info(
            "curation: tickets=%d chunks_out=%d below_min=%d superseded=%d corrections=%d gaps=%d suggestions=%d",
            stats.tickets,
            stats.chunks_out,
            stats.chunks_below_min,
            stats.chunks_superseded,
            stats.corrections_added,
            metrics.gaps,
            metrics.suggested_articles,
        )
        return CurationResult(chunks=out, scores=scores, suggestions=suggestions, metrics=metrics, stats=stats)

    # Retrieval helpers ---------------------------------------------------------------------

    def rerank(self, hits: Iterable[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        """Blend retrieval relevance (0..1) with curation priority; best first."""
        rw = self.config.relevance_weight
        blended = [(chunk, round(rw * rel + (1 - rw) * chunk_priority(chunk), 6)) for chunk, rel in hits]
        return sorted(blended, key=lambda pair: pair[1], reverse=True)

    def observe_query(self, query: str, hits: Iterable[Chunk], *, ticket_id: str | None = None) -> bool:
        """Record ``query`` as a gap when none of the relevant ``hits`` is a good answer. Returns True if recorded."""
        best = max((chunk_priority(c) for c in hits), default=None)
        return self.gaps.observe(query, best, ticket_id=ticket_id) is not None


def curate(
    tickets: Iterable[Ticket],
    chunks: Iterable[Chunk],
    feedback: Iterable[Feedback] = (),
    config: CurationConfig | None = None,
    **kwargs: Any,
) -> CurationResult:
    """Convenience wrapper around ``CurationPipeline(config, **kwargs).curate(...)``."""
    return CurationPipeline(config, **kwargs).curate(tickets, chunks, feedback)
