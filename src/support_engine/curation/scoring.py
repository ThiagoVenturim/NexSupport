"""Resolution score of an answer from ticket signals, an optional LLM judge and human feedback.

Signals, each in 0..1:

- ``solved``: 1 for solved or closed tickets, ``pending_value`` for pending, ``unresolved_value`` otherwise;
- ``no_reopen``: ``1 - reopen_penalty * reopen_count``, floored at 0;
- ``csat``: the ticket satisfaction, when present;
- ``judge``: the judge verdict, when a judge is configured and answers;
- ``feedback``: Bayesian mean of human ratings, when there is any.

The score is the weighted mean of the signals that are present. ``confidence`` is the
share of the total weight those signals carry, so a score built from status alone is
flagged as weak.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from support_engine.curation.config import ScoringConfig
from support_engine.curation.feedback import Feedback, bayesian_rating
from support_engine.curation.judge import Judge
from support_engine.models import Comment, Ticket, TicketStatus


class ResolutionScore(BaseModel):
    """Score of the answer of one ticket. Holds numbers only, no ticket text."""

    ticket_id: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    has_answer: bool = True
    components: dict[str, float] = Field(default_factory=dict)
    feedback_count: int = 0


def answer_comment(ticket: Ticket, roles: Iterable[str] = ("agent",)) -> Comment | None:
    """The last public reply written by one of ``roles``, taken as the ticket answer."""
    allowed = set(roles)
    replies = [c for c in ticket.comments if c.public and c.author_role in allowed and c.body.strip()]
    return max(replies, key=lambda c: c.created_at) if replies else None


def problem_text(ticket: Ticket) -> str:
    """Subject plus description, or the first customer message when there is no description."""
    body = ticket.description.strip()
    if not body:
        first = next(
            (c for c in sorted(ticket.comments, key=lambda c: c.created_at) if c.author_role == "customer"), None
        )
        body = first.body.strip() if first else ""
    return f"{ticket.subject}\n\n{body}".strip()


def _solved_signal(ticket: Ticket, cfg: ScoringConfig) -> float:
    if ticket.is_resolved:
        return 1.0
    if ticket.status == TicketStatus.PENDING:
        return cfg.pending_value
    return cfg.unresolved_value


def score_ticket(
    ticket: Ticket,
    config: ScoringConfig | None = None,
    *,
    judge: Judge | None = None,
    feedback: Iterable[Feedback] = (),
) -> ResolutionScore:
    """Resolution score of the ticket answer."""
    cfg = config or ScoringConfig()
    items = [f for f in feedback if f.ticket_id == ticket.id]
    answer = answer_comment(ticket, cfg.answer_roles)
    if answer is None and cfg.require_answer:
        return ResolutionScore(ticket_id=ticket.id, score=0.0, confidence=0.0, has_answer=False,
                               feedback_count=len(items))

    w = cfg.weights
    components: dict[str, float] = {
        "solved": _solved_signal(ticket, cfg),
        "no_reopen": max(0.0, 1.0 - cfg.reopen_penalty * ticket.reopen_count),
    }
    if ticket.csat is not None:
        components["csat"] = float(ticket.csat)
    if judge is not None and answer is not None and w.judge > 0:
        verdict = judge(problem_text(ticket), answer.body)
        if verdict is not None:
            components["judge"] = min(1.0, max(0.0, float(verdict)))
    rating = bayesian_rating((f.rating for f in items), cfg.feedback_prior_mean, cfg.feedback_prior_weight)
    if rating is not None:
        components["feedback"] = rating

    weights = {name: getattr(w, name) for name in components}
    present = sum(weights.values())
    score = sum(components[n] * weights[n] for n in components) / present if present > 0 else 0.0
    return ResolutionScore(
        ticket_id=ticket.id,
        score=round(min(1.0, max(0.0, score)), 6),
        confidence=round(present / w.total(), 6),
        has_answer=answer is not None,
        components={k: round(v, 6) for k, v in components.items()},
        feedback_count=len(items),
    )
