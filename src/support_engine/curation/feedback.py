"""Human feedback on answers: ratings, corrections and the questions that were asked.

Free text (``correction`` and ``query``) is masked with the default PII rules when a
``Feedback`` is built, so nothing downstream (store, index, logs) sees raw values.
Reviewers are identified by role only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from support_engine.pii import mask_pii


def _now() -> datetime:
    return datetime.now(UTC)


class Feedback(BaseModel):
    """One human judgement of an answer (a ticket resolution or one of its chunks)."""

    id: str = ""
    ticket_id: str
    chunk_id: str | None = Field(default=None, description="answer chunk rated; None rates the ticket answer")
    rating: float = Field(ge=0.0, le=1.0, description="0 = wrong or useless, 1 = solved the problem")
    correction: str | None = Field(default=None, description="better answer written by a person")
    query: str | None = Field(default=None, description="question the answer was shown for")
    reviewer_role: str = Field(default="agent", description="agent, customer, reviewer; never a person")
    created_at: datetime = Field(default_factory=_now)

    @field_validator("correction", "query")
    @classmethod
    def _mask(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = mask_pii(value).strip()
        return value or None

    @model_validator(mode="after")
    def _default_id(self) -> Feedback:
        if not self.id:
            seed = f"{self.ticket_id}|{self.chunk_id}|{self.created_at.isoformat()}|{self.rating}|{self.correction}"
            self.id = "fb-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return self

    @classmethod
    def vote(cls, ticket_id: str, helpful: bool, **kwargs: object) -> Feedback:
        """Thumbs up or down."""
        return cls(ticket_id=ticket_id, rating=1.0 if helpful else 0.0, **kwargs)

    @classmethod
    def stars(cls, ticket_id: str, stars: int, scale: int = 5, **kwargs: object) -> Feedback:
        """Star rating from 1..``scale`` normalised to 0..1."""
        if scale < 2 or not 1 <= stars <= scale:
            raise ValueError(f"stars must be between 1 and {scale}")
        return cls(ticket_id=ticket_id, rating=(stars - 1) / (scale - 1), **kwargs)


def group_by_ticket(feedback: Iterable[Feedback]) -> dict[str, list[Feedback]]:
    """Feedback lists keyed by ticket id, in input order."""
    out: dict[str, list[Feedback]] = {}
    for item in feedback:
        out.setdefault(item.ticket_id, []).append(item)
    return out


def bayesian_rating(ratings: Iterable[float], prior_mean: float = 0.5, prior_weight: float = 2.0) -> float | None:
    """Mean rating shrunk toward ``prior_mean``; None when there are no ratings."""
    values = list(ratings)
    if not values:
        return None
    return (prior_mean * prior_weight + sum(values)) / (prior_weight + len(values))
