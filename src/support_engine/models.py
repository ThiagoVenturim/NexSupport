"""Common ticket model.

Every connector (Zendesk, Freshdesk, Jira Service Management, Intercom, CSV and
JSON files) maps its records to ``Ticket`` so that ingestion, retrieval and
analytics never depend on a specific help desk.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TicketStatus(StrEnum):
    NEW = "new"
    OPEN = "open"
    PENDING = "pending"
    SOLVED = "solved"
    CLOSED = "closed"


class Comment(BaseModel):
    """One interaction in a ticket thread."""

    id: str
    author_role: str = Field(description="customer, agent or system")
    body: str
    created_at: datetime
    public: bool = True


class Ticket(BaseModel):
    """A support ticket normalised across platforms."""

    id: str
    source: str = Field(description="connector name, for example zendesk or csv")
    subject: str
    description: str = ""
    status: TicketStatus = TicketStatus.OPEN
    priority: str | None = None
    product: str | None = None
    tags: list[str] = Field(default_factory=list)
    language: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    solved_at: datetime | None = None
    reopen_count: int = 0
    csat: float | None = Field(default=None, ge=0.0, le=1.0, description="satisfaction normalised to 0..1")
    comments: list[Comment] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.status in (TicketStatus.SOLVED, TicketStatus.CLOSED)

    def thread_text(self) -> str:
        """Subject, description and public comments joined in order."""
        parts = [self.subject, self.description]
        parts += [c.body for c in sorted(self.comments, key=lambda c: c.created_at) if c.public]
        return "\n\n".join(p for p in parts if p)
