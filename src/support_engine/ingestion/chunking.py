"""Chunking of tickets for retrieval: one chunk set per interaction plus a case summary.

Long interactions are split on paragraph and sentence boundaries into pieces of at
most ``max_chars`` characters, with ``overlap_chars`` of trailing context carried
into the next piece. The case summary is extractive by default (problem from the
first customer message, resolution from the last agent message); a custom
``summarizer`` callable can replace it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from support_engine.models import Ticket

ChunkKind = Literal["interaction", "summary"]
Summarizer = Callable[[Ticket], str]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkingConfig(BaseModel):
    """Parameters for ``chunk_ticket``."""

    max_chars: int = Field(default=1200, ge=50)
    overlap_chars: int = Field(default=150, ge=0)
    min_chars: int = Field(default=20, ge=0, description="shorter interaction pieces are dropped")
    include_description: bool = True
    include_private: bool = Field(default=False, description="index internal notes too")
    include_summary: bool = True
    summary_max_chars: int = Field(default=800, ge=50)
    context_header: bool = Field(default=True, description="prefix embed_text with subject and role")

    @model_validator(mode="after")
    def _overlap_below_max(self) -> ChunkingConfig:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        return self


class Chunk(BaseModel):
    """A retrievable piece of a ticket."""

    id: str
    ticket_id: str
    kind: ChunkKind
    text: str
    context: str = ""
    position: int = Field(default=0, description="interaction index in the thread, 0 is the description")
    part: int = Field(default=0, description="piece index inside a long interaction")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text to embed: context header (if any) followed by the chunk text."""
        return f"{self.context}\n{self.text}" if self.context else self.text


def _hard_split(text: str, max_chars: int) -> list[str]:
    words, pieces, current = text.split(" "), [], ""
    for word in words:
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Split into (separator, unit) pairs, each unit at most ``max_chars`` long."""
    units: list[tuple[str, str]] = []
    for p_idx, paragraph in enumerate(p for p in text.split("\n\n") if p.strip()):
        sep = "\n\n" if p_idx else ""
        if len(paragraph) <= max_chars:
            units.append((sep, paragraph))
            continue
        for s_idx, sentence in enumerate(s for s in _SENTENCE_END.split(paragraph) if s):
            s_sep = sep if s_idx == 0 else " "
            for h_idx, piece in enumerate(_hard_split(sentence, max_chars)):
                units.append((s_sep if h_idx == 0 else " ", piece))
    return units


def _tail(text: str, size: int) -> str:
    if size <= 0 or len(text) <= size:
        return text if size > 0 else ""
    cut = text[-size:]
    space = cut.find(" ")
    return cut[space + 1 :] if 0 <= space < len(cut) - 1 else cut


def split_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:
    """Pack paragraphs and sentences into pieces of at most ``max_chars``, with word-aligned overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sep, unit in _units(text, max_chars):
        candidate = f"{current}{sep or ' '}{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current.strip())
        overlap = _tail(current, overlap_chars)
        current = f"{overlap} {unit}" if overlap and len(overlap) + 1 + len(unit) <= max_chars else unit
    if current:
        pieces.append(current.strip())
    return pieces


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rsplit(" ", 1)[0]
    return f"{cut}..."


def extractive_summary(ticket: Ticket, max_chars: int = 800) -> str:
    """Case summary built from the ticket fields, the first customer text and the last agent reply."""
    public = [c for c in sorted(ticket.comments, key=lambda c: c.created_at) if c.public and c.body.strip()]
    problem = ticket.description.strip() or next((c.body for c in public if c.author_role == "customer"), "")
    resolution = next((c.body for c in reversed(public) if c.author_role == "agent"), "")
    budget = max(40, (max_chars - len(ticket.subject) - 80) // 2)
    lines = [f"Subject: {ticket.subject}", f"Status: {ticket.status.value}"]
    if ticket.product:
        lines.append(f"Product: {ticket.product}")
    if ticket.tags:
        lines.append(f"Tags: {', '.join(ticket.tags)}")
    if problem:
        lines.append(f"Problem: {_truncate(problem, budget)}")
    if resolution:
        lines.append(f"Resolution: {_truncate(resolution, budget)}")
    summary = "\n".join(lines)
    return summary if len(summary) <= max_chars else summary[: max_chars - 3].rstrip() + "..."


def _base_metadata(ticket: Ticket) -> dict[str, Any]:
    return {
        "source": ticket.source,
        "status": ticket.status.value,
        "product": ticket.product,
        "tags": list(ticket.tags),
        "language": ticket.language,
        "subject": ticket.subject,
    }


def chunk_ticket(
    ticket: Ticket,
    config: ChunkingConfig | None = None,
    summarizer: Summarizer | None = None,
) -> list[Chunk]:
    """Turn one (already cleaned and masked) ticket into interaction chunks plus a summary chunk."""
    cfg = config or ChunkingConfig()
    base = _base_metadata(ticket)
    interactions: list[tuple[int, str, str, dict[str, Any]]] = []
    if cfg.include_description and ticket.description.strip():
        interactions.append((0, "customer", ticket.description, {"comment_id": None}))
    ordered = sorted(ticket.comments, key=lambda c: c.created_at)
    for idx, comment in enumerate(ordered, start=1):
        if not comment.public and not cfg.include_private:
            continue
        meta = {"comment_id": comment.id, "public": comment.public, "created_at": comment.created_at.isoformat()}
        interactions.append((idx, comment.author_role, comment.body, meta))

    chunks: list[Chunk] = []
    for position, role, body, meta in interactions:
        pieces = [p for p in split_text(body, cfg.max_chars, cfg.overlap_chars) if len(p) >= cfg.min_chars]
        for part, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{ticket.id}:i{position}:{part}",
                    ticket_id=ticket.id,
                    kind="interaction",
                    text=piece,
                    context=f"{ticket.subject}\n{role}:" if cfg.context_header else "",
                    position=position,
                    part=part,
                    metadata={**base, **meta, "author_role": role},
                )
            )

    if cfg.include_summary:
        summary = summarizer(ticket) if summarizer else extractive_summary(ticket, cfg.summary_max_chars)
        summary = summary.strip()
        if summary:
            chunks.append(
                Chunk(
                    id=f"{ticket.id}:summary",
                    ticket_id=ticket.id,
                    kind="summary",
                    text=summary[: cfg.summary_max_chars],
                    position=-1,
                    metadata={**base, "resolved": ticket.is_resolved},
                )
            )
    return chunks
