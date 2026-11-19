"""Ingestion pipeline: cleanup, PII masking, deduplication and chunking.

Order for every ticket, all in memory:

1. cleanup of subject, description and comment bodies (``CleanupConfig``);
2. PII masking with a fresh ``Anonymizer`` per ticket, so no cross-ticket mapping of
   real values is kept (``PrivacyConfig``);
3. removal of empty and repeated interactions inside the ticket;
4. chunking per interaction plus a case summary (``ChunkingConfig``);
5. near-duplicate removal of interaction chunks across the corpus (``DedupConfig``).

Masking runs before anything is returned, stored or handed to a summarizer, and it
is on by default. Turning it off requires ``allow_unmasked=True`` as well. Log lines
carry counts only, never ticket text or detected values.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from support_engine.ingestion.chunking import Chunk, ChunkingConfig, Summarizer, chunk_ticket
from support_engine.ingestion.cleanup import CleanupConfig, clean_text, normalize_text
from support_engine.ingestion.dedup import DedupConfig, NearDuplicateIndex
from support_engine.models import Ticket
from support_engine.pii import Anonymizer, NameDetector

logger = logging.getLogger(__name__)

DEFAULT_NAME_FIELDS: tuple[str, ...] = (
    "requester_name",
    "requester",
    "customer_name",
    "contact_name",
    "author_name",
    "assignee_name",
    "agent_name",
    "name",
)


class PrivacyConfig(BaseModel):
    """PII handling. Masking is on unless explicitly disabled with ``allow_unmasked``."""

    mask_pii: bool = True
    mask_names: bool = True
    mask_extra: bool = Field(default=True, description="also mask string values in Ticket.extra")
    name_fields: list[str] = Field(
        default_factory=lambda: list(DEFAULT_NAME_FIELDS),
        description="Ticket.extra keys whose values are personal names to mask everywhere in the ticket",
    )
    allow_unmasked: bool = Field(default=False, description="required to turn mask_pii off")

    @model_validator(mode="after")
    def _explicit_opt_out(self) -> PrivacyConfig:
        if not self.mask_pii and not self.allow_unmasked:
            raise ValueError("mask_pii=False keeps customer data in clear text; set allow_unmasked=True to confirm")
        return self


class IngestionConfig(BaseModel):
    """All ingestion parameters."""

    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    drop_empty_comments: bool = True
    dedup_comments: bool = Field(default=True, description="drop repeated interactions inside a ticket")
    dedup_chunks: Literal["none", "ticket", "corpus"] = "corpus"


class IngestionStats(BaseModel):
    """Counters of one run. Holds no ticket text or PII."""

    tickets: int = 0
    comments_in: int = 0
    comments_kept: int = 0
    comments_empty: int = 0
    comments_duplicate: int = 0
    chunks: int = 0
    chunks_duplicate: int = 0
    pii_masked: dict[str, int] = Field(default_factory=dict)
    duplicate_of: dict[str, str] = Field(default_factory=dict, description="removed chunk id -> kept chunk id")


class IngestionResult(BaseModel):
    """Cleaned and masked tickets, their chunks and the run statistics."""

    tickets: list[Ticket] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    stats: IngestionStats = Field(default_factory=IngestionStats)


class IngestionPipeline:
    """Runs cleanup, masking, deduplication and chunking over tickets."""

    def __init__(
        self,
        config: IngestionConfig | None = None,
        *,
        summarizer: Summarizer | None = None,
        name_detectors: Iterable[NameDetector] = (),
    ) -> None:
        self.config = config or IngestionConfig()
        self.summarizer = summarizer
        self.name_detectors = list(name_detectors)

    def clean_ticket(self, ticket: Ticket) -> Ticket:
        """Return a copy with cleaned subject, description and comment bodies."""
        cfg = self.config.cleanup
        comments = [c.model_copy(update={"body": clean_text(c.body, cfg)}) for c in ticket.comments]
        return ticket.model_copy(
            update={
                "subject": normalize_text(ticket.subject, normalize_unicode=cfg.normalize_unicode).replace("\n", " "),
                "description": clean_text(ticket.description, cfg),
                "comments": comments,
            }
        )

    def _known_names(self, ticket: Ticket) -> list[str]:
        names: list[str] = []
        for field in self.config.privacy.name_fields:
            value: Any = ticket.extra.get(field)
            if isinstance(value, str) and value.strip():
                names.append(value)
        return names

    def mask_ticket(self, ticket: Ticket) -> tuple[Ticket, dict[str, int]]:
        """Return a masked copy and the number of masked values per kind."""
        privacy = self.config.privacy
        if not privacy.mask_pii:
            return ticket, {}
        anon = Anonymizer(
            mask_names=privacy.mask_names,
            known_names=self._known_names(ticket) if privacy.mask_names else (),
            name_detectors=self.name_detectors if privacy.mask_names else (),
        )
        # Detect names across the whole thread first so a name seen in one message is masked in all.
        if privacy.mask_names:
            anon.anonymize_text(ticket.thread_text())
            for comment in ticket.comments:
                if not comment.public:
                    anon.anonymize_text(comment.body)
        comments = [c.model_copy(update={"body": anon.anonymize_text(c.body)}) for c in ticket.comments]
        update: dict[str, Any] = {
            "subject": anon.anonymize_text(ticket.subject),
            "description": anon.anonymize_text(ticket.description),
            "comments": comments,
        }
        if privacy.mask_extra:
            update["extra"] = anon.anonymize_record(ticket.extra)
        return ticket.model_copy(update=update), anon.counts

    def _dedup_comments(self, ticket: Ticket, stats: IngestionStats) -> Ticket:
        kept = []
        index = NearDuplicateIndex(self.config.dedup) if self.config.dedup_comments else None
        if index is not None and ticket.description.strip():
            index.add("description", ticket.description)
        for comment in ticket.comments:
            if self.config.drop_empty_comments and not comment.body.strip():
                stats.comments_empty += 1
                continue
            if index is not None and index.add(comment.id, comment.body) is not None:
                stats.comments_duplicate += 1
                continue
            kept.append(comment)
        stats.comments_kept += len(kept)
        return ticket.model_copy(update={"comments": kept})

    def process(self, tickets: Iterable[Ticket]) -> IngestionResult:
        """Run the whole pipeline and return masked tickets, chunks and statistics."""
        cfg = self.config
        stats = IngestionStats()
        out_tickets: list[Ticket] = []
        out_chunks: list[Chunk] = []
        corpus_index = NearDuplicateIndex(cfg.dedup) if cfg.dedup.enabled and cfg.dedup_chunks == "corpus" else None

        for raw in tickets:
            stats.tickets += 1
            stats.comments_in += len(raw.comments)
            ticket, counts = self.mask_ticket(self.clean_ticket(raw))
            for kind, n in counts.items():
                stats.pii_masked[kind] = stats.pii_masked.get(kind, 0) + n
            ticket = self._dedup_comments(ticket, stats)
            out_tickets.append(ticket)

            index = corpus_index
            if index is None and cfg.dedup.enabled and cfg.dedup_chunks == "ticket":
                index = NearDuplicateIndex(cfg.dedup)
            for chunk in chunk_ticket(ticket, cfg.chunking, self.summarizer):
                if index is not None and chunk.kind == "interaction":
                    original = index.add(chunk.id, chunk.text)
                    if original is not None:
                        stats.chunks_duplicate += 1
                        stats.duplicate_of[chunk.id] = original
                        continue
                out_chunks.append(chunk)

        stats.chunks = len(out_chunks)
        logger.info(
            "ingestion: tickets=%d comments_kept=%d duplicates=%d chunks=%d pii_masked=%s",
            stats.tickets,
            stats.comments_kept,
            stats.comments_duplicate + stats.chunks_duplicate,
            stats.chunks,
            dict(sorted(stats.pii_masked.items())),
        )
        return IngestionResult(tickets=out_tickets, chunks=out_chunks, stats=stats)


def ingest_tickets(tickets: Iterable[Ticket], config: IngestionConfig | None = None, **kwargs: Any) -> IngestionResult:
    """Convenience wrapper around ``IngestionPipeline(config, **kwargs).process(tickets)``."""
    return IngestionPipeline(config, **kwargs).process(tickets)
