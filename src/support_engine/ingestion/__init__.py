"""Ingestion quality: cleanup, PII masking, chunking and deduplication of support tickets.

Typical use::

    from support_engine.ingestion import IngestionPipeline

    result = IngestionPipeline().process(tickets)
    result.tickets  # cleaned and masked tickets, safe to store
    result.chunks   # interaction and summary chunks for retrieval
    result.stats    # counters only, no ticket text
"""

from support_engine.ingestion.chunking import (
    Chunk,
    ChunkingConfig,
    Summarizer,
    chunk_ticket,
    extractive_summary,
    split_text,
)
from support_engine.ingestion.cleanup import (
    CleanupConfig,
    clean_text,
    html_to_text,
    normalize_text,
    strip_disclaimer,
    strip_quoted_replies,
    strip_signature,
)
from support_engine.ingestion.dedup import (
    DedupConfig,
    MinHasher,
    NearDuplicateIndex,
    deduplicate,
    estimate_jaccard,
    jaccard,
    shingles,
)
from support_engine.ingestion.pipeline import (
    IngestionConfig,
    IngestionPipeline,
    IngestionResult,
    IngestionStats,
    PrivacyConfig,
    ingest_tickets,
)

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "CleanupConfig",
    "DedupConfig",
    "IngestionConfig",
    "IngestionPipeline",
    "IngestionResult",
    "IngestionStats",
    "MinHasher",
    "NearDuplicateIndex",
    "PrivacyConfig",
    "Summarizer",
    "chunk_ticket",
    "clean_text",
    "deduplicate",
    "estimate_jaccard",
    "extractive_summary",
    "html_to_text",
    "ingest_tickets",
    "jaccard",
    "normalize_text",
    "shingles",
    "split_text",
    "strip_disclaimer",
    "strip_quoted_replies",
    "strip_signature",
]
