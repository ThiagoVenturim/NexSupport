"""Support Intelligence Engine.

Turns help desk tickets from any platform into a curated knowledge base and
operational analytics:

- ``support_engine.models``: the common ``Ticket`` model every connector maps to.
- ``support_engine.connectors``: Zendesk, Freshdesk, Jira Service Management,
  Intercom, CSV and JSON connectors.
- ``support_engine.storage``: pluggable SQLAlchemy storage with DB_URL support (SQLite default).
- ``support_engine.pii``: deterministic PII masking and anonymization (regex rules, names, NER hook).
- ``support_engine.ingestion``: cleanup, PII masking, chunking and near-duplicate removal.
- ingestion and RAG (connectors, PII masking, chunking, hybrid retrieval, copilot);
- analytics (sentiment, trends, risk radar, quality, root cause, deflection).

Models, tracing and retrieval primitives come from ``nexcoder-core``.
"""

from support_engine.connectors import (
    CSVConnector,
    FreshdeskConnector,
    IntercomConnector,
    JiraSMConnector,
    JSONConnector,
    TicketConnector,
    ZendeskConnector,
    get_connector,
    load_tickets_csv,
    load_tickets_file,
    load_tickets_json,
)
from support_engine.ingestion import (
    Chunk,
    ChunkingConfig,
    CleanupConfig,
    DedupConfig,
    IngestionConfig,
    IngestionPipeline,
    IngestionResult,
    PrivacyConfig,
    ingest_tickets,
)
from support_engine.models import Comment, Ticket, TicketStatus
from support_engine.pii import (
    AnonymizationRule,
    Anonymizer,
    build_ner_detector,
    detect_names,
    mask_pii,
    mask_ticket,
)
from support_engine.storage import (
    Store,
    TicketStore,
    build_engine,
    load_env_db_url,
    resolve_db_url,
)

__all__ = [
    "AnonymizationRule",
    "Anonymizer",
    "CSVConnector",
    "Chunk",
    "ChunkingConfig",
    "CleanupConfig",
    "Comment",
    "DedupConfig",
    "FreshdeskConnector",
    "IngestionConfig",
    "IngestionPipeline",
    "IngestionResult",
    "IntercomConnector",
    "JSONConnector",
    "JiraSMConnector",
    "PrivacyConfig",
    "Store",
    "Ticket",
    "TicketConnector",
    "TicketStatus",
    "TicketStore",
    "ZendeskConnector",
    "build_engine",
    "build_ner_detector",
    "detect_names",
    "get_connector",
    "ingest_tickets",
    "load_env_db_url",
    "load_tickets_csv",
    "load_tickets_file",
    "load_tickets_json",
    "mask_pii",
    "mask_ticket",
    "resolve_db_url",
]

__version__ = "0.1.0"
