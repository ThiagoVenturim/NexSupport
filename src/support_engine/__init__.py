"""Support Intelligence Engine.

Turns help desk tickets from any platform into a curated knowledge base and
operational analytics:

- ``support_engine.models``: the common ``Ticket`` model every connector maps to.
- ``support_engine.connectors``: Zendesk, Freshdesk, Jira Service Management,
  Intercom, CSV and JSON connectors.
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
from support_engine.models import Comment, Ticket, TicketStatus

__all__ = [
    "CSVConnector",
    "Comment",
    "FreshdeskConnector",
    "IntercomConnector",
    "JSONConnector",
    "JiraSMConnector",
    "Ticket",
    "TicketConnector",
    "TicketStatus",
    "ZendeskConnector",
    "get_connector",
    "load_tickets_csv",
    "load_tickets_file",
    "load_tickets_json",
]

__version__ = "0.1.0"
