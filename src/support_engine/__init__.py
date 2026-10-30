"""Support Intelligence Engine.

Turns help desk tickets from any platform into a curated knowledge base and
operational analytics:

- ``support_engine.models``: the common ``Ticket`` model every connector maps to.
- ingestion and RAG (connectors, PII masking, chunking, hybrid retrieval, copilot);
- analytics (sentiment, trends, risk radar, quality, root cause, deflection).

Models, tracing and retrieval primitives come from ``faccoder-core``.
"""

from support_engine.models import Comment, Ticket, TicketStatus

__all__ = ["Comment", "Ticket", "TicketStatus"]

__version__ = "0.1.0"
