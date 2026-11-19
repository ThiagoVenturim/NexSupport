"""Pull tickets from a help desk, normalize them to the common model and mask PII before storing."""

from __future__ import annotations

import logging
from typing import Any

from support_engine.connectors.base import TicketConnector
from support_engine.ingestion import IngestionConfig, IngestionPipeline, IngestionResult
from support_engine.storage import TicketStore

logger = logging.getLogger(__name__)


def sync_tickets(
    connector: TicketConnector,
    *,
    store: TicketStore | None = None,
    config: IngestionConfig | None = None,
    **fetch_kwargs: Any,
) -> IngestionResult:
    """Fetch (following pagination), clean, mask PII and optionally save the tickets.

    ``fetch_kwargs`` go to the connector's ``fetch_tickets`` (for example ``max_pages``).
    """
    fetch = getattr(connector, "fetch_tickets", None)
    if fetch is None:
        raise TypeError(f"{type(connector).__name__} cannot fetch tickets from an API.")
    tickets = fetch(**fetch_kwargs)
    result = IngestionPipeline(config).process(tickets)
    if store is not None:
        # Already masked by the pipeline; the store masks again as a safety net.
        store.save_tickets(result.tickets, mask_pii=True)
    logger.info("sync %s: %d ticket(s) normalized", connector.source, result.stats.tickets)
    return result


__all__ = ["sync_tickets"]
