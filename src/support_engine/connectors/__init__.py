"""Help desk connectors normalizing external tickets into ``Ticket``."""

from __future__ import annotations

from typing import Any

from support_engine.connectors.base import (
    TicketConnector,
    normalize_csat,
    normalize_priority,
    normalize_status,
    parse_datetime,
    strip_html,
)
from support_engine.connectors.files import (
    CSVConnector,
    JSONConnector,
    load_tickets_csv,
    load_tickets_file,
    load_tickets_json,
)
from support_engine.connectors.freshdesk import FreshdeskConnector
from support_engine.connectors.intercom import IntercomConnector
from support_engine.connectors.jira_sm import JiraSMConnector, extract_adf_text
from support_engine.connectors.zendesk import ZendeskConnector

CONNECTOR_REGISTRY: dict[str, type[TicketConnector]] = {
    "zendesk": ZendeskConnector,
    "freshdesk": FreshdeskConnector,
    "jira_sm": JiraSMConnector,
    "jira": JiraSMConnector,
    "intercom": IntercomConnector,
    "csv": CSVConnector,
    "json": JSONConnector,
}


def get_connector(name: str, **kwargs: Any) -> TicketConnector:
    """Instantiate a ticket connector by platform name."""
    key = name.strip().lower().replace("-", "_")
    connector_cls = CONNECTOR_REGISTRY.get(key)
    if connector_cls is None:
        supported = ", ".join(sorted(CONNECTOR_REGISTRY))
        raise ValueError(f"Unknown connector {name!r}. Supported connectors: {supported}")
    return connector_cls(**kwargs)


__all__ = [
    "CONNECTOR_REGISTRY",
    "CSVConnector",
    "FreshdeskConnector",
    "IntercomConnector",
    "JSONConnector",
    "JiraSMConnector",
    "TicketConnector",
    "ZendeskConnector",
    "extract_adf_text",
    "get_connector",
    "load_tickets_csv",
    "load_tickets_file",
    "load_tickets_json",
    "normalize_csat",
    "normalize_priority",
    "normalize_status",
    "parse_datetime",
    "strip_html",
]
