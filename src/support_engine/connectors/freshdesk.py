"""Freshdesk ticket connector."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from support_engine.connectors.base import (
    TicketConnector,
    normalize_csat,
    normalize_priority,
    normalize_status,
    parse_datetime,
    strip_html,
)
from support_engine.models import Comment, Ticket, TicketStatus

FRESHDESK_STATUS_MAP: dict[Any, TicketStatus] = {
    2: TicketStatus.OPEN,
    "2": TicketStatus.OPEN,
    3: TicketStatus.PENDING,
    "3": TicketStatus.PENDING,
    4: TicketStatus.SOLVED,
    "4": TicketStatus.SOLVED,
    5: TicketStatus.CLOSED,
    "5": TicketStatus.CLOSED,
    6: TicketStatus.PENDING,
    "6": TicketStatus.PENDING,
    7: TicketStatus.PENDING,
    "7": TicketStatus.PENDING,
    "new": TicketStatus.NEW,
    "open": TicketStatus.OPEN,
    "pending": TicketStatus.PENDING,
    "resolved": TicketStatus.SOLVED,
    "solved": TicketStatus.SOLVED,
    "closed": TicketStatus.CLOSED,
}

FRESHDESK_PRIORITY_MAP: dict[Any, str] = {
    1: "low",
    "1": "low",
    2: "medium",
    "2": "medium",
    3: "high",
    "3": "high",
    4: "urgent",
    "4": "urgent",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "urgent": "urgent",
}


class FreshdeskConnector(TicketConnector):
    """Connector for Freshdesk API v2."""

    source = "freshdesk"

    def __init__(
        self,
        base_url: str = "https://example.freshdesk.com",
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = client
        self.timeout = timeout

    def _build_client(self) -> httpx.Client:
        auth = (self.api_key, "X") if self.api_key else None
        return httpx.Client(
            base_url=self.base_url,
            headers={"Accept": "application/json"},
            auth=auth,
            timeout=self.timeout,
        )

    def to_comment(
        self,
        raw_conv: Mapping[str, Any],
        requester_id: Any = None,
        fallback_time: datetime | None = None,
    ) -> Comment:
        body = raw_conv.get("body_text")
        if not body:
            body = strip_html(str(raw_conv.get("body") or ""))
        incoming = raw_conv.get("incoming")
        user_id = raw_conv.get("user_id")
        if isinstance(raw_conv.get("author_role"), str):
            author_role = str(raw_conv["author_role"]).strip().lower()
        elif incoming is True or (
            user_id is not None and requester_id is not None and str(user_id) == str(requester_id)
        ):
            author_role = "customer"
        elif raw_conv.get("source") == 5:
            author_role = "system"
        else:
            author_role = "agent"

        private = bool(raw_conv.get("private", False))
        public = bool(raw_conv.get("public", not private))
        created_at = parse_datetime(raw_conv.get("created_at")) or fallback_time or datetime.now(tz=UTC)
        return Comment(
            id=str(raw_conv.get("id", "")),
            author_role=author_role,
            body=str(body).strip(),
            created_at=created_at,
            public=public,
        )

    def to_ticket(
        self,
        raw: Mapping[str, Any],
        conversations: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Ticket:
        ticket_id = str(raw.get("id", ""))
        created_at = parse_datetime(raw.get("created_at")) or datetime.now(tz=UTC)
        updated_at = parse_datetime(raw.get("updated_at"))
        stats = raw.get("stats") if isinstance(raw.get("stats"), Mapping) else {}
        solved_at = parse_datetime(
            raw.get("solved_at") or stats.get("resolved_at") or stats.get("closed_at")
        )
        status = normalize_status(raw.get("status"), FRESHDESK_STATUS_MAP)
        if solved_at is None and status in (TicketStatus.SOLVED, TicketStatus.CLOSED):
            solved_at = updated_at

        description = raw.get("description_text")
        if not description:
            description = strip_html(str(raw.get("description") or ""))

        reopen_count = int(raw.get("reopen_count") or stats.get("reopened_count") or 0)

        requester_id = raw.get("requester_id")
        raw_convs = (
            conversations
            if conversations is not None
            else (raw.get("conversations") or raw.get("comments") or [])
        )
        parsed_comments: list[Comment] = []
        if isinstance(raw_convs, Iterable):
            for idx, conv in enumerate(raw_convs):
                if isinstance(conv, Comment):
                    parsed_comments.append(conv)
                elif isinstance(conv, Mapping):
                    conv_dict = dict(conv)
                    conv_dict.setdefault("id", f"{ticket_id}-c{idx + 1}")
                    parsed_comments.append(
                        self.to_comment(conv_dict, requester_id=requester_id, fallback_time=created_at)
                    )

        tags_raw = raw.get("tags") or []
        tags = [str(t) for t in tags_raw] if isinstance(tags_raw, Iterable) and not isinstance(tags_raw, str) else []

        csat_raw = raw.get("satisfaction_rating") or raw.get("survey_result") or raw.get("csat")
        if isinstance(csat_raw, Mapping) and "ratings" in csat_raw and isinstance(csat_raw["ratings"], Mapping):
            ratings_values = list(csat_raw["ratings"].values())
            csat = normalize_csat(ratings_values[0]) if ratings_values else None
        else:
            csat = normalize_csat(csat_raw)

        extra: dict[str, Any] = {}
        for key in ("requester_id", "responder_id", "group_id", "company_id", "product_id", "type", "custom_fields"):
            if raw.get(key) is not None:
                extra[key] = raw[key]

        product = raw.get("product") or (str(raw["product_id"]) if raw.get("product_id") is not None else None)

        return Ticket(
            id=ticket_id,
            source=self.source,
            subject=str(raw.get("subject") or f"Ticket {ticket_id}"),
            description=str(description).strip(),
            status=status,
            priority=normalize_priority(raw.get("priority"), FRESHDESK_PRIORITY_MAP),
            product=product,
            tags=tags,
            language=raw.get("language") or None,
            created_at=created_at,
            updated_at=updated_at,
            solved_at=solved_at,
            reopen_count=reopen_count,
            csat=csat,
            comments=parsed_comments,
            extra=extra,
        )

    def fetch_conversations(self, ticket_id: str | int, client: httpx.Client | None = None) -> list[dict[str, Any]]:
        """Fetch conversations for a single Freshdesk ticket."""
        active_client = client or self._client
        owns_client = active_client is None
        if active_client is None:
            active_client = self._build_client()
        try:
            response = active_client.get(f"/api/v2/tickets/{ticket_id}/conversations")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return [c for c in payload if isinstance(c, dict)]
            return []
        finally:
            if owns_client:
                active_client.close()

    def fetch_tickets(
        self,
        include_conversations: bool = False,
        per_page: int = 30,
        max_pages: int = 10,
    ) -> list[Ticket]:
        """Fetch tickets from Freshdesk API v2."""
        active_client = self._client
        owns_client = active_client is None
        if active_client is None:
            active_client = self._build_client()
        try:
            tickets: list[Ticket] = []
            for page in range(1, max_pages + 1):
                response = active_client.get(
                    "/api/v2/tickets",
                    params={"per_page": per_page, "page": page, "include": "stats,description"},
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or not payload:
                    break
                for raw in payload:
                    if not isinstance(raw, Mapping):
                        continue
                    conversations: list[dict[str, Any]] | None = None
                    if include_conversations and "conversations" not in raw and raw.get("id") is not None:
                        conversations = self.fetch_conversations(raw["id"], client=active_client)
                    tickets.append(self.to_ticket(raw, conversations=conversations))
                if len(payload) < per_page and "link" not in response.headers:
                    break
            return tickets
        finally:
            if owns_client:
                active_client.close()
