"""Zendesk ticket connector."""

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

ZENDESK_STATUS_MAP: dict[str, TicketStatus] = {
    "new": TicketStatus.NEW,
    "open": TicketStatus.OPEN,
    "pending": TicketStatus.PENDING,
    "hold": TicketStatus.PENDING,
    "solved": TicketStatus.SOLVED,
    "closed": TicketStatus.CLOSED,
}


class ZendeskConnector(TicketConnector):
    """Connector for Zendesk Support API v2."""

    source = "zendesk"

    def __init__(
        self,
        base_url: str = "https://example.zendesk.com",
        email: str | None = None,
        api_token: str | None = None,
        oauth_token: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.api_token = api_token
        self.oauth_token = oauth_token
        self._client = client
        self.timeout = timeout

    def _build_client(self) -> httpx.Client:
        headers: dict[str, str] = {"Accept": "application/json"}
        auth: tuple[str, str] | None = None
        if self.oauth_token:
            headers["Authorization"] = f"Bearer {self.oauth_token}"
        elif self.email and self.api_token:
            auth = (f"{self.email}/token", self.api_token)
        return httpx.Client(base_url=self.base_url, headers=headers, auth=auth, timeout=self.timeout)

    def _resolve_author_role(
        self,
        raw_comment: Mapping[str, Any],
        requester_id: Any = None,
        submitter_id: Any = None,
    ) -> str:
        explicit = raw_comment.get("author_role")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower()
        author_id = raw_comment.get("author_id")
        if author_id is not None and requester_id is not None and str(author_id) == str(requester_id):
            return "customer"
        if author_id == -1 or str(author_id).lower() == "system":
            return "system"
        via = raw_comment.get("via")
        if isinstance(via, Mapping) and via.get("channel") == "system":
            return "system"
        if author_id is not None and submitter_id is not None and str(author_id) == str(submitter_id):
            if requester_id is not None and str(submitter_id) != str(requester_id):
                return "agent"
        return "agent"

    def to_comment(
        self,
        raw_comment: Mapping[str, Any],
        requester_id: Any = None,
        submitter_id: Any = None,
        fallback_time: datetime | None = None,
    ) -> Comment:
        body = raw_comment.get("plain_body") or raw_comment.get("body") or ""
        if not raw_comment.get("plain_body") and raw_comment.get("html_body"):
            body = strip_html(str(raw_comment.get("html_body")))
        created_at = parse_datetime(raw_comment.get("created_at")) or fallback_time or datetime.now(tz=UTC)
        return Comment(
            id=str(raw_comment.get("id", "")),
            author_role=self._resolve_author_role(raw_comment, requester_id=requester_id, submitter_id=submitter_id),
            body=str(body).strip(),
            created_at=created_at,
            public=bool(raw_comment.get("public", True)),
        )

    def to_ticket(
        self,
        raw: Mapping[str, Any],
        comments: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Ticket:
        ticket_data = raw.get("ticket") if isinstance(raw.get("ticket"), Mapping) else raw
        ticket_id = str(ticket_data.get("id", ""))
        created_at = parse_datetime(ticket_data.get("created_at")) or datetime.now(tz=UTC)
        updated_at = parse_datetime(ticket_data.get("updated_at"))
        solved_at = parse_datetime(
            ticket_data.get("solved_at")
            or (
                ticket_data.get("metric_set", {}).get("solved_at")
                if isinstance(ticket_data.get("metric_set"), Mapping)
                else None
            )
        )
        status = normalize_status(ticket_data.get("status"), ZENDESK_STATUS_MAP)
        if solved_at is None and status in (TicketStatus.SOLVED, TicketStatus.CLOSED):
            solved_at = updated_at

        reopen_count = int(
            ticket_data.get("reopen_count")
            or (
                ticket_data.get("metric_set", {}).get("reopens", 0)
                if isinstance(ticket_data.get("metric_set"), Mapping)
                else 0
            )
            or 0
        )

        raw_comments = comments if comments is not None else ticket_data.get("comments", [])
        requester_id = ticket_data.get("requester_id")
        submitter_id = ticket_data.get("submitter_id")
        parsed_comments: list[Comment] = []
        if isinstance(raw_comments, Iterable):
            for idx, item in enumerate(raw_comments):
                if isinstance(item, Comment):
                    parsed_comments.append(item)
                elif isinstance(item, Mapping):
                    item_dict = dict(item)
                    item_dict.setdefault("id", f"{ticket_id}-c{idx + 1}")
                    parsed_comments.append(
                        self.to_comment(
                            item_dict,
                            requester_id=requester_id,
                            submitter_id=submitter_id,
                            fallback_time=created_at,
                        )
                    )

        tags_raw = ticket_data.get("tags") or []
        tags = [str(t) for t in tags_raw] if isinstance(tags_raw, Iterable) and not isinstance(tags_raw, str) else []

        csat = normalize_csat(ticket_data.get("satisfaction_rating") or ticket_data.get("csat"))
        extra: dict[str, Any] = {}
        for key in ("requester_id", "assignee_id", "group_id", "organization_id", "brand_id", "custom_fields", "via"):
            if ticket_data.get(key) is not None:
                extra[key] = ticket_data[key]

        return Ticket(
            id=ticket_id,
            source=self.source,
            subject=str(ticket_data.get("subject") or ticket_data.get("raw_subject") or f"Ticket {ticket_id}"),
            description=str(ticket_data.get("description") or "").strip(),
            status=status,
            priority=normalize_priority(ticket_data.get("priority")),
            product=ticket_data.get("product") or None,
            tags=tags,
            language=ticket_data.get("locale") or ticket_data.get("language") or None,
            created_at=created_at,
            updated_at=updated_at,
            solved_at=solved_at,
            reopen_count=reopen_count,
            csat=csat,
            comments=parsed_comments,
            extra=extra,
        )

    def fetch_comments(self, ticket_id: str | int, client: httpx.Client | None = None) -> list[dict[str, Any]]:
        """Fetch all comments for a single Zendesk ticket."""
        active_client = client or self._client
        owns_client = active_client is None
        if active_client is None:
            active_client = self._build_client()
        try:
            url: str | None = f"/api/v2/tickets/{ticket_id}/comments.json"
            collected: list[dict[str, Any]] = []
            while url:
                response = active_client.get(url)
                response.raise_for_status()
                payload = response.json()
                comments = payload.get("comments", [])
                if isinstance(comments, list):
                    collected.extend(c for c in comments if isinstance(c, dict))
                url = payload.get("next_page")
            return collected
        finally:
            if owns_client:
                active_client.close()

    def fetch_tickets(self, include_comments: bool = False, max_pages: int = 10) -> list[Ticket]:
        """Fetch tickets from Zendesk Support API v2, following pagination."""
        active_client = self._client
        owns_client = active_client is None
        if active_client is None:
            active_client = self._build_client()
        try:
            url: str | None = "/api/v2/tickets.json"
            tickets: list[Ticket] = []
            pages = 0
            while url and pages < max_pages:
                pages += 1
                response = active_client.get(url)
                response.raise_for_status()
                payload = response.json()
                raw_tickets = payload.get("tickets", [])
                for raw in raw_tickets:
                    if not isinstance(raw, Mapping):
                        continue
                    comments: list[dict[str, Any]] | None = None
                    if include_comments and "comments" not in raw and raw.get("id") is not None:
                        comments = self.fetch_comments(raw["id"], client=active_client)
                    tickets.append(self.to_ticket(raw, comments=comments))
                if payload.get("meta", {}).get("has_more") and payload.get("links", {}).get("next"):
                    url = payload["links"]["next"]
                else:
                    url = payload.get("next_page")
            return tickets
        finally:
            if owns_client:
                active_client.close()
