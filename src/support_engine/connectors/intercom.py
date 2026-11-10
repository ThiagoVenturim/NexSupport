"""Intercom conversation connector."""

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

INTERCOM_STATUS_MAP: dict[str, TicketStatus] = {
    "open": TicketStatus.OPEN,
    "snoozed": TicketStatus.PENDING,
    "closed": TicketStatus.CLOSED,
    "resolved": TicketStatus.SOLVED,
}


class IntercomConnector(TicketConnector):
    """Connector for Intercom Conversations REST API."""

    source = "intercom"

    def __init__(
        self,
        access_token: str | None = None,
        base_url: str = "https://api.intercom.io",
        api_version: str = "2.11",
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.access_token = access_token
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self._client = client
        self.timeout = timeout

    def _build_client(self) -> httpx.Client:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Intercom-Version": self.api_version,
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)

    def _resolve_author_role(self, author: Mapping[str, Any] | None) -> str:
        if not isinstance(author, Mapping):
            return "agent"
        author_type = str(author.get("type") or "").strip().lower()
        if author_type in ("user", "lead", "contact", "customer"):
            return "customer"
        if author_type in ("bot", "operator", "system"):
            return "system"
        return "agent"

    def to_comment(
        self,
        raw_part: Mapping[str, Any],
        fallback_time: datetime | None = None,
    ) -> Comment | None:
        body = strip_html(str(raw_part.get("body") or ""))
        if not body:
            return None
        part_type = str(raw_part.get("part_type") or "comment").strip().lower()
        public = bool(raw_part.get("public", part_type != "note"))
        author = raw_part.get("author") if isinstance(raw_part.get("author"), Mapping) else None
        author_role = (
            str(raw_part["author_role"]).strip().lower()
            if isinstance(raw_part.get("author_role"), str)
            else self._resolve_author_role(author)
        )
        created_at = parse_datetime(raw_part.get("created_at")) or fallback_time or datetime.now(tz=UTC)
        return Comment(
            id=str(raw_part.get("id", "")),
            author_role=author_role,
            body=body,
            created_at=created_at,
            public=public,
        )

    def to_ticket(
        self,
        raw: Mapping[str, Any],
        parts: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Ticket:
        ticket_id = str(raw.get("id", ""))
        created_at = parse_datetime(raw.get("created_at")) or datetime.now(tz=UTC)
        updated_at = parse_datetime(raw.get("updated_at"))

        stats = raw.get("statistics") if isinstance(raw.get("statistics"), Mapping) else {}
        solved_at = parse_datetime(raw.get("solved_at") or stats.get("last_close_at"))
        status = normalize_status(raw.get("state") or raw.get("status"), INTERCOM_STATUS_MAP)
        if solved_at is None and status in (TicketStatus.SOLVED, TicketStatus.CLOSED):
            solved_at = updated_at

        reopen_count = int(raw.get("reopen_count") or stats.get("count_reopens") or 0)

        source_obj = raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
        description = strip_html(str(source_obj.get("body") or raw.get("description") or ""))
        subject_raw = (
            raw.get("title")
            or source_obj.get("subject")
            or raw.get("subject")
            or (description.splitlines()[0][:80] if description else f"Conversation {ticket_id}")
        )
        subject = strip_html(str(subject_raw)) or f"Conversation {ticket_id}"

        raw_parts: Iterable[Any]
        if parts is not None:
            raw_parts = parts
        elif isinstance(raw.get("conversation_parts"), Mapping):
            raw_parts = raw["conversation_parts"].get("conversation_parts", [])
        else:
            raw_parts = raw.get("comments") or []

        parsed_comments: list[Comment] = []
        for idx, item in enumerate(raw_parts):
            if isinstance(item, Comment):
                parsed_comments.append(item)
            elif isinstance(item, Mapping):
                item_dict = dict(item)
                item_dict.setdefault("id", f"{ticket_id}-p{idx + 1}")
                comment = self.to_comment(item_dict, fallback_time=created_at)
                if comment is not None:
                    parsed_comments.append(comment)

        tags_field = raw.get("tags")
        tags: list[str] = []
        if isinstance(tags_field, Mapping) and isinstance(tags_field.get("tags"), Iterable):
            for tag in tags_field["tags"]:
                if isinstance(tag, Mapping) and tag.get("name"):
                    tags.append(str(tag["name"]))
                elif isinstance(tag, str):
                    tags.append(tag)
        elif isinstance(tags_field, Iterable) and not isinstance(tags_field, str):
            for tag in tags_field:
                if isinstance(tag, Mapping) and tag.get("name"):
                    tags.append(str(tag["name"]))
                else:
                    tags.append(str(tag))

        rating_obj = raw.get("conversation_rating") or raw.get("csat")
        csat = normalize_csat(rating_obj)

        custom_attrs = raw.get("custom_attributes") if isinstance(raw.get("custom_attributes"), Mapping) else {}
        product = raw.get("product") or custom_attrs.get("product")
        language = raw.get("language") or custom_attrs.get("language")

        extra: dict[str, Any] = {}
        for key in ("admin_assignee_id", "team_assignee_id", "open", "read", "custom_attributes"):
            if raw.get(key) is not None:
                extra[key] = raw[key]

        return Ticket(
            id=ticket_id,
            source=self.source,
            subject=subject,
            description=description,
            status=status,
            priority=normalize_priority(raw.get("priority")),
            product=str(product) if product else None,
            tags=tags,
            language=str(language) if language else None,
            created_at=created_at,
            updated_at=updated_at,
            solved_at=solved_at,
            reopen_count=reopen_count,
            csat=csat,
            comments=parsed_comments,
            extra=extra,
        )

    def fetch_conversation(self, conversation_id: str, client: httpx.Client | None = None) -> dict[str, Any]:
        """Fetch a single Intercom conversation with full ``conversation_parts``."""
        active_client = client or self._client
        owns_client = active_client is None
        if active_client is None:
            active_client = self._build_client()
        try:
            response = active_client.get(f"/conversations/{conversation_id}")
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        finally:
            if owns_client:
                active_client.close()

    def fetch_tickets(
        self,
        include_parts: bool = False,
        per_page: int = 20,
        max_pages: int = 10,
    ) -> list[Ticket]:
        """Fetch conversations from Intercom REST API, following cursor pagination."""
        active_client = self._client
        owns_client = active_client is None
        if active_client is None:
            active_client = self._build_client()
        try:
            tickets: list[Ticket] = []
            starting_after: str | None = None
            for _ in range(max_pages):
                params: dict[str, Any] = {"per_page": per_page}
                if starting_after:
                    params["starting_after"] = starting_after
                response = active_client.get("/conversations", params=params)
                response.raise_for_status()
                payload = response.json()
                conversations = payload.get("conversations", [])
                if not isinstance(conversations, list) or not conversations:
                    break
                for conv in conversations:
                    if not isinstance(conv, Mapping):
                        continue
                    record: Mapping[str, Any] = conv
                    if include_parts and "conversation_parts" not in conv and conv.get("id"):
                        detailed = self.fetch_conversation(str(conv["id"]), client=active_client)
                        if detailed:
                            record = detailed
                    tickets.append(self.to_ticket(record))
                next_page = payload.get("pages", {}).get("next") if isinstance(payload.get("pages"), Mapping) else None
                if isinstance(next_page, Mapping) and next_page.get("starting_after"):
                    starting_after = str(next_page["starting_after"])
                else:
                    break
            return tickets
        finally:
            if owns_client:
                active_client.close()
