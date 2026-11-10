"""Jira Service Management (Jira SM) ticket connector."""

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

JIRA_STATUS_MAP: dict[str, TicketStatus] = {
    "to do": TicketStatus.NEW,
    "new": TicketStatus.NEW,
    "open": TicketStatus.OPEN,
    "in progress": TicketStatus.OPEN,
    "indeterminate": TicketStatus.OPEN,
    "waiting for support": TicketStatus.OPEN,
    "waiting for customer": TicketStatus.PENDING,
    "pending": TicketStatus.PENDING,
    "on hold": TicketStatus.PENDING,
    "resolved": TicketStatus.SOLVED,
    "done": TicketStatus.SOLVED,
    "closed": TicketStatus.CLOSED,
    "canceled": TicketStatus.CLOSED,
    "cancelled": TicketStatus.CLOSED,
    "declined": TicketStatus.CLOSED,
}

JIRA_PRIORITY_MAP: dict[str, str] = {
    "lowest": "low",
    "low": "low",
    "minor": "low",
    "medium": "medium",
    "major": "high",
    "high": "high",
    "highest": "urgent",
    "critical": "urgent",
    "blocker": "urgent",
    "urgent": "urgent",
}


def extract_adf_text(node: Any) -> str:
    """Extract plain text from an Atlassian Document Format (ADF) node or string."""
    if node is None:
        return ""
    if isinstance(node, str):
        return strip_html(node)
    if isinstance(node, Mapping):
        node_type = node.get("type")
        if node_type == "text":
            return str(node.get("text") or "")
        if node_type == "hardBreak":
            return "\n"
        children = node.get("content")
        if isinstance(children, Iterable):
            parts = [extract_adf_text(child) for child in children]
            if node_type in ("paragraph", "heading", "blockquote", "codeBlock", "listItem"):
                return "".join(parts).strip() + "\n\n"
            return "".join(parts)
        if "value" in node and isinstance(node["value"], str):
            return strip_html(node["value"])
    if isinstance(node, Iterable):
        return "".join(extract_adf_text(item) for item in node)
    return str(node)


class JiraSMConnector(TicketConnector):
    """Connector for Jira Service Management and Jira Cloud REST API v3."""

    source = "jira_sm"

    def __init__(
        self,
        base_url: str = "https://example.atlassian.net",
        email: str | None = None,
        api_token: str | None = None,
        bearer_token: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.api_token = api_token
        self.bearer_token = bearer_token
        self._client = client
        self.timeout = timeout

    def _build_client(self) -> httpx.Client:
        headers: dict[str, str] = {"Accept": "application/json"}
        auth: tuple[str, str] | None = None
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.email and self.api_token:
            auth = (self.email, self.api_token)
        return httpx.Client(base_url=self.base_url, headers=headers, auth=auth, timeout=self.timeout)

    def _resolve_comment_datetime(self, raw_comment: Mapping[str, Any], fallback: datetime) -> datetime:
        created = raw_comment.get("created") or raw_comment.get("created_at")
        if isinstance(created, Mapping):
            created = created.get("iso8601") or created.get("epochMillis")
        return parse_datetime(created) or fallback

    def to_comment(
        self,
        raw_comment: Mapping[str, Any],
        reporter_account_id: str | None = None,
        fallback_time: datetime | None = None,
    ) -> Comment:
        body_raw = raw_comment.get("body")
        body = extract_adf_text(body_raw).strip()

        author = raw_comment.get("author") if isinstance(raw_comment.get("author"), Mapping) else {}
        account_id = author.get("accountId") or author.get("name") or author.get("emailAddress")
        account_type = str(author.get("accountType") or "").lower()

        if isinstance(raw_comment.get("author_role"), str):
            author_role = str(raw_comment["author_role"]).strip().lower()
        elif account_type == "app":
            author_role = "system"
        elif account_type == "customer" or (
            account_id and reporter_account_id and str(account_id) == str(reporter_account_id)
        ):
            author_role = "customer"
        else:
            author_role = "agent"

        public = True
        if "jsdPublic" in raw_comment:
            public = bool(raw_comment["jsdPublic"])
        elif "public" in raw_comment:
            public = bool(raw_comment["public"])
        elif isinstance(raw_comment.get("visibility"), Mapping):
            public = False
        else:
            properties = raw_comment.get("properties")
            if isinstance(properties, Iterable):
                for prop in properties:
                    if isinstance(prop, Mapping) and prop.get("key") == "sd.public.comment":
                        val = prop.get("value")
                        if isinstance(val, Mapping) and "internal" in val:
                            public = not bool(val["internal"])

        created_at = self._resolve_comment_datetime(raw_comment, fallback_time or datetime.now(tz=UTC))
        return Comment(
            id=str(raw_comment.get("id", "")),
            author_role=author_role,
            body=body,
            created_at=created_at,
            public=public,
        )

    def to_ticket(
        self,
        raw: Mapping[str, Any],
        comments: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Ticket:
        fields = raw.get("fields") if isinstance(raw.get("fields"), Mapping) else raw
        ticket_id = str(raw.get("key") or raw.get("issueKey") or raw.get("id") or "")

        created_raw = fields.get("created") or raw.get("createdDate") or fields.get("created_at")
        if isinstance(created_raw, Mapping):
            created_raw = created_raw.get("iso8601") or created_raw.get("epochMillis")
        created_at = parse_datetime(created_raw) or datetime.now(tz=UTC)

        updated_raw = fields.get("updated") or fields.get("updated_at")
        if isinstance(updated_raw, Mapping):
            updated_raw = updated_raw.get("iso8601") or updated_raw.get("epochMillis")
        updated_at = parse_datetime(updated_raw)

        solved_raw = fields.get("resolutiondate") or fields.get("solved_at")
        solved_at = parse_datetime(solved_raw)

        status_obj = fields.get("status") or raw.get("currentStatus")
        status_value: Any = status_obj
        if isinstance(status_obj, Mapping):
            status_value = (
                status_obj.get("status")
                or status_obj.get("name")
                or (
                    status_obj.get("statusCategory", {}).get("key")
                    if isinstance(status_obj.get("statusCategory"), Mapping)
                    else None
                )
            )
        status = normalize_status(status_value, JIRA_STATUS_MAP)
        if solved_at is None and status in (TicketStatus.SOLVED, TicketStatus.CLOSED):
            solved_at = updated_at

        priority_obj = fields.get("priority")
        if isinstance(priority_obj, Mapping):
            priority_raw = priority_obj.get("name")
        else:
            priority_raw = priority_obj

        reporter = fields.get("reporter") if isinstance(fields.get("reporter"), Mapping) else {}
        reporter_id = reporter.get("accountId") or reporter.get("name") or reporter.get("emailAddress")

        raw_comments: Iterable[Any]
        if comments is not None:
            raw_comments = comments
        elif isinstance(fields.get("comment"), Mapping):
            raw_comments = fields["comment"].get("comments", [])
        elif isinstance(raw.get("comments"), Mapping):
            raw_comments = raw["comments"].get("values", [])
        else:
            raw_comments = fields.get("comments") or []

        parsed_comments: list[Comment] = []
        for idx, item in enumerate(raw_comments):
            if isinstance(item, Comment):
                parsed_comments.append(item)
            elif isinstance(item, Mapping):
                item_dict = dict(item)
                item_dict.setdefault("id", f"{ticket_id}-c{idx + 1}")
                parsed_comments.append(
                    self.to_comment(item_dict, reporter_account_id=reporter_id, fallback_time=created_at)
                )

        labels = fields.get("labels") or fields.get("tags") or []
        tags = [str(label) for label in labels] if isinstance(labels, Iterable) and not isinstance(labels, str) else []

        components = fields.get("components")
        product: str | None = fields.get("product")
        if not product and isinstance(components, list) and components:
            first_comp = components[0]
            if isinstance(first_comp, Mapping) and first_comp.get("name"):
                product = str(first_comp["name"])

        csat = normalize_csat(fields.get("satisfaction") or fields.get("csat") or raw.get("satisfaction"))
        reopen_count = int(fields.get("reopen_count") or 0)

        extra: dict[str, Any] = {}
        if raw.get("id") is not None:
            extra["jira_id"] = raw["id"]
        if isinstance(fields.get("issuetype"), Mapping) and fields["issuetype"].get("name"):
            extra["issue_type"] = fields["issuetype"]["name"]
        if isinstance(fields.get("project"), Mapping) and fields["project"].get("key"):
            extra["project_key"] = fields["project"]["key"]
        if reporter_id:
            extra["reporter_id"] = reporter_id

        summary = fields.get("summary") or raw.get("summary") or fields.get("subject") or f"Issue {ticket_id}"
        description = extract_adf_text(fields.get("description") or raw.get("description")).strip()

        return Ticket(
            id=ticket_id,
            source=self.source,
            subject=str(summary),
            description=description,
            status=status,
            priority=normalize_priority(priority_raw, JIRA_PRIORITY_MAP),
            product=product,
            tags=tags,
            language=fields.get("language") or None,
            created_at=created_at,
            updated_at=updated_at,
            solved_at=solved_at,
            reopen_count=reopen_count,
            csat=csat,
            comments=parsed_comments,
            extra=extra,
        )

    def fetch_tickets(
        self,
        jql: str = "ORDER BY created DESC",
        max_results: int = 50,
        max_pages: int = 10,
    ) -> list[Ticket]:
        """Search Jira issues via ``/rest/api/3/search`` and convert them to ``Ticket``."""
        active_client = self._client
        owns_client = active_client is None
        if active_client is None:
            active_client = self._build_client()
        try:
            tickets: list[Ticket] = []
            start_at = 0
            for _ in range(max_pages):
                response = active_client.get(
                    "/rest/api/3/search",
                    params={
                        "jql": jql,
                        "startAt": start_at,
                        "maxResults": max_results,
                        "fields": (
                            "summary,description,status,priority,labels,components,"
                            "created,updated,resolutiondate,comment,reporter,issuetype,project"
                        ),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                issues = payload.get("issues", [])
                if not isinstance(issues, list) or not issues:
                    break
                for issue in issues:
                    if isinstance(issue, Mapping):
                        tickets.append(self.to_ticket(issue))
                start_at += len(issues)
                total = int(payload.get("total", len(issues)))
                if start_at >= total:
                    break
            return tickets
        finally:
            if owns_client:
                active_client.close()
