"""CSV and JSON/JSONL file connectors."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexcoder_core import resolve_inside

from support_engine.connectors.base import (
    TicketConnector,
    normalize_csat,
    normalize_priority,
    normalize_status,
    parse_datetime,
    strip_html,
)
from support_engine.models import Comment, Ticket

_KNOWN_KEYS = {
    "id",
    "ticket_id",
    "source",
    "subject",
    "title",
    "summary",
    "description",
    "body",
    "status",
    "priority",
    "product",
    "tags",
    "language",
    "created_at",
    "created",
    "updated_at",
    "updated",
    "solved_at",
    "resolved_at",
    "reopen_count",
    "reopens",
    "csat",
    "satisfaction",
    "comments",
    "extra",
}


def _resolve_file_path(path: str | Path, sandbox_root: str | Path | None = None) -> Path:
    if sandbox_root is not None:
        return resolve_inside(sandbox_root, path)
    target = Path(path)
    return resolve_inside(target.parent, target.name)


def _parse_tags(raw_tags: Any) -> list[str]:
    if raw_tags is None or raw_tags == "":
        return []
    if isinstance(raw_tags, list):
        return [str(t).strip() for t in raw_tags if str(t).strip()]
    if isinstance(raw_tags, str):
        text = raw_tags.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    return [str(t).strip() for t in loaded if str(t).strip()]
            except json.JSONDecodeError:
                pass
        sep = ";" if ";" in text and "," not in text else ","
        return [part.strip() for part in text.split(sep) if part.strip()]
    return [str(raw_tags)]


def _parse_comments(raw_comments: Any, ticket_id: str, fallback_time: datetime) -> list[Comment]:
    if not raw_comments:
        return []
    items: Any = raw_comments
    if isinstance(raw_comments, str):
        text = raw_comments.strip()
        if not text:
            return []
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return [
                Comment(
                    id=f"{ticket_id}-c1",
                    author_role="customer",
                    body=strip_html(text),
                    created_at=fallback_time,
                    public=True,
                )
            ]
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes, Mapping)):
        return []

    comments: list[Comment] = []
    for idx, item in enumerate(items):
        if isinstance(item, Comment):
            comments.append(item)
        elif isinstance(item, Mapping):
            created_at = parse_datetime(item.get("created_at") or item.get("created")) or fallback_time
            public_raw = item.get("public", True)
            if isinstance(public_raw, str):
                public = public_raw.strip().lower() not in ("false", "0", "no")
            else:
                public = bool(public_raw)
            comments.append(
                Comment(
                    id=str(item.get("id") or f"{ticket_id}-c{idx + 1}"),
                    author_role=str(item.get("author_role") or item.get("role") or "customer").strip().lower(),
                    body=strip_html(str(item.get("body") or item.get("text") or "")),
                    created_at=created_at,
                    public=public,
                )
            )
    return comments


def _record_to_ticket(raw: Mapping[str, Any], default_source: str) -> Ticket:
    ticket_id = str(raw.get("id") or raw.get("ticket_id") or "")
    source = str(raw.get("source") or default_source)
    subject = str(raw.get("subject") or raw.get("title") or raw.get("summary") or f"Ticket {ticket_id}")
    description = strip_html(str(raw.get("description") or raw.get("body") or ""))
    status = normalize_status(raw.get("status"))
    priority = normalize_priority(raw.get("priority"))
    product_raw = raw.get("product")
    product = str(product_raw).strip() if product_raw not in (None, "") else None
    language_raw = raw.get("language")
    language = str(language_raw).strip() if language_raw not in (None, "") else None

    created_at = parse_datetime(raw.get("created_at") or raw.get("created")) or datetime.now(tz=UTC)
    updated_at = parse_datetime(raw.get("updated_at") or raw.get("updated"))
    solved_at = parse_datetime(raw.get("solved_at") or raw.get("resolved_at"))

    reopen_raw = raw.get("reopen_count") if raw.get("reopen_count") not in (None, "") else raw.get("reopens", 0)
    reopen_count = int(reopen_raw or 0)
    csat = normalize_csat(raw.get("csat") if raw.get("csat") not in (None, "") else raw.get("satisfaction"))
    tags = _parse_tags(raw.get("tags"))
    comments = _parse_comments(raw.get("comments"), ticket_id=ticket_id, fallback_time=created_at)

    extra: dict[str, Any] = {}
    if isinstance(raw.get("extra"), Mapping):
        extra.update(raw["extra"])
    for key, value in raw.items():
        if key not in _KNOWN_KEYS and value not in (None, ""):
            extra[key] = value

    return Ticket(
        id=ticket_id,
        source=source,
        subject=subject,
        description=description,
        status=status,
        priority=priority,
        product=product,
        tags=tags,
        language=language,
        created_at=created_at,
        updated_at=updated_at,
        solved_at=solved_at,
        reopen_count=reopen_count,
        csat=csat,
        comments=comments,
        extra=extra,
    )


class CSVConnector(TicketConnector):
    """Connector that imports tickets from CSV files or CSV strings."""

    source = "csv"

    def __init__(self, sandbox_root: str | Path | None = None, delimiter: str = ",") -> None:
        self.sandbox_root = sandbox_root
        self.delimiter = delimiter

    def to_ticket(self, raw: Mapping[str, Any], **kwargs: Any) -> Ticket:
        return _record_to_ticket(raw, default_source=self.source)

    def load_text(self, csv_text: str) -> list[Ticket]:
        """Parse tickets from an in-memory CSV string."""
        reader = csv.DictReader(io.StringIO(csv_text), delimiter=self.delimiter)
        return [self.to_ticket(row) for row in reader if any(v for v in row.values())]

    def load_file(self, path: str | Path, sandbox_root: str | Path | None = None) -> list[Ticket]:
        """Load tickets from a CSV file on disk, enforcing ``resolve_inside`` when sandboxed."""
        resolved = _resolve_file_path(path, sandbox_root=sandbox_root or self.sandbox_root)
        text = resolved.read_text(encoding="utf-8-sig")
        return self.load_text(text)


class JSONConnector(TicketConnector):
    """Connector that imports tickets from JSON or JSON Lines (`.jsonl`) files."""

    source = "json"

    def __init__(self, sandbox_root: str | Path | None = None) -> None:
        self.sandbox_root = sandbox_root

    def to_ticket(self, raw: Mapping[str, Any], **kwargs: Any) -> Ticket:
        return _record_to_ticket(raw, default_source=self.source)

    def load_text(self, json_text: str) -> list[Ticket]:
        """Parse tickets from a JSON array, ``{"tickets": [...]}`` object, or JSONL string."""
        stripped = json_text.strip()
        if not stripped:
            return []
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            records = [json.loads(line) for line in stripped.splitlines() if line.strip()]
            return [self.to_ticket(r) for r in records if isinstance(r, Mapping)]

        if isinstance(payload, list):
            return [self.to_ticket(r) for r in payload if isinstance(r, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("tickets", "items", "data", "issues", "conversations"):
                if isinstance(payload.get(key), list):
                    return [self.to_ticket(r) for r in payload[key] if isinstance(r, Mapping)]
            return [self.to_ticket(payload)]
        raise ValueError("Unsupported JSON structure for tickets")

    def load_file(self, path: str | Path, sandbox_root: str | Path | None = None) -> list[Ticket]:
        """Load tickets from a JSON or JSONL file on disk, enforcing ``resolve_inside``."""
        resolved = _resolve_file_path(path, sandbox_root=sandbox_root or self.sandbox_root)
        text = resolved.read_text(encoding="utf-8-sig")
        return self.load_text(text)


def load_tickets_csv(path: str | Path, sandbox_root: str | Path | None = None) -> list[Ticket]:
    """Convenience helper to load tickets from a CSV file."""
    return CSVConnector(sandbox_root=sandbox_root).load_file(path)


def load_tickets_json(path: str | Path, sandbox_root: str | Path | None = None) -> list[Ticket]:
    """Convenience helper to load tickets from a JSON or JSONL file."""
    return JSONConnector(sandbox_root=sandbox_root).load_file(path)


def load_tickets_file(path: str | Path, sandbox_root: str | Path | None = None) -> list[Ticket]:
    """Load tickets from a `.csv`, `.json`, `.jsonl` or `.ndjson` file based on suffix."""
    resolved = _resolve_file_path(path, sandbox_root=sandbox_root)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return CSVConnector(sandbox_root=sandbox_root).load_file(path)
    if suffix in (".json", ".jsonl", ".ndjson"):
        return JSONConnector(sandbox_root=sandbox_root).load_file(path)
    raise ValueError(f"Unsupported ticket file extension: {suffix!r}")
