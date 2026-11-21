"""Base classes and shared normalization utilities for ticket connectors."""

from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from support_engine.models import Ticket, TicketStatus

_HTML_BREAK_RE = re.compile(r"<(?:br\s*/?|/p|/div|/li|/tr)>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_COMPACT_TZ_RE = re.compile(r"([+-]\d{2})(\d{2})$")

DEFAULT_STATUS_MAP: dict[str, TicketStatus] = {
    "new": TicketStatus.NEW,
    "unassigned": TicketStatus.NEW,
    "to do": TicketStatus.NEW,
    "todo": TicketStatus.NEW,
    "open": TicketStatus.OPEN,
    "in progress": TicketStatus.OPEN,
    "in_progress": TicketStatus.OPEN,
    "active": TicketStatus.OPEN,
    "pending": TicketStatus.PENDING,
    "hold": TicketStatus.PENDING,
    "on-hold": TicketStatus.PENDING,
    "on_hold": TicketStatus.PENDING,
    "waiting": TicketStatus.PENDING,
    "waiting for customer": TicketStatus.PENDING,
    "waiting for support": TicketStatus.PENDING,
    "snoozed": TicketStatus.PENDING,
    "solved": TicketStatus.SOLVED,
    "resolved": TicketStatus.SOLVED,
    "done": TicketStatus.SOLVED,
    "closed": TicketStatus.CLOSED,
    "canceled": TicketStatus.CLOSED,
    "cancelled": TicketStatus.CLOSED,
    "declined": TicketStatus.CLOSED,
}


def strip_html(value: str | None) -> str:
    """Convert an HTML fragment into readable plain text."""
    if not value:
        return ""
    text = _HTML_BREAK_RE.sub("\n", value)
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return _MULTI_NEWLINE_RE.sub("\n\n", cleaned)


def parse_datetime(value: Any) -> datetime | None:
    """Parse ISO-8601 strings, Jira timestamps or UNIX epochs into ``datetime``."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=UTC)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return parse_datetime(int(raw))
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        raw = _COMPACT_TZ_RE.sub(r"\1:\2", raw)
        return datetime.fromisoformat(raw)
    raise ValueError(f"Unsupported datetime value: {value!r}")


def normalize_status(
    value: Any,
    mapping: Mapping[Any, TicketStatus] | None = None,
    default: TicketStatus = TicketStatus.OPEN,
) -> TicketStatus:
    """Normalize a platform status code or label into ``TicketStatus``."""
    if isinstance(value, TicketStatus):
        return value
    if value is None:
        return default
    if mapping and value in mapping:
        return mapping[value]
    key = str(value).strip().lower()
    if mapping and key in mapping:
        return mapping[key]
    return DEFAULT_STATUS_MAP.get(key, default)


def normalize_priority(
    value: Any,
    mapping: Mapping[Any, str] | None = None,
) -> str | None:
    """Normalize a priority code or label to a lowercase string."""
    if value is None or value == "":
        return None
    if mapping and value in mapping:
        return mapping[value]
    text = str(value).strip().lower()
    if not text or text in ("none", "null", "not_priority"):
        return None
    if mapping and text in mapping:
        return mapping[text]
    return text


def normalize_csat(value: Any) -> float | None:
    """Normalize satisfaction ratings across platforms to ``0.0..1.0``."""
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        for key in ("score", "rating", "value"):
            if key in value:
                return normalize_csat(value[key])
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("", "unoffered", "offered", "none", "null", "unrated"):
            return None
        if lowered in ("good", "positive", "satisfied", "happy", "yes", "true"):
            return 1.0
        if lowered in ("neutral", "ok", "okay"):
            return 0.5
        if lowered in ("bad", "negative", "unsatisfied", "unhappy", "no", "false"):
            return 0.0
        try:
            value = float(lowered)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        num = float(value)
        if 0.0 <= num <= 1.0:
            return num
        if 1.0 < num <= 5.0:
            return round((num - 1.0) / 4.0, 4)
        if 5.0 < num <= 100.0:
            return round(num / 100.0, 4)
        if -103.0 <= num <= -101.0:
            return 0.0
        if num == 100.0:
            return 0.5
        if 101.0 <= num <= 103.0:
            return 1.0
    return None


class TicketConnector(ABC):
    """Abstract base class for help desk ticket connectors."""

    source: str

    @abstractmethod
    def to_ticket(self, raw: Mapping[str, Any], **kwargs: Any) -> Ticket:
        """Convert a single platform record into a common ``Ticket``."""

    def parse_tickets(self, records: Iterable[Mapping[str, Any]]) -> list[Ticket]:
        """Convert an iterable of platform records into a list of ``Ticket`` objects."""
        return [self.to_ticket(record) for record in records]
