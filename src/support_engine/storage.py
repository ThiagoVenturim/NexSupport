"""Pluggable SQLAlchemy storage for support tickets and comments.

Defaults to local SQLite (zero configuration) and supports any SQLAlchemy-compatible
database URL configured via ``DB_URL`` in the environment or a local ``.env`` file
(for example ``sqlite:///...``, ``postgresql+psycopg://...``, ``mysql+pymysql://...``,
``mssql+pyodbc://...``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import URL
from sqlalchemy.pool import StaticPool

from support_engine.connectors.base import parse_datetime
from support_engine.models import Comment, Ticket, TicketStatus
from support_engine.pii import mask_ticket

DEFAULT_SQLITE_PATH = "support_engine.db"

metadata = MetaData()

tickets_table = Table(
    "tickets",
    metadata,
    Column("id", String(255), primary_key=True),
    Column("source", String(64), nullable=False),
    Column("subject", Text, nullable=False),
    Column("description", Text, nullable=False, server_default=""),
    Column("status", String(32), nullable=False, server_default="open"),
    Column("priority", String(32), nullable=True),
    Column("product", String(128), nullable=True),
    Column("tags", Text, nullable=False, server_default="[]"),
    Column("language", String(16), nullable=True),
    Column("created_at", String(64), nullable=False),
    Column("updated_at", String(64), nullable=True),
    Column("solved_at", String(64), nullable=True),
    Column("reopen_count", Integer, nullable=False, server_default="0"),
    Column("csat", Float, nullable=True),
    Column("extra", Text, nullable=False, server_default="{}"),
    Index("idx_tickets_status", "status"),
    Index("idx_tickets_source", "source"),
)

comments_table = Table(
    "comments",
    metadata,
    Column("id", String(255), primary_key=True),
    Column(
        "ticket_id",
        String(255),
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("author_role", String(32), nullable=False),
    Column("body", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("public", Integer, nullable=False, server_default="1"),
    Index("idx_comments_ticket_id", "ticket_id"),
)


def load_env_db_url(env_file: str | Path | None = ".env") -> str | None:
    """Read ``DB_URL`` from process environment or from a local ``.env`` file."""
    env_val = os.environ.get("DB_URL", "").strip()
    if env_val:
        return env_val
    if env_file is None:
        return None
    path = Path(env_file)
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == "DB_URL":
            cleaned = value.strip().strip("'").strip('"').strip()
            if cleaned:
                return cleaned
    return None


def resolve_db_url(
    target: str | Path | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> str:
    """Resolve a path, SQLAlchemy URL or ``DB_URL`` setting into a SQLAlchemy URL string."""
    raw = str(target).strip() if target is not None else ""
    if not raw:
        raw = load_env_db_url(env_file) or DEFAULT_SQLITE_PATH

    if raw == ":memory:":
        return "sqlite+pysqlite:///:memory:"

    if "://" in raw:
        if raw.startswith("postgres://"):
            return "postgresql+psycopg://" + raw[len("postgres://") :]
        return raw

    db_path = Path(raw).expanduser().resolve()
    return URL.create("sqlite+pysqlite", database=str(db_path)).render_as_string(
        hide_password=False
    )


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


def build_engine(
    target: str | Path | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> Engine:
    """Build a SQLAlchemy ``Engine`` for the given target or configured ``DB_URL``."""
    url = resolve_db_url(target, env_file=env_file)
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        if ":memory:" in url:
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = StaticPool
        engine = create_engine(url, **kwargs)
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
        return engine
    return create_engine(url, pool_pre_ping=True)


class TicketStore:
    """Database-agnostic ticket and comment store backed by SQLAlchemy Core and parameterized queries."""

    def __init__(
        self,
        target: str | Path | None = None,
        *,
        engine: Engine | None = None,
        env_file: str | Path | None = ".env",
    ) -> None:
        if engine is not None:
            self.engine = engine
            self.url = engine.url.render_as_string(hide_password=False)
            if self.engine.dialect.name == "sqlite":
                event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        else:
            self.url = resolve_db_url(target, env_file=env_file)
            self.engine = build_engine(self.url, env_file=env_file)
        metadata.create_all(self.engine, checkfirst=True)

    def __enter__(self) -> TicketStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Dispose the underlying SQLAlchemy engine."""
        self.engine.dispose()

    def clear(self) -> None:
        """Delete all records from tickets and comments."""
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM comments"))
            conn.execute(text("DELETE FROM tickets"))

    def save_ticket(self, ticket: Ticket, *, mask_pii: bool = True) -> None:
        """Insert or update a ticket and its comments using parameterized SQL.

        PII is masked before writing unless ``mask_pii=False`` is passed explicitly.
        """
        record = mask_ticket(ticket) if mask_pii else ticket

        created_str = (
            record.created_at.isoformat()
            if isinstance(record.created_at, datetime)
            else str(record.created_at)
        )
        updated_str = (
            record.updated_at.isoformat()
            if isinstance(record.updated_at, datetime)
            else (str(record.updated_at) if record.updated_at else None)
        )
        solved_str = (
            record.solved_at.isoformat()
            if isinstance(record.solved_at, datetime)
            else (str(record.solved_at) if record.solved_at else None)
        )

        params: dict[str, Any] = {
            "id": record.id,
            "source": record.source,
            "subject": record.subject,
            "description": record.description,
            "status": str(record.status),
            "priority": record.priority,
            "product": record.product,
            "tags": json.dumps(record.tags),
            "language": record.language,
            "created_at": created_str,
            "updated_at": updated_str,
            "solved_at": solved_str,
            "reopen_count": int(record.reopen_count),
            "csat": float(record.csat) if record.csat is not None else None,
            "extra": json.dumps(record.extra),
        }

        with self.engine.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM tickets WHERE id = :id"),
                {"id": record.id},
            ).scalar()

            if exists:
                conn.execute(
                    text(
                        "UPDATE tickets SET source = :source, subject = :subject, "
                        "description = :description, status = :status, priority = :priority, "
                        "product = :product, tags = :tags, language = :language, "
                        "created_at = :created_at, updated_at = :updated_at, solved_at = :solved_at, "
                        "reopen_count = :reopen_count, csat = :csat, extra = :extra "
                        "WHERE id = :id"
                    ),
                    params,
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO tickets (id, source, subject, description, status, priority, "
                        "product, tags, language, created_at, updated_at, solved_at, reopen_count, "
                        "csat, extra) VALUES (:id, :source, :subject, :description, :status, :priority, "
                        ":product, :tags, :language, :created_at, :updated_at, :solved_at, :reopen_count, "
                        ":csat, :extra)"
                    ),
                    params,
                )

            # Replace comments
            conn.execute(
                text("DELETE FROM comments WHERE ticket_id = :ticket_id"),
                {"ticket_id": record.id},
            )
            for comment in record.comments:
                c_created = (
                    comment.created_at.isoformat()
                    if isinstance(comment.created_at, datetime)
                    else str(comment.created_at)
                )
                conn.execute(
                    text(
                        "INSERT INTO comments (id, ticket_id, author_role, body, created_at, public) "
                        "VALUES (:id, :ticket_id, :author_role, :body, :created_at, :public)"
                    ),
                    {
                        "id": comment.id,
                        "ticket_id": record.id,
                        "author_role": comment.author_role,
                        "body": comment.body,
                        "created_at": c_created,
                        "public": 1 if comment.public else 0,
                    },
                )

    def save_tickets(self, tickets: Iterable[Ticket], *, mask_pii: bool = True) -> int:
        """Save an iterable of tickets, returning the number of tickets processed."""
        count = 0
        for ticket in tickets:
            self.save_ticket(ticket, mask_pii=mask_pii)
            count += 1
        return count

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        """Retrieve a ticket and its comments by ID."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM tickets WHERE id = :id"),
                {"id": ticket_id},
            ).mappings().first()

            if not row:
                return None

            c_rows = conn.execute(
                text("SELECT * FROM comments WHERE ticket_id = :ticket_id ORDER BY created_at ASC"),
                {"ticket_id": ticket_id},
            ).mappings().all()

            comments = [
                Comment(
                    id=str(cr["id"]),
                    author_role=str(cr["author_role"]),
                    body=str(cr["body"]),
                    created_at=parse_datetime(cr["created_at"]) or datetime.now(),
                    public=bool(cr["public"]),
                )
                for cr in c_rows
            ]

            tags_raw = row["tags"]
            tags = json.loads(tags_raw) if tags_raw else []

            extra_raw = row["extra"]
            extra = json.loads(extra_raw) if extra_raw else {}

            raw_status = row["status"]
            status_val = (
                TicketStatus(raw_status)
                if raw_status in TicketStatus.__members__.values()
                else TicketStatus.OPEN
            )

            return Ticket(
                id=str(row["id"]),
                source=str(row["source"]),
                subject=str(row["subject"]),
                description=str(row["description"] or ""),
                status=status_val,
                priority=str(row["priority"]) if row["priority"] is not None else None,
                product=str(row["product"]) if row["product"] is not None else None,
                tags=tags,
                language=str(row["language"]) if row["language"] is not None else None,
                created_at=parse_datetime(row["created_at"]) or datetime.now(),
                updated_at=parse_datetime(row["updated_at"]),
                solved_at=parse_datetime(row["solved_at"]),
                reopen_count=int(row["reopen_count"] or 0),
                csat=float(row["csat"]) if row["csat"] is not None else None,
                comments=comments,
                extra=extra,
            )

    def list_tickets(
        self,
        status: TicketStatus | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Ticket]:
        """List tickets optionally filtered by status with pagination."""
        with self.engine.connect() as conn:
            if status is not None:
                status_str = str(status.value if isinstance(status, TicketStatus) else status)
                rows = conn.execute(
                    text(
                        "SELECT id FROM tickets WHERE status = :status "
                        "ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
                    ),
                    {"status": status_str, "limit": limit, "offset": offset},
                ).mappings().all()
            else:
                rows = conn.execute(
                    text(
                        "SELECT id FROM tickets "
                        "ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
                    ),
                    {"limit": limit, "offset": offset},
                ).mappings().all()

        tickets: list[Ticket] = []
        for r in rows:
            t = self.get_ticket(str(r["id"]))
            if t is not None:
                tickets.append(t)
        return tickets

    def delete_ticket(self, ticket_id: str) -> bool:
        """Delete a ticket and its associated comments."""
        with self.engine.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM tickets WHERE id = :id"),
                {"id": ticket_id},
            ).scalar()
            if not exists:
                return False
            conn.execute(
                text("DELETE FROM comments WHERE ticket_id = :ticket_id"),
                {"ticket_id": ticket_id},
            )
            conn.execute(
                text("DELETE FROM tickets WHERE id = :id"),
                {"id": ticket_id},
            )
            return True

    def count_tickets(self, status: TicketStatus | str | None = None) -> int:
        """Count tickets, optionally filtered by status."""
        with self.engine.connect() as conn:
            if status is not None:
                status_str = str(status.value if isinstance(status, TicketStatus) else status)
                val = conn.execute(
                    text("SELECT COUNT(*) FROM tickets WHERE status = :status"),
                    {"status": status_str},
                ).scalar()
            else:
                val = conn.execute(text("SELECT COUNT(*) FROM tickets")).scalar()
            return int(val or 0)


Store = TicketStore
