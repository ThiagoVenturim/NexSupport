"""Persistent feedback on the same database as the tickets (``DB_URL``, SQLite by default)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Column, Engine, Float, Index, MetaData, String, Table, Text, text

from support_engine.connectors.base import parse_datetime
from support_engine.curation.feedback import Feedback
from support_engine.storage import build_engine

curation_metadata = MetaData()

feedback_table = Table(
    "answer_feedback",
    curation_metadata,
    Column("id", String(64), primary_key=True),
    Column("ticket_id", String(255), nullable=False),
    Column("chunk_id", String(255), nullable=True),
    Column("rating", Float, nullable=False),
    Column("correction", Text, nullable=True),
    Column("query", Text, nullable=True),
    Column("reviewer_role", String(32), nullable=False, server_default="agent"),
    Column("created_at", String(64), nullable=False),
    Index("idx_feedback_ticket_id", "ticket_id"),
)


class FeedbackStore:
    """Insert-or-replace store of ``Feedback`` using parameterized SQL. Text is masked by ``Feedback``."""

    def __init__(
        self,
        target: str | Path | None = None,
        *,
        engine: Engine | None = None,
        env_file: str | Path | None = ".env",
    ) -> None:
        self.engine = engine or build_engine(target, env_file=env_file)
        self._owns_engine = engine is None
        curation_metadata.create_all(self.engine, checkfirst=True)

    def __enter__(self) -> FeedbackStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_engine:
            self.engine.dispose()

    def add(self, item: Feedback) -> None:
        params: dict[str, Any] = {
            "id": item.id,
            "ticket_id": item.ticket_id,
            "chunk_id": item.chunk_id,
            "rating": float(item.rating),
            "correction": item.correction,
            "query": item.query,
            "reviewer_role": item.reviewer_role,
            "created_at": item.created_at.isoformat(),
        }
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM answer_feedback WHERE id = :id"), {"id": item.id})
            conn.execute(
                text(
                    "INSERT INTO answer_feedback (id, ticket_id, chunk_id, rating, correction, query, "
                    "reviewer_role, created_at) VALUES (:id, :ticket_id, :chunk_id, :rating, :correction, "
                    ":query, :reviewer_role, :created_at)"
                ),
                params,
            )

    def add_many(self, items: Iterable[Feedback]) -> int:
        count = 0
        for item in items:
            self.add(item)
            count += 1
        return count

    def list_feedback(self, ticket_id: str | None = None) -> list[Feedback]:
        sql = "SELECT * FROM answer_feedback"
        params: dict[str, Any] = {}
        if ticket_id is not None:
            sql += " WHERE ticket_id = :ticket_id"
            params["ticket_id"] = ticket_id
        sql += " ORDER BY created_at ASC, id ASC"
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [
            Feedback(
                id=str(r["id"]),
                ticket_id=str(r["ticket_id"]),
                chunk_id=r["chunk_id"],
                rating=float(r["rating"]),
                correction=r["correction"],
                query=r["query"],
                reviewer_role=str(r["reviewer_role"]),
                created_at=parse_datetime(r["created_at"]) or datetime.fromisoformat(str(r["created_at"])),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text("SELECT COUNT(*) FROM answer_feedback")).scalar() or 0)
