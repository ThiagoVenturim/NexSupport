"""Tests for pluggable SQLAlchemy storage and parameterized queries."""

import os
from datetime import UTC, datetime
from pathlib import Path

from support_engine.models import Comment, Ticket, TicketStatus
from support_engine.storage import (
    TicketStore,
    build_engine,
    load_env_db_url,
    resolve_db_url,
)


def _make_ticket(
    ticket_id: str = "T-100",
    subject: str = "Login failure",
    status: TicketStatus = TicketStatus.OPEN,
    csat: float | None = 0.75,
) -> Ticket:
    return Ticket(
        id=ticket_id,
        source="test",
        subject=subject,
        description="User cannot login on web dashboard.",
        status=status,
        priority="high",
        product="core-app",
        tags=["auth", "web"],
        language="en",
        created_at=datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 4, 1, 12, 30, 0, tzinfo=UTC),
        solved_at=None,
        reopen_count=1,
        csat=csat,
        comments=[
            Comment(
                id=f"{ticket_id}-C1",
                author_role="customer",
                body="I see error code 401.",
                created_at=datetime(2026, 4, 1, 12, 5, 0, tzinfo=UTC),
                public=True,
            ),
            Comment(
                id=f"{ticket_id}-C2",
                author_role="agent",
                body="Investigating the auth logs.",
                created_at=datetime(2026, 4, 1, 12, 15, 0, tzinfo=UTC),
                public=False,
            ),
        ],
        extra={"ip": "127.0.0.1"},
    )


def test_in_memory_sqlite_save_and_get() -> None:
    with TicketStore(":memory:") as store:
        ticket = _make_ticket()
        store.save_ticket(ticket)

        retrieved = store.get_ticket("T-100")
        assert retrieved is not None
        assert retrieved.id == "T-100"
        assert retrieved.subject == "Login failure"
        assert retrieved.status == TicketStatus.OPEN
        assert retrieved.priority == "high"
        assert retrieved.product == "core-app"
        assert retrieved.tags == ["auth", "web"]
        assert retrieved.reopen_count == 1
        assert retrieved.csat == 0.75
        assert len(retrieved.comments) == 2
        assert retrieved.comments[0].id == "T-100-C1"
        assert retrieved.comments[0].public is True
        assert retrieved.comments[1].id == "T-100-C2"
        assert retrieved.comments[1].public is False


def test_upsert_ticket_and_replace_comments() -> None:
    with TicketStore(":memory:") as store:
        ticket = _make_ticket()
        store.save_ticket(ticket)

        updated_ticket = ticket.model_copy(
            update={
                "subject": "Login failure - resolved",
                "status": TicketStatus.SOLVED,
                "solved_at": datetime(2026, 4, 1, 13, 0, 0, tzinfo=UTC),
                "comments": [
                    Comment(
                        id="T-100-C3",
                        author_role="agent",
                        body="Password reset sent.",
                        created_at=datetime(2026, 4, 1, 12, 45, 0, tzinfo=UTC),
                        public=True,
                    )
                ],
            }
        )
        store.save_ticket(updated_ticket)

        retrieved = store.get_ticket("T-100")
        assert retrieved is not None
        assert retrieved.subject == "Login failure - resolved"
        assert retrieved.status == TicketStatus.SOLVED
        assert retrieved.solved_at is not None
        assert len(retrieved.comments) == 1
        assert retrieved.comments[0].id == "T-100-C3"


def test_delete_ticket_and_cascade_comments() -> None:
    with TicketStore(":memory:") as store:
        store.save_ticket(_make_ticket("T-DEL"))
        assert store.get_ticket("T-DEL") is not None
        assert store.count_tickets() == 1

        deleted = store.delete_ticket("T-DEL")
        assert deleted is True
        assert store.get_ticket("T-DEL") is None
        assert store.count_tickets() == 0

        # Deleting nonexistent returns False
        assert store.delete_ticket("T-NOTFOUND") is False


def test_list_and_count_tickets() -> None:
    with TicketStore(":memory:") as store:
        store.save_ticket(_make_ticket("T-1", status=TicketStatus.OPEN))
        store.save_ticket(_make_ticket("T-2", status=TicketStatus.OPEN))
        store.save_ticket(_make_ticket("T-3", status=TicketStatus.SOLVED))

        assert store.count_tickets() == 3
        assert store.count_tickets(TicketStatus.OPEN) == 2
        assert store.count_tickets(TicketStatus.SOLVED) == 1
        assert store.count_tickets(TicketStatus.CLOSED) == 0

        open_tickets = store.list_tickets(status=TicketStatus.OPEN)
        assert len(open_tickets) == 2

        all_tickets = store.list_tickets(limit=2)
        assert len(all_tickets) == 2


def test_save_ticket_with_mask_pii() -> None:
    with TicketStore(":memory:") as store:
        ticket = Ticket(
            id="T-PII",
            source="test",
            subject="Assistance for john.doe@example.com",
            description="Call me at +55 11 99999-8888 or use CPF 111.222.333-44",
            status=TicketStatus.OPEN,
            created_at=datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC),
            comments=[
                Comment(
                    id="C-PII",
                    author_role="customer",
                    body="My email again: john.doe@example.com",
                    created_at=datetime(2026, 4, 1, 10, 5, 0, tzinfo=UTC),
                    public=True,
                )
            ],
        )
        store.save_ticket(ticket, mask_pii=True)

        retrieved = store.get_ticket("T-PII")
        assert retrieved is not None
        assert "john.doe@example.com" not in retrieved.subject
        assert "[EMAIL_1]" in retrieved.subject
        assert "+55 11 99999-8888" not in retrieved.description
        assert "111.222.333-44" not in retrieved.description
        assert "john.doe@example.com" not in retrieved.comments[0].body


def test_parameterized_queries_sql_injection_defense() -> None:
    with TicketStore(":memory:") as store:
        malicious_id = "T'; DROP TABLE tickets; --"
        malicious_subject = "'; DROP TABLE comments; SELECT '1'='"
        ticket = _make_ticket(ticket_id=malicious_id, subject=malicious_subject)
        store.save_ticket(ticket)

        retrieved = store.get_ticket(malicious_id)
        assert retrieved is not None
        assert retrieved.id == malicious_id
        assert retrieved.subject == malicious_subject

        # Verify tables still exist and query works
        assert store.count_tickets() == 1


def test_resolve_db_url_and_load_env(tmp_path: Path) -> None:
    assert resolve_db_url(":memory:") == "sqlite+pysqlite:///:memory:"
    assert resolve_db_url("postgres://u:p@localhost/db") == "postgresql+psycopg://u:p@localhost/db"
    assert resolve_db_url("sqlite:///data.db") == "sqlite:///data.db"

    env_file = tmp_path / ".env"
    env_file.write_text("DB_URL=postgresql+psycopg://custom:secret@db/test\n", encoding="utf-8")
    assert load_env_db_url(env_file) == "postgresql+psycopg://custom:secret@db/test"

    # Clean env before test
    old_env = os.environ.get("DB_URL")
    try:
        os.environ["DB_URL"] = "sqlite:///from_env.db"
        assert resolve_db_url(env_file=env_file) == "sqlite:///from_env.db"
    finally:
        if old_env is not None:
            os.environ["DB_URL"] = old_env
        else:
            os.environ.pop("DB_URL", None)


def test_build_engine_smoke(tmp_path: Path) -> None:
    db_file = tmp_path / "smoke.db"
    engine = build_engine(str(db_file))
    try:
        store = TicketStore(engine=engine)
        store.save_ticket(_make_ticket("T-SMOKE"))
        assert store.count_tickets() == 1
        store.close()
    finally:
        engine.dispose()
