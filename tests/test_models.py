from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from support_engine import Comment, Ticket, TicketStatus

T0 = datetime(2026, 1, 5, 9, 0)


def make_ticket(**kwargs) -> Ticket:
    data = {"id": "T-1001", "source": "csv", "subject": "Cannot export invoice", "created_at": T0}
    data.update(kwargs)
    return Ticket(**data)


def test_thread_text_orders_public_comments():
    ticket = make_ticket(
        description="Export button does nothing.",
        comments=[
            Comment(id="c2", author_role="agent", body="Fixed in 4.2.", created_at=T0 + timedelta(hours=2)),
            Comment(id="c1", author_role="customer", body="Still failing.", created_at=T0 + timedelta(hours=1)),
            Comment(id="c3", author_role="agent", body="internal note", created_at=T0, public=False),
        ],
    )
    assert ticket.thread_text().split("\n\n") == [
        "Cannot export invoice",
        "Export button does nothing.",
        "Still failing.",
        "Fixed in 4.2.",
    ]


def test_is_resolved():
    assert make_ticket(status=TicketStatus.SOLVED).is_resolved
    assert not make_ticket(status="pending").is_resolved


def test_csat_is_normalised():
    with pytest.raises(ValidationError):
        make_ticket(csat=5)
