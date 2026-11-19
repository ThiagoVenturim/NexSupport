"""Deterministic synthetic help desk tickets and agent outcomes for metric examples and tests.

Every name, address, product and company here is invented. A few messages carry a
made-up e-mail address and phone number on purpose, so examples can show that they
never reach the metric tables, the dashboards or the logs.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from support_engine.models import Comment, Ticket, TicketStatus

START = datetime(2026, 4, 1, 8, 0, tzinfo=UTC)
END = datetime(2026, 6, 30, 18, 0, tzinfo=UTC)
TEAMS = ("tier1", "tier2")
QUEUES = ("billing", "access", "reports")
CHANNELS = ("email", "chat", "web", "phone")
PRIORITIES = ("low", "normal", "normal", "high", "urgent")
FAKE_EMAIL = "maria.lima@example.com"
FAKE_PHONE = "+55 11 91234-5678"

_TOPICS: dict[str, tuple[str, str]] = {
    "billing": ("Charged twice on the invoice", "billing"),
    "access": ("SSO login redirect loop", "identity"),
    "reports": ("Monthly report does not open", "reports"),
}

_POSITIVE = (
    "Thanks, it works now. Great support!",
    "Muito obrigado, funcionou perfeitamente.",
    "Muchas gracias, funciona perfecto.",
)
_NEUTRAL = (
    "How do I change the billing address?",
    "Qual o prazo para gerar o relatorio?",
    "Where can I download the invoice?",
)
_NEGATIVE = (
    "This is unacceptable, it is still not working. I want a refund or I will cancel.",
    "Pessimo, o sistema nao funciona e ninguem responde. Vou abrir reclamacao no procon.",
    "Very frustrating, the page is broken again.",
)


def _comments(tid: str, created: datetime, customer: list[str], replies: list[float]) -> list[Comment]:
    out = [
        Comment(id=f"{tid}-c{i}", author_role="customer", body=body, created_at=created + timedelta(hours=2 * i))
        for i, body in enumerate(customer)
    ]
    out += [
        Comment(id=f"{tid}-a{i}", author_role="agent", body="We are looking into it.", created_at=created + hours)
        for i, hours in enumerate(timedelta(hours=h) for h in replies)
    ]
    return out


def synthetic_tickets(count: int = 120, *, seed: int = 7) -> list[Ticket]:
    """``count`` tickets from April to June 2026 plus a spike of export errors on the last day."""
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data, not security
    span = (END - START).total_seconds() - 3 * 86400
    tickets: list[Ticket] = []
    for i in range(count):
        queue = QUEUES[i % len(QUEUES)]
        subject, product = _TOPICS[queue]
        created = START + timedelta(seconds=rng.uniform(0, span))
        mood = rng.random()
        if mood < 0.3:
            customer = [rng.choice(_NEUTRAL), rng.choice(_NEGATIVE)]
        elif mood < 0.65:
            customer = [rng.choice(_NEUTRAL)]
        else:
            customer = [rng.choice(_NEUTRAL), rng.choice(_POSITIVE)]
        if i % 9 == 0:
            customer.append(f"Please call me at {FAKE_PHONE} or write to {FAKE_EMAIL}.")
        resolved = rng.random() < 0.75
        first_reply = round(rng.uniform(0.2, 12.0), 2)
        replies = [] if (resolved and rng.random() < 0.2) else [first_reply]
        solved_at = created + timedelta(hours=rng.uniform(first_reply + 0.5, 60.0)) if resolved else None
        tickets.append(
            Ticket(
                id=f"SUP-{i + 1:04d}",
                source="zendesk" if i % 2 else "freshdesk",
                subject=subject,
                description=customer[0],
                status=TicketStatus.SOLVED if resolved else rng.choice((TicketStatus.OPEN, TicketStatus.PENDING)),
                priority=PRIORITIES[i % len(PRIORITIES)],
                product=product,
                tags=["deflected"] if resolved and not replies else [],
                created_at=created,
                solved_at=solved_at,
                reopen_count=1 if rng.random() < 0.1 else 0,
                csat=round(rng.uniform(0.6, 1.0) if mood >= 0.3 else rng.uniform(0.0, 0.5), 2) if resolved else None,
                comments=_comments(f"SUP-{i + 1:04d}", created, customer, replies),
                extra={"team": TEAMS[i % len(TEAMS)], "queue": queue, "channel": CHANNELS[i % len(CHANNELS)]},
            )
        )
    for j in range(6):
        created = END - timedelta(hours=10 - j)
        tid = f"EXP-{j + 1:02d}"
        tickets.append(
            Ticket(
                id=tid,
                source="zendesk",
                subject="Invoice export fails with error 500 after the update",
                description="Since version 2.3 the invoice export fails with error 500.",
                priority="high",
                product="billing",
                tags=["export"],
                created_at=created,
                comments=_comments(tid, created, ["Since version 2.3 the invoice export fails. Very frustrating."], []),
                extra={"team": "tier2", "queue": "billing", "channel": "web", "version": "2.3"},
            )
        )
    return tickets


def synthetic_outcomes(tickets: list[Ticket], *, seed: int = 11) -> list[dict[str, Any]]:
    """Support agent outcomes for about two thirds of the tickets, tied by ``ticket_id``."""
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data, not security
    outcomes = []
    for ticket in tickets:
        roll = rng.random()
        if roll < 0.3:
            outcome = {"action": "answer", "level": 1, "cached": rng.random() < 0.3}
        elif roll < 0.5:
            outcome = {"action": "review", "level": 2}
        elif roll < 0.65:
            outcome = {"action": "escalate", "level": 3}
        else:
            continue
        outcome["attempts"] = 1 + int(rng.random() < 0.2)
        outcomes.append({"ticket_id": ticket.id, **outcome})
    return outcomes


__all__ = ["CHANNELS", "FAKE_EMAIL", "FAKE_PHONE", "QUEUES", "TEAMS", "synthetic_outcomes", "synthetic_tickets"]
