"""Synthetic tickets shared by the analytics tests. All names, addresses and products are invented."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from support_engine.models import Comment, Ticket, TicketStatus

NOW = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)


def make_ticket(
    tid: str,
    subject: str,
    customer: Sequence[str] = (),
    *,
    created_at: datetime = NOW - timedelta(hours=2),
    agent: Sequence[tuple[float, str]] = (),
    customer_gap_hours: float = 1.0,
    **kwargs,
) -> Ticket:
    """Customer messages every ``customer_gap_hours`` from creation; agent replies at (hours after creation, text)."""
    comments = [
        Comment(
            id=f"{tid}-u{i}",
            author_role="customer",
            body=body,
            created_at=created_at + timedelta(hours=i * customer_gap_hours),
        )
        for i, body in enumerate(customer)
    ]
    comments += [
        Comment(id=f"{tid}-a{i}", author_role="agent", body=body, created_at=created_at + timedelta(hours=h))
        for i, (h, body) in enumerate(agent)
    ]
    base = {"id": tid, "source": "csv", "subject": subject, "created_at": created_at, "comments": comments}
    base.update(kwargs)
    return Ticket(**base)


def trend_tickets() -> list[Ticket]:
    """Seven quiet days of two topics, then a spike of export errors on the last day."""
    out = []
    for day in range(7, 0, -1):
        created = NOW - timedelta(days=day)
        out.append(
            make_ticket(
                f"EXP-{day}",
                "Invoice export fails with error 500",
                ["The invoice export fails with error 500."],
                created_at=created,
                product="billing",
                status=TicketStatus.SOLVED,
                solved_at=created + timedelta(hours=3),
            )
        )
        out.append(
            make_ticket(
                f"SSO-{day}",
                "SSO login redirect loop",
                ["Users loop back to the SSO login screen."],
                created_at=created + timedelta(hours=1),
                product="identity",
                status=TicketStatus.SOLVED,
                solved_at=created + timedelta(hours=4),
            )
        )
    for i in range(5):
        out.append(
            make_ticket(
                f"EXP-NOW-{i}",
                "Invoice export fails with error 500 after the update",
                ["Since version 2.3 the invoice export fails with error 500. Very frustrating."],
                created_at=NOW - timedelta(hours=6 - i),
                product="billing",
                tags=["export"],
                extra={"version": "2.3"},
            )
        )
    out.append(
        make_ticket(
            "SSO-NOW",
            "SSO login redirect loop",
            ["Users loop back to the SSO login screen."],
            created_at=NOW - timedelta(hours=3),
            product="identity",
        )
    )
    return out


def mixed_tickets() -> list[Ticket]:
    """Open and solved tickets in three languages, with personal data that must stay out of reports."""
    return [
        make_ticket(
            "R-1",
            "Charged twice",
            [
                "Hi, I was charged twice this month.",
                "Still no response. This is unacceptable, I want a refund or I will cancel. "
                "Contact me at ana.souza@example.com, Ana Souza.",
            ],
            created_at=NOW - timedelta(hours=40),
            customer_gap_hours=12,
            priority="high",
            product="billing",
        ),
        make_ticket(
            "R-2",
            "Relatorio nao abre",
            ["O relatorio nao abre depois da atualizacao, pessimo. Vou abrir reclamacao no procon."],
            created_at=NOW - timedelta(hours=10),
            agent=[(9.0, "Estamos verificando.")],
            priority="normal",
            product="reports",
            reopen_count=2,
            language="pt",
        ),
        make_ticket(
            "R-3",
            "Pregunta sobre facturas",
            ["Muchas gracias, funciona perfecto."],
            created_at=NOW - timedelta(hours=30),
            agent=[(0.5, "Listo, ya puede descargar la factura.")],
            status=TicketStatus.SOLVED,
            solved_at=NOW - timedelta(hours=29),
            csat=1.0,
            product="billing",
            language="es",
        ),
        make_ticket(
            "R-4",
            "Question about seats",
            ["How do I add two more seats to the plan?"],
            created_at=NOW - timedelta(hours=1),
            agent=[(0.5, "Open Settings, then Billing, then Seats.")],
            priority="low",
            product="admin",
        ),
    ]
