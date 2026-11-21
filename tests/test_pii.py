"""Tests for PII masking and anonymization."""

from datetime import UTC, datetime

from support_engine.models import Comment, Ticket, TicketStatus
from support_engine.pii import (
    Anonymizer,
    mask_pii,
    mask_ticket,
)


def test_anonymize_entities() -> None:
    anon = Anonymizer()
    text = (
        "Customer contact: alice@example.com, CPF 123.456.789-00, CNPJ 12.345.678/0001-90, "
        "Phone +55 11 98765-4321, Card 4111 2222 3333 4444, IP 192.168.1.100, "
        "Token sk-abcdef1234567890abcdef123456"
    )
    masked = anon.anonymize_text(text)

    assert "alice@example.com" not in masked
    assert "123.456.789-00" not in masked
    assert "12.345.678/0001-90" not in masked
    assert "98765-4321" not in masked
    assert "4111 2222 3333 4444" not in masked
    assert "192.168.1.100" not in masked
    assert "sk-abcdef1234567890abcdef123456" not in masked

    assert "[EMAIL_1]" in masked
    assert "[CPF_1]" in masked
    assert "[CNPJ_1]" in masked
    assert "[PHONE_1]" in masked
    assert "[CARD_1]" in masked
    assert "[IP_1]" in masked
    assert "[SECRET_1]" in masked

    # Verify deanonymize
    restored = anon.deanonymize_text(masked)
    assert restored == text


def test_mask_pii_convenience() -> None:
    result = mask_pii("Contact support@company.com or call (11) 91234-5678")
    assert "support@company.com" not in result
    assert "91234-5678" not in result
    assert "[EMAIL_1]" in result
    assert "[PHONE_1]" in result


def test_mask_ticket() -> None:
    ticket = Ticket(
        id="TCK-999",
        source="test",
        subject="Issue with user bob@test.com",
        description="The customer with CPF 987.654.321-99 is having issues.",
        status=TicketStatus.OPEN,
        created_at=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
        comments=[
            Comment(
                id="C-1",
                author_role="customer",
                body="Here is my card 1234-5678-9012-3456 please charge it.",
                created_at=datetime(2026, 3, 1, 10, 5, 0, tzinfo=UTC),
                public=True,
            )
        ],
    )

    masked = mask_ticket(ticket)

    assert masked.id == "TCK-999"
    assert "bob@test.com" not in masked.subject
    assert "[EMAIL_1]" in masked.subject
    assert "987.654.321-99" not in masked.description
    assert "[CPF_1]" in masked.description
    assert "1234-5678-9012-3456" not in masked.comments[0].body
    assert "[CARD_1]" in masked.comments[0].body
    assert masked.comments[0].id == "C-1"
    assert masked.status == TicketStatus.OPEN
