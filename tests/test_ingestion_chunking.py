"""Tests for chunking per interaction plus case summary."""

from datetime import UTC, datetime, timedelta

import pytest

from support_engine.ingestion.chunking import ChunkingConfig, chunk_ticket, extractive_summary, split_text
from support_engine.models import Comment, Ticket, TicketStatus

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def _ticket(**kw: object) -> Ticket:
    data: dict[str, object] = {
        "id": "T-1",
        "source": "test",
        "subject": "Export fails",
        "description": "The CSV export fails with a timeout after two minutes.",
        "status": TicketStatus.SOLVED,
        "product": "reports",
        "tags": ["export"],
        "created_at": T0,
        "comments": [
            Comment(id="c1", author_role="agent", body="Please share the report id.", created_at=T0 + timedelta(1)),
            Comment(id="c2", author_role="agent", body="Internal: check worker logs.", created_at=T0, public=False),
            Comment(
                id="c3",
                author_role="agent",
                body="We raised the export timeout to ten minutes; it works now.",
                created_at=T0 + timedelta(2),
            ),
        ],
    }
    data.update(kw)
    return Ticket(**data)  # type: ignore[arg-type]


def test_split_text_short_text_is_one_piece() -> None:
    assert split_text("short text", 100, 10) == ["short text"]
    assert split_text("   ", 100, 10) == []


def test_split_text_respects_max_and_overlaps() -> None:
    sentences = [f"Sentence number {i} explains one step of the fix." for i in range(30)]
    text = " ".join(sentences)
    pieces = split_text(text, max_chars=200, overlap_chars=40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    # Overlap: each piece starts with words from the end of the previous one.
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        assert nxt.split()[0] in prev.split()[-12:]
    # Nothing lost.
    joined = " ".join(pieces)
    for s in sentences:
        assert s in joined


def test_split_text_hard_splits_long_words() -> None:
    pieces = split_text("x" * 250, max_chars=100, overlap_chars=0)
    assert [len(p) for p in pieces] == [100, 100, 50]


def test_chunk_ticket_interactions_and_summary() -> None:
    chunks = chunk_ticket(_ticket())
    kinds = [c.kind for c in chunks]
    assert kinds.count("summary") == 1
    interaction_ids = [c.metadata.get("comment_id") for c in chunks if c.kind == "interaction"]
    assert interaction_ids == [None, "c1", "c3"]  # description first, private note skipped, time order
    first = chunks[0]
    assert first.ticket_id == "T-1"
    assert first.metadata["product"] == "reports"
    assert first.embed_text.startswith("Export fails\ncustomer:")
    summary = chunks[-1]
    assert "Problem: The CSV export fails" in summary.text
    assert "Resolution: We raised the export timeout" in summary.text
    assert summary.metadata["resolved"] is True


def test_chunk_ticket_config_options() -> None:
    cfg = ChunkingConfig(include_private=True, include_summary=False, context_header=False, min_chars=0)
    chunks = chunk_ticket(_ticket(), cfg)
    assert all(c.kind == "interaction" for c in chunks)
    assert "c2" in [c.metadata.get("comment_id") for c in chunks]
    assert all(c.embed_text == c.text for c in chunks)


def test_chunk_ticket_custom_summarizer() -> None:
    chunks = chunk_ticket(_ticket(), summarizer=lambda t: f"custom summary of {t.id}")
    assert chunks[-1].text == "custom summary of T-1"


def test_chunk_ids_are_unique_for_long_interactions() -> None:
    long_body = " ".join(f"Line {i} of a long troubleshooting log entry." for i in range(80))
    ticket = _ticket(comments=[Comment(id="c9", author_role="agent", body=long_body, created_at=T0)])
    chunks = chunk_ticket(ticket, ChunkingConfig(max_chars=300, overlap_chars=50))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert max(c.part for c in chunks) >= 2


def test_extractive_summary_limit() -> None:
    ticket = _ticket(description="word " * 500)
    assert len(extractive_summary(ticket, max_chars=200)) <= 200


def test_config_rejects_overlap_not_below_max() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_chars=100, overlap_chars=100)
