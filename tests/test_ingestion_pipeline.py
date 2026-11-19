"""End-to-end tests for the ingestion pipeline, including PII masking guarantees (LGPD)."""

import io
import json
import logging
import socket
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from support_engine.cli import main
from support_engine.ingestion import (
    ChunkingConfig,
    DedupConfig,
    IngestionConfig,
    IngestionPipeline,
    PrivacyConfig,
    ingest_tickets,
)
from support_engine.models import Comment, Ticket, TicketStatus
from support_engine.storage import TicketStore

T0 = datetime(2026, 4, 2, 8, 0, tzinfo=UTC)

PII_VALUES = [
    "carla.nunes@example.com",
    "123.456.789-09",
    "(11) 98888-7777",
    "4111 1111 1111 1111",
    "Carla Nunes",
    "Carla",
    "Nunes",
    "Diego Ramos",
]

MACRO = (
    "To reset the integration token open Settings, then Integrations, click Regenerate token and paste "
    "the new value in the connector configuration before running the sync again."
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_a: object, **_k: object) -> None:
        raise AssertionError("network access is not allowed in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def _tickets() -> list[Ticket]:
    t1 = Ticket(
        id="P-1",
        source="test",
        subject="Sync broken for carla.nunes@example.com",
        description=(
            "<p>Hi team,</p><p>My name is Carla Nunes, CPF 123.456.789-09. The sync stopped. "
            "Call me at (11) 98888-7777.</p><p>Thanks,</p><p>Carla</p>"
        ),
        status=TicketStatus.SOLVED,
        created_at=T0,
        extra={"requester_name": "Carla Nunes", "plan": "pro"},
        comments=[
            Comment(id="a1", author_role="agent", body=f"Hello Carla,\n\n{MACRO}\n\nBest regards,\nDiego Ramos",
                    created_at=T0 + timedelta(hours=1)),
            Comment(id="a2", author_role="customer",
                    body="Done, card 4111 1111 1111 1111 was charged twice too.\n\nOn Tue, Diego wrote:\n> old",
                    created_at=T0 + timedelta(hours=2)),
            Comment(id="a3", author_role="customer",
                    body="Done, card 4111 1111 1111 1111 was charged twice too.",
                    created_at=T0 + timedelta(hours=3)),
            Comment(id="a4", author_role="customer", body="Sent from my phone", created_at=T0 + timedelta(hours=4)),
        ],
    )
    t2 = Ticket(
        id="P-2",
        source="test",
        subject="Token expired",
        description="The connector says the token expired.",
        status=TicketStatus.SOLVED,
        created_at=T0,
        comments=[Comment(id="b1", author_role="agent", body=MACRO, created_at=T0 + timedelta(hours=1))],
    )
    return [t1, t2]


def _all_text(result: object) -> str:
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)  # type: ignore[attr-defined]


def test_pipeline_masks_all_pii_by_default() -> None:
    result = IngestionPipeline().process(_tickets())
    blob = _all_text(result)
    for value in PII_VALUES:
        assert value not in blob, f"PII leaked: {value!r}"
    t1 = result.tickets[0]
    assert "[EMAIL_1]" in t1.subject
    assert "[CPF_1]" in t1.description
    assert "[NAME_" in t1.description
    assert t1.extra["plan"] == "pro"
    assert result.stats.pii_masked["NAME"] >= 2
    assert result.stats.pii_masked["CARD"] == 1


def test_pipeline_cleans_and_deduplicates() -> None:
    result = IngestionPipeline().process(_tickets())
    t1 = result.tickets[0]
    assert "<p>" not in t1.description
    assert "Thanks" not in t1.description
    # a2 lost its quoted reply, a3 repeats a2, a4 is only a footer.
    assert [c.id for c in t1.comments] == ["a1", "a2"]
    assert "wrote:" not in t1.comments[1].body
    assert "Best regards" not in t1.comments[0].body
    stats = result.stats
    assert stats.comments_in == 5
    assert stats.comments_kept == 3
    assert stats.comments_duplicate == 1
    assert stats.comments_empty == 1
    # The macro answer of P-2 duplicates the one in P-1 across the corpus.
    assert stats.chunks_duplicate == 1
    assert stats.duplicate_of == {"P-2:i1:0": "P-1:i1:0"}
    assert {c.kind for c in result.chunks} == {"interaction", "summary"}
    assert sum(1 for c in result.chunks if c.kind == "summary") == 2


def test_dedup_scope_ticket_and_none() -> None:
    per_ticket = ingest_tickets(_tickets(), IngestionConfig(dedup_chunks="ticket"))
    assert per_ticket.stats.chunks_duplicate == 0
    off = ingest_tickets(_tickets(), IngestionConfig(dedup=DedupConfig(enabled=False), dedup_comments=False))
    assert off.stats.chunks_duplicate == 0
    assert off.stats.comments_duplicate == 0


def test_unmasked_requires_explicit_opt_out() -> None:
    with pytest.raises(ValueError):
        PrivacyConfig(mask_pii=False)
    cfg = IngestionConfig(privacy=PrivacyConfig(mask_pii=False, allow_unmasked=True))
    result = IngestionPipeline(cfg).process(_tickets())
    assert "carla.nunes@example.com" in result.tickets[0].subject


def test_summarizer_only_sees_masked_text() -> None:
    seen: list[str] = []

    def summarizer(ticket: Ticket) -> str:
        seen.append(ticket.model_dump_json())
        return "summary"

    IngestionPipeline(summarizer=summarizer).process(_tickets())
    assert seen
    for payload in seen:
        for value in PII_VALUES:
            assert value not in payload


def test_logs_never_contain_pii(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        IngestionPipeline().process(_tickets())
    assert caplog.records
    for value in PII_VALUES:
        assert value not in caplog.text


def test_chunk_size_is_configurable() -> None:
    cfg = IngestionConfig(chunking=ChunkingConfig(max_chars=80, overlap_chars=10, include_summary=False))
    result = IngestionPipeline(cfg).process(_tickets())
    assert all(len(c.text) <= 80 for c in result.chunks)
    assert all(c.kind == "interaction" for c in result.chunks)


def test_masked_tickets_persist_without_pii(tmp_path: Path) -> None:
    result = IngestionPipeline().process(_tickets())
    db_url = f"sqlite:///{(tmp_path / 'ingest.db').as_posix()}"
    with TicketStore(db_url) as store:
        store.save_tickets(result.tickets)
        stored = store.get_ticket("P-1")
    assert stored is not None
    blob = stored.model_dump_json()
    for value in PII_VALUES:
        assert value not in blob


def test_storage_masks_by_default(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'raw.db').as_posix()}"
    with TicketStore(db_url) as store:
        store.save_ticket(_tickets()[0])
        stored = store.get_ticket("P-1")
    assert stored is not None
    assert "carla.nunes@example.com" not in stored.subject
    assert "123.456.789-09" not in stored.description


def test_cli_ingest_writes_masked_chunks(tmp_path: Path) -> None:
    src = tmp_path / "tickets.json"
    src.write_text(json.dumps([t.model_dump(mode="json") for t in _tickets()]), encoding="utf-8")
    out = tmp_path / "out" / "chunks.jsonl"
    db_url = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--db-url", db_url, "ingest", "--input", str(src), "--chunks-out", str(out), "--max-chars", "400"])
    assert code == 0
    report = buf.getvalue()
    assert "Ingested 2 ticket(s)" in report
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines
    blob = "\n".join(lines) + report
    for value in PII_VALUES:
        assert value not in blob
    with TicketStore(db_url) as store:
        assert store.count_tickets() == 2


def test_cli_ingest_rejects_bad_settings(tmp_path: Path) -> None:
    src = tmp_path / "tickets.json"
    src.write_text("[]", encoding="utf-8")
    assert main(["ingest", "--input", str(src), "--no-store", "--max-chars", "100", "--overlap-chars", "100"]) == 2


def test_cli_import_masks_by_default(tmp_path: Path) -> None:
    src = tmp_path / "tickets.json"
    src.write_text(json.dumps([_tickets()[0].model_dump(mode="json")]), encoding="utf-8")
    db_url = f"sqlite:///{(tmp_path / 'imp.db').as_posix()}"
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["--db-url", db_url, "import", "--input", str(src)]) == 0
    assert "(PII masked)" in buf.getvalue()
    with TicketStore(db_url) as store:
        stored = store.get_ticket("P-1")
    assert stored is not None
    assert "carla.nunes@example.com" not in stored.subject
