"""Feedback loop: masking, corrections back into the index, superseded answers, storage."""

import logging
from datetime import UTC, datetime, timedelta

import pytest

from support_engine.curation import (
    CurationConfig,
    CurationPipeline,
    Feedback,
    FeedbackConfig,
    FeedbackStore,
    bayesian_rating,
    chunk_priority,
)
from support_engine.ingestion import IngestionPipeline
from support_engine.models import Comment, Ticket, TicketStatus

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def tickets() -> list[Ticket]:
    def t(tid: str, status: TicketStatus, csat: float | None, answer: str) -> Ticket:
        return Ticket(
            id=tid,
            source="csv",
            subject=f"Problem {tid}",
            description="The nightly export job stops with an error and no file is produced.",
            status=status,
            csat=csat,
            created_at=T0,
            comments=[
                Comment(id=f"{tid}-c1", author_role="agent", body=answer, created_at=T0 + timedelta(hours=1)),
            ],
        )

    return [
        t("T-1", TicketStatus.SOLVED, 1.0, "Increase the export timeout to 300 seconds in the job settings."),
        t("T-2", TicketStatus.OPEN, None, "We are looking into it and will get back to you soon."),
    ]


def ingest():
    result = IngestionPipeline().process(tickets())
    return result.tickets, result.chunks


def test_feedback_masks_free_text() -> None:
    fb = Feedback.vote("T-1", helpful=False, correction="Write to ana.souza@example.com", query="call 11 98765-4321")
    assert "example.com" not in fb.correction and "[EMAIL_1]" in fb.correction
    assert "98765" not in fb.query


def test_feedback_id_is_stable_and_stars_normalised() -> None:
    a = Feedback(ticket_id="T-1", rating=1.0, created_at=T0)
    b = Feedback(ticket_id="T-1", rating=1.0, created_at=T0)
    assert a.id == b.id and a.id.startswith("fb-")
    assert Feedback.stars("T-1", 5).rating == 1.0
    assert Feedback.stars("T-1", 1).rating == 0.0
    with pytest.raises(ValueError):
        Feedback.stars("T-1", 6)


def test_bayesian_rating() -> None:
    assert bayesian_rating([]) is None
    assert bayesian_rating([1.0, 1.0], prior_mean=0.5, prior_weight=2) == pytest.approx(0.75)


def test_scores_are_attached_to_every_chunk() -> None:
    ts, chunks = ingest()
    result = CurationPipeline().curate(ts, chunks)
    assert result.stats.chunks_scored == len(chunks)
    for chunk in result.chunks:
        assert chunk.metadata["resolution_score"] == result.scores[chunk.ticket_id].score
        assert "score_confidence" in chunk.metadata
    assert result.scores["T-1"].score > result.scores["T-2"].score
    # Inputs are not modified.
    assert all("resolution_score" not in c.metadata for c in chunks)


def test_correction_enters_index_and_demotes_answer() -> None:
    ts, chunks = ingest()
    fb = Feedback.vote("T-2", helpful=False, correction="Restart the export worker, then rerun the job.")
    result = CurationPipeline().curate(ts, chunks, [fb])
    corrections = [c for c in result.chunks if c.kind == "correction"]
    assert len(corrections) == 1
    corr = corrections[0]
    assert corr.text == "Restart the export worker, then rerun the job."
    assert corr.ticket_id == "T-2" and corr.metadata["resolution_score"] == 1.0
    replaced = set(corr.metadata["corrects"])
    assert "T-2:summary" in replaced and any(":i1:" in cid for cid in replaced)
    for chunk in result.chunks:
        if chunk.id in replaced:
            assert chunk.metadata["superseded_by"] == [corr.id]
            assert chunk.metadata["curation_priority"] == pytest.approx(chunk.metadata["resolution_score"] * 0.5)
    assert result.stats.corrections_added == 1 and result.stats.chunks_superseded == len(replaced)


def test_correction_on_a_chunk_supersedes_only_that_chunk() -> None:
    ts, chunks = ingest()
    target = next(c for c in chunks if c.ticket_id == "T-1" and c.kind == "interaction")
    fb = Feedback(ticket_id="T-1", chunk_id=target.id, rating=0.0, correction="Use 600 seconds for large exports.")
    result = CurationPipeline().curate(ts, chunks, [fb])
    corr = next(c for c in result.chunks if c.kind == "correction")
    assert corr.metadata["corrects"] == [target.id]


@pytest.mark.parametrize("policy", ["drop", "keep"])
def test_superseded_policy(policy: str) -> None:
    ts, chunks = ingest()
    fb = Feedback.vote("T-2", helpful=False, correction="Restart the export worker.")
    cfg = CurationConfig(feedback=FeedbackConfig(superseded_policy=policy))
    result = CurationPipeline(cfg).curate(ts, chunks, [fb])
    ids = {c.id for c in result.chunks}
    if policy == "drop":
        assert "T-2:summary" not in ids
    else:
        kept = next(c for c in result.chunks if c.id == "T-2:summary")
        assert kept.metadata["curation_priority"] == kept.metadata["resolution_score"]


def test_corrections_can_be_disabled() -> None:
    ts, chunks = ingest()
    fb = Feedback.vote("T-2", helpful=False, correction="Restart the export worker.")
    cfg = CurationConfig(feedback=FeedbackConfig(apply_corrections=False))
    result = CurationPipeline(cfg).curate(ts, chunks, [fb])
    assert not [c for c in result.chunks if c.kind == "correction"]


def test_chunk_level_rating_adjusts_only_that_chunk() -> None:
    ts, chunks = ingest()
    target = next(c for c in chunks if c.ticket_id == "T-1" and c.kind == "interaction")
    fb = [Feedback(ticket_id="T-1", chunk_id=target.id, rating=0.0) for _ in range(5)]
    result = CurationPipeline().curate(ts, chunks, fb)
    by_id = {c.id: c for c in result.chunks}
    assert by_id[target.id].metadata["resolution_score"] < by_id["T-1:summary"].metadata["resolution_score"]


def test_min_score_drops_weak_chunks_but_keeps_corrections() -> None:
    ts, chunks = ingest()
    fb = Feedback.vote("T-2", helpful=False, correction="Restart the export worker.")
    result = CurationPipeline(CurationConfig(min_score=0.6)).curate(ts, chunks, [fb])
    assert {c.ticket_id for c in result.chunks if c.kind != "correction"} == {"T-1"}
    assert any(c.kind == "correction" for c in result.chunks)
    assert result.stats.chunks_below_min > 0


def test_feedback_for_unknown_ticket_is_counted_not_used() -> None:
    ts, chunks = ingest()
    result = CurationPipeline().curate(ts, chunks, [Feedback.vote("NOPE", helpful=True, correction="x y z")])
    assert result.stats.feedback_unknown_ticket == 1
    assert not [c for c in result.chunks if c.kind == "correction"]


def test_rerank_prefers_curated_answers() -> None:
    ts, chunks = ingest()
    result = CurationPipeline().curate(ts, chunks)
    good = next(c for c in result.chunks if c.id == "T-1:summary")
    weak = next(c for c in result.chunks if c.id == "T-2:summary")
    ranked = CurationPipeline().rerank([(weak, 0.8), (good, 0.75)])
    assert ranked[0][0].id == "T-1:summary"
    assert chunk_priority(chunks[0]) == 0.5


def test_logs_hold_no_pii(caplog: pytest.LogCaptureFixture) -> None:
    ts, chunks = ingest()
    fb = Feedback.vote("T-2", helpful=False, correction="Mail ana.souza@example.com", query="ana.souza@example.com")
    with caplog.at_level(logging.DEBUG):
        CurationPipeline().curate(ts, chunks, [fb])
    text = caplog.text
    assert "curation:" in text
    assert "example.com" not in text and "export worker" not in text and "nightly export" not in text


def test_feedback_store_round_trip(tmp_path) -> None:
    db = f"sqlite:///{(tmp_path / 'fb.db').as_posix()}"
    items = [
        Feedback(ticket_id="T-1", rating=1.0, created_at=T0),
        Feedback(ticket_id="T-2", chunk_id="T-2:summary", rating=0.0, correction="Mail bob@example.com",
                 query="why export fails", reviewer_role="reviewer", created_at=T0 + timedelta(minutes=1)),
    ]
    with FeedbackStore(db, env_file=None) as store:
        assert store.add_many(items) == 2
        store.add(items[0])  # replace, not duplicate
        assert store.count() == 2
        loaded = store.list_feedback()
        assert [f.id for f in loaded] == [f.id for f in items]
        assert loaded[1].correction == "Mail [EMAIL_1]" and loaded[1].chunk_id == "T-2:summary"
        assert [f.id for f in store.list_feedback("T-1")] == [items[0].id]
