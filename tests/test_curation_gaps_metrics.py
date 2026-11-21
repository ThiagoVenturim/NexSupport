"""Knowledge gaps, article suggestions and quality metrics."""

from support_engine.curation import (
    CurationPipeline,
    Feedback,
    GapConfig,
    GapDetector,
    ResolutionScore,
    compute_metrics,
    query_tokens,
)
from support_engine.ingestion.chunking import Chunk


def test_query_tokens_drop_stopwords_and_placeholders() -> None:
    assert query_tokens("How do I reset the [EMAIL_1] password?") == {"reset", "password"}


def test_repeated_gap_suggests_an_article() -> None:
    gaps = GapDetector(GapConfig(min_occurrences=3, similarity=0.4))
    gaps.observe("How to reset two factor authentication", None, ticket_id="A")
    gaps.observe("reset two factor authentication on new phone", 0.1, ticket_id="B")
    gaps.observe("two factor authentication reset not working", 0.2, ticket_id="C")
    gaps.observe("invoice shows wrong currency", 0.0)
    assert gaps.observe("reset two factor authentication", 0.9) is None  # good answer, not a gap
    suggestions = gaps.suggestions()
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s.occurrences == 3 and s.ticket_ids == ["A", "B", "C"]
    assert {"reset", "factor", "authentication"} <= set(s.keywords)
    assert len(gaps.clusters) == 2 and len(gaps) == 4


def test_gap_below_min_occurrences_is_not_suggested() -> None:
    gaps = GapDetector(GapConfig(min_occurrences=2))
    gaps.observe("printer driver missing", None)
    assert gaps.suggestions() == []


def test_gap_queries_are_masked() -> None:
    gaps = GapDetector(GapConfig(min_occurrences=1))
    gaps.observe("refund for ana.souza@example.com order", None)
    sample = gaps.suggestions()[0].sample_queries[0]
    assert "example.com" not in sample and "[EMAIL_1]" in sample


def test_disabled_gap_detection() -> None:
    gaps = GapDetector(GapConfig(enabled=False))
    assert gaps.observe("anything at all here", None) is None
    assert gaps.suggestions() == []


def test_unhelpful_feedback_without_correction_is_a_gap() -> None:
    gaps = GapDetector(GapConfig(min_occurrences=2))
    added = gaps.observe_feedback([
        Feedback.vote("T-1", helpful=False, query="sync calendar outlook"),
        Feedback.vote("T-2", helpful=False, query="calendar sync outlook"),
        Feedback.vote("T-3", helpful=False, query="calendar sync outlook", correction="Re-link the account."),
        Feedback.vote("T-4", helpful=True, query="calendar sync outlook"),
    ])
    assert added == 2
    assert gaps.suggestions()[0].occurrences == 2


def test_pipeline_observe_query_uses_chunk_priority() -> None:
    pipe = CurationPipeline()
    weak = Chunk(id="x", ticket_id="T", kind="summary", text="t", metadata={"curation_priority": 0.1})
    good = Chunk(id="y", ticket_id="T", kind="summary", text="t", metadata={"curation_priority": 0.9})
    assert pipe.observe_query("vpn drops every hour", [weak]) is True
    assert pipe.observe_query("vpn drops every hour", [weak, good]) is False
    assert pipe.observe_query("vpn drops every hour", []) is True


def test_metrics() -> None:
    scores = [
        ResolutionScore(ticket_id="a", score=0.9, confidence=1.0, components={"solved": 1, "csat": 0.8}),
        ResolutionScore(ticket_id="b", score=0.5, confidence=0.5, components={"solved": 1}),
        ResolutionScore(ticket_id="c", score=0.1, confidence=0.4, components={"solved": 0}),
        ResolutionScore(ticket_id="d", score=0.0, confidence=0.0, has_answer=False),
    ]
    fb = [Feedback.vote("a", helpful=True), Feedback.vote("b", helpful=False, correction="Do X instead.")]
    gaps = GapDetector(GapConfig(min_occurrences=1))
    gaps.observe("missing article topic", None)
    m = compute_metrics(scores, fb, gaps)
    assert m.answers == 4 and m.answered == 3
    assert m.mean_score == round((0.9 + 0.5 + 0.1) / 3, 4)
    assert m.median_score == 0.5
    assert m.share_high == round(1 / 3, 4) and m.share_low == round(1 / 3, 4)
    assert m.distribution == {"0.0-0.2": 1, "0.2-0.4": 0, "0.4-0.6": 1, "0.6-0.8": 0, "0.8-1.0": 1}
    assert m.signal_coverage == {"csat": round(1 / 3, 4), "solved": 1.0}
    assert m.feedback_count == 2 and m.helpful_rate == 0.5 and m.corrections == 1
    assert m.gaps == 1 and m.gap_clusters == 1 and m.suggested_articles == 1


def test_metrics_empty() -> None:
    m = compute_metrics([])
    assert m.answers == 0 and m.mean_score == 0.0 and m.helpful_rate is None
