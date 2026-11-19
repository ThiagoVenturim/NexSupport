"""Resolution score: signals, weights, missing signals, judge and feedback."""

from datetime import UTC, datetime, timedelta

import pytest

from support_engine.curation import Feedback, LLMJudge, ScoreWeights, ScoringConfig, answer_comment, score_ticket
from support_engine.models import Comment, Ticket, TicketStatus

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def make_ticket(**kwargs) -> Ticket:
    comments = kwargs.pop(
        "comments",
        [
            Comment(id="c1", author_role="customer", body="Export fails with error 500.", created_at=T0),
            Comment(id="c2", author_role="agent", body="Clear the cache and retry the export.",
                    created_at=T0 + timedelta(hours=1)),
        ],
    )
    base = {"id": "T-1", "source": "csv", "subject": "Export fails", "created_at": T0, "comments": comments}
    base.update(kwargs)
    return Ticket(**base)


def test_solved_no_reopen_full_csat_scores_one() -> None:
    s = score_ticket(make_ticket(status=TicketStatus.SOLVED, csat=1.0))
    assert s.score == pytest.approx(1.0)
    assert s.components == {"solved": 1.0, "no_reopen": 1.0, "csat": 1.0}
    w = ScoreWeights()
    assert s.confidence == pytest.approx((w.solved + w.no_reopen + w.csat) / w.total())


def test_missing_signals_are_renormalised() -> None:
    # Only status and reopen signals: solved=1, no_reopen=0.5 with one reopen.
    s = score_ticket(make_ticket(status=TicketStatus.SOLVED, reopen_count=1))
    w = ScoreWeights()
    expected = (w.solved * 1.0 + w.no_reopen * 0.5) / (w.solved + w.no_reopen)
    assert s.score == pytest.approx(expected, abs=1e-6)
    assert "csat" not in s.components


def test_reopens_and_open_status_lower_the_score() -> None:
    good = score_ticket(make_ticket(status=TicketStatus.SOLVED))
    reopened = score_ticket(make_ticket(status=TicketStatus.SOLVED, reopen_count=3))
    open_ = score_ticket(make_ticket(status=TicketStatus.OPEN))
    pending = score_ticket(make_ticket(status=TicketStatus.PENDING))
    assert reopened.components["no_reopen"] == 0.0
    assert good.score > reopened.score
    assert good.score > pending.score > open_.score


def test_no_agent_answer_scores_zero() -> None:
    t = make_ticket(status=TicketStatus.SOLVED, comments=[
        Comment(id="c1", author_role="customer", body="Never mind.", created_at=T0)])
    s = score_ticket(t)
    assert s.score == 0.0 and not s.has_answer and s.confidence == 0.0


def test_require_answer_can_be_disabled() -> None:
    t = make_ticket(status=TicketStatus.SOLVED, comments=[])
    s = score_ticket(t, ScoringConfig(require_answer=False))
    assert s.score == pytest.approx(1.0) and not s.has_answer


def test_answer_is_last_public_agent_reply() -> None:
    t = make_ticket(comments=[
        Comment(id="a1", author_role="agent", body="First try.", created_at=T0),
        Comment(id="a2", author_role="agent", body="Internal note.", created_at=T0 + timedelta(hours=2), public=False),
        Comment(id="a3", author_role="agent", body="Final fix.", created_at=T0 + timedelta(hours=1)),
    ])
    assert answer_comment(t).id == "a3"


def test_judge_signal_is_used_and_clamped() -> None:
    calls = []

    def judge(problem: str, answer: str) -> float:
        calls.append((problem, answer))
        return 1.7

    s = score_ticket(make_ticket(status=TicketStatus.OPEN), judge=judge)
    assert s.components["judge"] == 1.0
    assert calls[0][1] == "Clear the cache and retry the export."
    assert "Export fails" in calls[0][0]


def test_judge_returning_none_drops_the_signal() -> None:
    s = score_ticket(make_ticket(status=TicketStatus.SOLVED), judge=lambda p, a: None)
    assert "judge" not in s.components


def test_feedback_moves_the_score_with_a_prior() -> None:
    t = make_ticket(status=TicketStatus.SOLVED, csat=1.0)
    bad = [Feedback.vote("T-1", helpful=False) for _ in range(4)]
    s = score_ticket(t, feedback=bad)
    # Bayesian mean with prior 0.5 x 2 votes: (1 + 0) / 6.
    assert s.components["feedback"] == pytest.approx(1 / 6, abs=1e-6)
    assert s.score < score_ticket(t).score
    assert s.feedback_count == 4


def test_feedback_for_other_tickets_is_ignored() -> None:
    s = score_ticket(make_ticket(), feedback=[Feedback.vote("OTHER", helpful=False)])
    assert "feedback" not in s.components and s.feedback_count == 0


def test_weights_are_parametrisable() -> None:
    cfg = ScoringConfig(weights=ScoreWeights(solved=0, no_reopen=0, csat=1, judge=0, feedback=0))
    s = score_ticket(make_ticket(status=TicketStatus.OPEN, csat=0.4), cfg)
    assert s.score == pytest.approx(0.4)


def test_all_zero_weights_rejected() -> None:
    with pytest.raises(ValueError):
        ScoreWeights(solved=0, no_reopen=0, csat=0, judge=0, feedback=0)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def ainvoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return self.reply


def test_llm_judge_offline_with_fake_model() -> None:
    llm = FakeLLM('{"score": 0.8, "reason": "Actionable."}')
    judge = LLMJudge(llm=llm)
    assert judge("Export fails", "Clear the cache.") == pytest.approx(0.8)
    assert "Clear the cache." in llm.prompts[0]


def test_llm_judge_failure_returns_none() -> None:
    judge = LLMJudge(llm=FakeLLM("not json at all"))
    assert judge("p", "a") is None
