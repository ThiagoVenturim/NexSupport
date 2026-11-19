"""Risk radar signals, levels and the service quality report."""

from datetime import timedelta

import pytest
from analytics_data import NOW, make_ticket, mixed_tickets

from support_engine.analytics import (
    QualityConfig,
    RiskConfig,
    RiskRadar,
    RiskWeights,
    SLAConfig,
    find_terms,
    first_response_hours,
    latest_timestamp,
    service_quality,
    sla_signal,
    ticket_sentiment,
    waiting_hours,
)
from support_engine.models import TicketStatus


def sentiments(tickets):
    return {t.id: ticket_sentiment(t) for t in tickets}


def test_radar_ranks_angry_customers_first_and_skips_solved() -> None:
    tickets = mixed_tickets()
    radar = RiskRadar().radar(tickets, sentiments(tickets), NOW)
    ids = [a.ticket_id for a in radar]
    assert "R-3" not in ids, "solved tickets are left out by default"
    assert ids[:2] == ["R-1", "R-2"] and ids[-1] == "R-4"
    r1, r2, r4 = radar[0], radar[1], radar[-1]
    assert r1.level == r2.level == "high" and r4.level == "low"
    assert set(r1.churn_terms) == {"refund", "cancel", "unacceptable"}
    assert r2.churn_terms == ["procon"] and r2.signals["reopens"] == 1.0
    assert {"churn_terms", "sla_breach", "negative_sentiment"} <= set(r1.reasons)
    assert r1.reasons[0] == "churn_terms", "strongest contribution first"


def test_include_resolved() -> None:
    tickets = mixed_tickets()
    radar = RiskRadar(RiskConfig(include_resolved=True)).radar(tickets, sentiments(tickets), NOW)
    solved = next(a for a in radar if a.ticket_id == "R-3")
    assert solved.level == "low" and solved.status == "solved"


def test_noisy_or_score_and_weights() -> None:
    ticket = make_ticket("W-1", "Help", ["hello"], created_at=NOW, reopen_count=1, priority="urgent")
    assessment = RiskRadar().assess(ticket, now=NOW)
    assert assessment.signals["reopens"] == 0.5 and assessment.signals["priority"] == 1.0
    expected = 1 - (1 - 0.4 * 0.5) * (1 - 0.2 * 1.0)
    assert assessment.score == pytest.approx(expected, abs=1e-4)
    muted = RiskRadar(RiskConfig(weights=RiskWeights(reopens=0.0, priority=0.0))).assess(ticket, now=NOW)
    assert muted.score == 0.0 and muted.level == "low"


def test_levels_follow_thresholds() -> None:
    radar = RiskRadar(RiskConfig(high=0.9, medium=0.1))
    assert [radar.level_for(x) for x in (0.05, 0.5, 0.95)] == ["low", "medium", "high"]
    with pytest.raises(ValueError):
        RiskConfig(high=0.3, medium=0.5)


def test_low_csat_and_sentiment_drop() -> None:
    ticket = make_ticket(
        "C-1", "Export", ["Thanks, great help", "This is terrible, still broken"], created_at=NOW, csat=0.2
    )
    a = RiskRadar().assess(ticket, ticket_sentiment(ticket), NOW + timedelta(hours=1))
    assert a.signals["low_csat"] == 1.0 and a.signals["sentiment_drop"] == 1.0
    assert "sentiment_drop" in a.reasons


def test_sla_signal_by_priority() -> None:
    sla = SLAConfig()
    urgent = make_ticket("S-1", "Down", ["down"], created_at=NOW - timedelta(hours=2), priority="urgent")
    assert sla_signal(urgent, NOW, sla) == 1.0, "no reply after the 1 hour first response target"
    replied = make_ticket(
        "S-2", "Down", ["down"], created_at=NOW - timedelta(hours=5), priority="urgent", agent=[(0.5, "On it")]
    )
    assert sla_signal(replied, NOW, sla) == pytest.approx(0.25), "5 of 8 resolution hours: rising from half"
    late_reply = make_ticket("S-3", "Down", ["down"], created_at=NOW - timedelta(hours=5), agent=[(9.5, "Sorry")])
    assert sla_signal(late_reply, NOW + timedelta(hours=5), sla) == 1.0
    solved_late = make_ticket(
        "S-4", "Slow", ["slow"], created_at=NOW - timedelta(days=5), agent=[(1, "Fixed")],
        status=TicketStatus.SOLVED, solved_at=NOW, priority="low",
    )
    assert sla_signal(solved_late, NOW, sla) == 0.0, "low priority allows 168 hours"
    assert sla_signal(solved_late, NOW, SLAConfig(resolution_hours={"low": 24.0})) == 1.0
    assert SLAConfig().resolution_target("unknown") == SLAConfig().resolution_target("normal")


def test_waiting_and_first_response() -> None:
    waiting = make_ticket("Q-1", "Help", ["help"], created_at=NOW - timedelta(hours=6), agent=[(1, "Reply")])
    assert waiting_hours(waiting, NOW) == 0.0, "the agent spoke last"
    assert first_response_hours(waiting) == 1.0
    customer_last = make_ticket(
        "Q-2", "Help", ["help", "anyone?"], created_at=NOW - timedelta(hours=6), customer_gap_hours=2, agent=[(1, "Hi")]
    )
    assert waiting_hours(customer_last, NOW) == 4.0
    assert first_response_hours(make_ticket("Q-3", "x", ["y"])) is None


def test_find_terms_ignores_case_accents_and_partial_words() -> None:
    assert find_terms("Vou CANCELAR e abrir reclama\u00e7\u00e3o no Reclame Aqui", ["cancelar", "reclame aqui"]) == [
        "cancelar",
        "reclame aqui",
    ]
    assert find_terms("the cancellation flow", ["cancel"]) == []
    assert find_terms("Quiero un reembolso", ["reembolso", "abogado"]) == ["reembolso"]


def test_service_quality_overall_and_per_product() -> None:
    tickets = mixed_tickets()
    scores = sentiments(tickets)
    radar = RiskRadar().radar(tickets, scores, NOW)
    report = service_quality(tickets, sentiments=scores, assessments=radar, now=NOW)
    assert report.group_by == "product"
    assert report.overall.tickets == 4 and report.overall.resolved == 1 and report.overall.high_risk == 2
    groups = {g.key: g for g in report.groups}
    assert set(groups) == {"admin", "billing", "reports"}
    billing = groups["billing"]
    assert billing.tickets == 2 and billing.resolved_rate == 0.5 and billing.mean_csat == 1.0
    assert billing.negative_share == 0.5
    assert groups["reports"].reopen_rate == 1.0 and groups["reports"].mean_first_response_hours == 9.0
    assert groups["admin"].sla_breach_rate == 0.0


def test_service_quality_other_groupings() -> None:
    tickets = mixed_tickets()
    by_language = service_quality(tickets, config=QualityConfig(group_by="language"), now=NOW)
    assert {g.key for g in by_language.groups} == {"es", "pt", "unknown"}
    by_tag = service_quality([make_ticket("X", "x", [], tags=["a", "b"])], config=QualityConfig(group_by="tag"))
    assert [g.key for g in by_tag.groups] == ["a", "b"]
    assert service_quality([]).overall.tickets == 0


def test_latest_timestamp() -> None:
    assert latest_timestamp(mixed_tickets()) <= NOW
    assert latest_timestamp([]) is None
