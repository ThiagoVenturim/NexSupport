"""Live trends: clustering, baseline, spikes, new topics and incremental updates."""

from datetime import timedelta

from analytics_data import NOW, make_ticket, trend_tickets

from support_engine.analytics import TrendConfig, TrendDetector, detect_trends


def by_keyword(trends, word):
    return next(t for t in trends if word in t.keywords)


def test_spike_against_a_quiet_baseline() -> None:
    trends = detect_trends(trend_tickets(), now=NOW)
    export = by_keyword(trends, "export")
    sso = by_keyword(trends, "sso")
    assert export.series == [1, 1, 1, 1, 1, 1, 1, 5]
    assert export.status == "spike" and export.alert
    assert export.baseline_mean == 1.0 and export.ratio == 5.0 and export.z_score == 4.0
    assert sorted(export.ticket_ids) == [f"EXP-NOW-{i}" for i in range(5)]
    assert sso.status == "stable" and not sso.alert
    assert trends[0].cluster_id == export.cluster_id, "alerts come first"
    assert export.window_start <= NOW < export.window_end


def test_new_topic_alerts_once_it_reaches_min_count() -> None:
    tickets = [
        make_ticket(f"N-{i}", "Webhook signature mismatch", ["signature mismatch"], created_at=NOW - timedelta(hours=i))
        for i in range(3)
    ]
    assert detect_trends(tickets, now=NOW)[0].status == "new"
    assert detect_trends(tickets[:2], now=NOW)[0].status == "rising"
    assert detect_trends(tickets[:2], TrendConfig(min_count=2), now=NOW)[0].alert


def test_falling_topic() -> None:
    tickets = [
        make_ticket(f"F-{d}-{i}", "Mobile app crash", ["app crash"], created_at=NOW - timedelta(days=d, hours=i))
        for d in range(1, 4)
        for i in range(3)
    ]
    trend = detect_trends(tickets, TrendConfig(baseline_windows=3), now=NOW)[0]
    assert trend.series == [3, 3, 3, 0] and trend.status == "falling" and not trend.alert


def test_window_size_is_configurable() -> None:
    tickets = trend_tickets()
    hourly = detect_trends(tickets, TrendConfig(window_hours=1, baseline_windows=24), now=NOW)
    assert all(len(t.series) == 25 for t in hourly)
    assert not any(t.alert for t in hourly)


def test_incremental_observe_is_order_independent_and_ignores_repeats() -> None:
    tickets = trend_tickets()
    detector = TrendDetector()
    for ticket in reversed(tickets):
        detector.observe(ticket)
    assert detector.observe(tickets[0]) is not None
    assert len(detector) == len(tickets)
    live = {t.cluster_id: t.series for t in detector.report(NOW)}
    batch = TrendDetector()
    batch.observe_many(tickets)
    assert sorted(live.values()) == sorted(t.series for t in batch.report(NOW))


def test_live_detector_updates_between_reports() -> None:
    detector = TrendDetector(TrendConfig(min_count=2))
    detector.observe(make_ticket("L-1", "Payment page timeout", ["payment page timeout"], created_at=NOW))
    assert detector.alerts(NOW) == []
    detector.observe(make_ticket("L-2", "Payment page timeout again", ["payment timeout"], created_at=NOW))
    alerts = detector.alerts(NOW)
    assert len(alerts) == 1 and alerts[0].current == 2
    assert detector.cluster_of("L-2") == alerts[0].cluster_id


def test_tags_and_product_join_topics_and_can_be_disabled() -> None:
    a = make_ticket("G-1", "Cannot authenticate", [], tags=["sso"], product="identity")
    b = make_ticket("G-2", "Sign in loop", [], tags=["sso"], product="identity")
    with_tags = TrendDetector()
    with_tags.observe_many([a, b])
    assert len(with_tags.clusters) == 1
    without = TrendDetector(TrendConfig(use_tags=False))
    without.observe_many([a, b])
    assert len(without.clusters) == 2


def test_report_holds_tokens_not_text() -> None:
    ticket = make_ticket("P-1", "Refund for [EMAIL_1] please", [], created_at=NOW)
    trend = detect_trends([ticket], now=NOW)[0]
    assert all("[" not in k and "email" not in k for k in trend.keywords)
    assert detect_trends([]) == []
    assert TrendDetector().observe(make_ticket("E", "", [])) is None
