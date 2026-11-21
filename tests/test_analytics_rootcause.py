"""Root cause: lift of features in a target group against a baseline."""

from analytics_data import make_ticket, trend_tickets

from support_engine.analytics import RootCauseAnalyzer, RootCauseConfig, change_features
from support_engine.analytics.rootcause import word_pairs


def test_trend_tickets_point_to_the_release() -> None:
    tickets = trend_tickets()
    analyzer = RootCauseAnalyzer()
    report = analyzer.analyze_where(tickets, lambda t: t.id.startswith("EXP-NOW"), name="export spike")
    assert report.name == "export spike" and report.target == 5 and report.baseline == len(tickets) - 5
    top = report.causes[0]
    assert top.feature == "version:2.3" and top.kind == "change"
    assert top.support == 5 and top.target_share == 1.0 and top.baseline_share == 0.0
    assert "tag:export" in top.related and "since version" in top.related
    assert "change:update" in top.related, "same tickets and lift: merged into the more specific feature"
    # Words every export ticket shares are not a cause of the spike.
    assert all(c.lift >= 1.5 for c in report.causes)
    assert sorted(top.ticket_ids) == [f"EXP-NOW-{i}" for i in range(5)]


def test_lift_formula_with_smoothing() -> None:
    target = [make_ticket(f"T{i}", "x", [], tags=["billing"]) for i in range(4)]
    baseline = [make_ticket(f"B{i}", "x", [], tags=["billing" if i < 2 else "other"]) for i in range(8)]
    report = RootCauseAnalyzer(RootCauseConfig(fields=["tag"], extra_keys=[], use_terms=False)).analyze(
        target, baseline
    )
    cause = report.causes[0]
    assert cause.feature == "tag:billing"
    assert cause.lift == round(((4 + 1) / (4 + 2)) / ((2 + 1) / (8 + 2)), 4)
    assert cause.baseline_share == 0.25


def test_min_support_min_lift_and_max_causes() -> None:
    target = [make_ticket("T1", "x", [], tags=["rare"]), make_ticket("T2", "x", [], tags=["common"])]
    baseline = [make_ticket("B1", "x", [], tags=["common"])]
    cfg = RootCauseConfig(fields=["tag"], extra_keys=[], use_terms=False, use_bigrams=False)
    assert RootCauseAnalyzer(cfg).analyze(target, baseline).causes == []
    loose = cfg.model_copy(update={"min_support": 1, "min_lift": 1.0, "max_causes": 1})
    causes = RootCauseAnalyzer(loose).analyze(target, baseline).causes
    assert len(causes) == 1 and causes[0].feature == "tag:rare"


def test_target_tickets_are_removed_from_the_baseline() -> None:
    tickets = [make_ticket(f"T{i}", "Sync conflict", ["sync conflict"]) for i in range(3)]
    report = RootCauseAnalyzer().analyze(tickets[:2], tickets)
    assert report.baseline == 1
    assert RootCauseAnalyzer().analyze([], tickets).causes == []


def test_change_mentions_in_three_languages() -> None:
    assert change_features("It broke after the update") == {"change:update"}
    assert change_features("Parou depois da atualiza\u00e7\u00e3o de ontem") == {"change:update"}
    assert change_features("Falla desde la \u00faltima actualizaci\u00f3n") == {"change:update"}
    assert change_features("Since version 4.1.2 the report is empty") == {"version:4.1.2", "change:update"}
    assert change_features("works on v2.0") == {"version:2.0"}
    assert change_features("nothing changed") == set()


def test_word_pairs_stay_inside_sentences_and_skip_stopwords() -> None:
    pairs = word_pairs("The invoice export fails. Error banner shows with details")
    assert "invoice export" in pairs and "export fails" in pairs
    assert "fails error" not in pairs, "pairs never cross a sentence"
    assert "shows details" not in pairs, "pairs never jump over a stopword"


def test_features_never_carry_masked_placeholders() -> None:
    ticket = make_ticket("M-1", "Write to [EMAIL_1] about [NAME_1]", ["call [PHONE_1] now"])
    features = RootCauseAnalyzer().features(ticket)
    assert not any("email" in f or "name" in f or "phone" in f or "[" in f for f in features)


def test_extra_keys_become_features() -> None:
    ticket = make_ticket("E-1", "x", [], extra={"environment": "Production", "secret_note": "hidden"})
    features = RootCauseAnalyzer().features(ticket)
    assert "environment:production" in features
    assert not any(f.startswith("secret_note") for f in features)
