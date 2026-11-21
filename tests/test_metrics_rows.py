"""Per-ticket metric rows built from the analytics: no askdash needed."""

import json
import logging

import pytest
from analytics_data import NOW, mixed_tickets, trend_tickets

from support_engine.analytics import Outcome
from support_engine.metrics import SupportMetrics, build_support_metrics, split_outcomes
from support_engine.metrics.synthetic import FAKE_EMAIL, FAKE_PHONE, synthetic_outcomes, synthetic_tickets

PII = ("ana.souza@example.com", "Ana Souza", "Souza", FAKE_EMAIL, FAKE_PHONE)


def tickets():
    out = mixed_tickets() + trend_tickets()
    for t in out:
        t.extra.setdefault("queue", "billing" if t.product == "billing" else "general")
    return out


def by_id(rows, key="ticket_id"):
    return {r[key]: r for r in rows}


def test_insight_rows_carry_risk_trend_root_cause_and_roi() -> None:
    results = [
        {"ticket_id": "R-3", "action": "answer", "level": 1},
        {"ticket_id": "R-4", "action": "review", "level": 2, "attempts": 2},
        {"action": "escalate", "level": 3},
    ]
    metrics = build_support_metrics(tickets(), results, now=NOW)
    rows = by_id(metrics.insight_rows())
    assert len(rows) == len(tickets())

    assert rows["R-1"]["risk_level"] == "high" and rows["R-1"]["high_risk_flag"] == 1
    assert rows["R-1"]["risk_rank"] == 1 and rows["R-1"]["risk_reason"] != "none"
    assert rows["R-3"]["risk_level"] == "not_rated" and rows["R-3"]["risk_score"] is None
    assert rows["R-1"]["sentiment"] == "negative" and rows["R-1"]["negative_flag"] == 1
    assert rows["R-3"]["sentiment"] == "positive"

    spike = rows["EXP-NOW-0"]
    assert spike["trend_alert"] == "yes" and spike["trend_alert_flag"] == 1 and spike["trend_status"] == "spike"
    assert "export" in spike["trend_topic"]
    assert spike["root_cause"] == "version:2.3"
    assert rows["EXP-1"]["trend_alert_flag"] == 0

    deflected, assisted = rows["R-3"], rows["R-4"]
    assert deflected["deflected_flag"] == 1 and deflected["hours_saved"] == pytest.approx(12 / 60, abs=1e-3)
    assert assisted["assisted_flag"] == 1 and assisted["hours_saved"] == pytest.approx(8 / 60, abs=1e-3)
    assert assisted["llm_cost"] == pytest.approx(2 * 2 * 0.002)
    assert rows["R-1"]["agent_action"] == "none" and rows["R-1"]["hours_saved"] == 0.0
    # The untied escalation still counts in the overall ROI.
    assert metrics.report.roi is not None and metrics.report.roi.questions == 3
    assert rows["R-1"]["opened_month"] == "2026-05" and rows["R-1"]["opened_date"] == "2026-05-18"
    assert rows["R-2"]["reopened_flag"] == 1


def test_per_ticket_savings_add_up_to_the_report_roi() -> None:
    data = synthetic_tickets(40)
    metrics = build_support_metrics(data, synthetic_outcomes(data))
    rows = metrics.insight_rows()
    roi = metrics.report.roi
    assert roi is not None and roi.questions == len(metrics.outcomes)
    assert sum(r["hours_saved"] for r in rows) == pytest.approx(roi.hours_saved, abs=1e-3)
    assert sum(r["net_savings"] for r in rows) == pytest.approx(roi.net_savings, abs=1e-2)
    assert sum(r["deflected_flag"] for r in rows) == roi.answered


def test_team_falls_back_to_queue_then_team_of() -> None:
    data = mixed_tickets()
    data[0].extra = {"team": "tier2", "queue": "billing"}
    data[1].extra = {"queue": "reports-queue"}
    data[2].extra = {}
    data[3].extra = {}
    data[3].product = None
    metrics = build_support_metrics(data, now=NOW, team_of={"billing": "finance"})
    rows = by_id(metrics.insight_rows())
    assert (rows["R-1"]["team"], rows["R-1"]["queue"]) == ("tier2", "billing")
    assert (rows["R-2"]["team"], rows["R-2"]["queue"]) == ("reports-queue", "reports-queue")
    assert rows["R-3"]["team"] == "finance" and rows["R-3"]["queue"] == "unassigned"
    assert rows["R-4"]["team"] == "unassigned" and rows["R-4"]["project"] == "unassigned"
    callable_team = build_support_metrics(data, now=NOW, team_of=lambda product: f"team-{product}")
    assert by_id(callable_team.insight_rows())["R-3"]["team"] == "team-billing"


def test_conversation_records_hold_no_text() -> None:
    results = {"R-3": Outcome(action="answer", level=1), "R-4": {"action": "escalate", "level": 3}}
    metrics = build_support_metrics(tickets(), results, now=NOW)
    records = {r["id"]: r for r in metrics.conversation_records()}
    for record in records.values():
        assert not {"subject", "description"} & set(record)
        assert all(set(c) == {"author_role", "public", "created_at"} for c in record["comments"])
        assert set(record["extra"]) <= {"team", "channel", "sentiment", "sentiment_score", "deflected"}
    assert records["R-3"]["extra"]["deflected"] is True
    assert records["R-4"]["extra"]["deflected"] is False
    assert "deflected" not in records["R-1"]["extra"]
    assert records["R-1"]["extra"]["sentiment"] == "negative" and records["R-1"]["extra"]["sentiment_score"] < 0


def test_personal_data_stays_out_of_rows_and_logs(caplog) -> None:
    data = tickets() + synthetic_tickets(20)
    with caplog.at_level(logging.DEBUG):
        metrics = build_support_metrics(data, synthetic_outcomes(data), now=NOW)
        dump = json.dumps([metrics.insight_rows(), metrics.conversation_records()], default=str)
    for value in PII:
        assert value not in dump
        assert value not in caplog.text
    assert "metrics:" in caplog.text


def test_split_outcomes_accepts_mappings_lists_and_objects() -> None:
    class Result:
        def __init__(self, ticket_id):
            self.ticket_id = ticket_id
            self.action = "answer"
            self.level = 1
            self.cached = True
            self.attempts = 1

    everything, tied = split_outcomes([Result("A"), {"action": "refuse"}, Outcome(action="review", level=2)])
    assert len(everything) == 3 and set(tied) == {"A"} and tied["A"].cached
    everything, tied = split_outcomes({"B": {"action": "escalate", "level": 3}})
    assert len(everything) == 1 and tied["B"].action == "escalate"
    assert split_outcomes(None) == ([], {})


def test_without_ingestion_and_without_outcomes() -> None:
    metrics = build_support_metrics(tickets(), now=NOW, ingest=False)
    assert isinstance(metrics, SupportMetrics) and metrics.report.roi is None
    rows = metrics.insight_rows()
    assert all(r["agent_action"] == "none" and r["deflected_flag"] == 0 for r in rows)
    assert metrics.now == NOW
