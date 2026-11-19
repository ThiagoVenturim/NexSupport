"""Analytics pipeline end to end over ingested tickets, the LLM backend with a fake model, and the CLI."""

import io
import json
import logging
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path

from analytics_data import NOW, mixed_tickets, trend_tickets
from nexcoder_core.llm.fake import FakeChatModel

from support_engine.analytics import AnalyticsConfig, AnalyticsPipeline, Outcome, TrendConfig, TrendDetector, analyze
from support_engine.cli import main
from support_engine.ingestion import IngestionPipeline

PII = ("ana.souza@example.com", "Ana Souza", "Souza")


def tickets():
    return mixed_tickets() + trend_tickets()


def test_report_over_ingested_tickets(caplog) -> None:
    ingested = IngestionPipeline().process(tickets())
    results = [Outcome(action="answer", level=1), Outcome(action="escalate", level=3)]
    with caplog.at_level(logging.DEBUG):
        report = AnalyticsPipeline().run(ingested.tickets, results, now=NOW)
    assert report.tickets == len(tickets()) and report.generated_at == NOW
    assert report.sentiment.negative >= 2 and set(report.sentiment.by_language) >= {"en", "pt", "es"}
    assert [a.status for a in report.alerts] == ["spike"]
    assert {r.ticket_id for r in report.high_risk()} >= {"R-1", "R-2"}
    assert report.quality is not None and report.quality.overall.tickets == report.tickets
    names = [r.name for r in report.root_causes]
    assert names[0] == "negative_or_high_risk" and names[1] == f"trend:{report.alerts[0].cluster_id}"
    trend_cause = report.root_causes[1].causes[0]
    assert trend_cause.feature == "version:2.3"
    assert report.roi is not None and report.roi.deflection_rate == 0.5
    dump = report.model_dump_json()
    for value in PII:
        assert value not in dump and value not in caplog.text
    assert "analytics:" in caplog.text


def test_raw_tickets_are_masked_by_the_pipeline() -> None:
    report = AnalyticsPipeline().run(tickets(), now=NOW)
    dump = report.model_dump_json()
    assert not any(value in dump for value in PII)
    assert not any("souza" in c.feature or "ana" == c.feature for r in report.root_causes for c in r.causes)


def test_no_agent_results_means_no_roi_and_default_now() -> None:
    report = analyze(mixed_tickets())
    assert report.roi is None
    assert report.generated_at is not None and report.generated_at <= NOW


def test_root_cause_targets_can_be_turned_off() -> None:
    config = AnalyticsConfig()
    config.root_cause.negative_target = False
    config.root_cause.trend_targets = False
    assert AnalyticsPipeline(config).run(tickets(), now=NOW).root_causes == []


def test_long_lived_trend_detector_keeps_history_between_runs() -> None:
    detector = TrendDetector(TrendConfig())
    rows = trend_tickets()
    history = [t for t in rows if t.created_at < NOW - timedelta(hours=12)]
    today = [t for t in rows if t.created_at >= NOW - timedelta(hours=12)]
    pipeline = AnalyticsPipeline()
    pipeline.run(history, now=NOW - timedelta(days=1), trends=detector)
    report = pipeline.run(today, now=NOW, trends=detector)
    assert [a.status for a in report.alerts] == ["spike"]
    assert report.alerts[0].series[:-1] == [1] * 7


def test_llm_sentiment_backend_with_a_fake_model() -> None:
    llm = FakeChatModel(responses=['{"score": -0.9, "language": "en"}'])
    config = AnalyticsConfig(sentiment_backend="llm")
    report = AnalyticsPipeline(config, llm=llm).run(mixed_tickets(), now=NOW)
    assert llm.calls > 0
    assert report.sentiment.negative == report.sentiment.tickets


def run(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def write_tickets(path: Path) -> None:
    records = [t.model_dump(mode="json") for t in tickets()]
    path.write_text(json.dumps({"tickets": records}), encoding="utf-8")


def test_cli_analyze_writes_the_report(tmp_path: Path) -> None:
    source, results, report_out = tmp_path / "tickets.json", tmp_path / "results.jsonl", tmp_path / "report.json"
    write_tickets(source)
    results.write_text(
        "\n".join(json.dumps(r) for r in [{"action": "answer", "level": 1, "attempts": 1}, {"action": "review"}]),
        encoding="utf-8",
    )
    code, out, err = run([
        "analyze", "-i", str(source), "--agent-results", str(results), "--report-out", str(report_out),
        "--now", NOW.isoformat(), "--currency", "EUR",
    ])
    assert code == 0, err
    assert "1 alert(s)" in out and "Deflection 50% of 2" in out and "EUR" in out
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["alerts"][0]["status"] == "spike" and report["roi"]["currency"] == "EUR"
    raw = report_out.read_text(encoding="utf-8")
    assert not any(value in raw for value in PII)


def test_cli_analyze_errors(tmp_path: Path) -> None:
    assert run(["analyze", "-i", str(tmp_path / "missing.json")])[0] == 1
    source = tmp_path / "tickets.json"
    write_tickets(source)
    assert run(["analyze", "-i", str(source), "--agent-results", str(tmp_path / "none.jsonl")])[0] == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"level": 1}', encoding="utf-8")
    code, _, err = run(["analyze", "-i", str(source), "--agent-results", str(bad)])
    assert code == 2 and "ValidationError" in err
    assert run(["analyze", "-i", str(source), "--window-hours", "0"])[0] == 2
