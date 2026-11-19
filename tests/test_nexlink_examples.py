"""The NexLink examples run offline: the Support Engine calls nexdash and answers simulated NexLink calls."""

import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "nexlink"
sys.path.insert(0, str(EXAMPLES))

demo_setup = importlib.import_module("demo_setup")
dashboard_example = importlib.import_module("support_asks_dashboard")
ingest_example = importlib.import_module("nexhub_triggers_ingest")
metrics_example = importlib.import_module("nexcoder_reads_metrics")

REAL = [False]
try:
    import nexdash.engine  # noqa: F401

    REAL.append(True)
except ImportError:
    pass


@pytest.mark.parametrize("real", REAL)
def test_support_engine_calls_nexdash_and_answers_a_nexlink_call(real: bool) -> None:
    demo = demo_setup.build_demo(prefer_real_nexdash=real)
    assert ("stub" in demo.nexdash_label) is (not real)
    result = metrics_example.run(demo)
    assert result["status"] == "ok"
    assert result["tickets"] == 3 and result["by_status"] == {"open": 1, "solved": 2}
    assert result["report_title"] == "Support metrics"
    assert {"volume", "csat"} <= set(result["report_panels"])
    assert result["dashboard_chars"] > 0


@pytest.mark.parametrize("real", REAL)
def test_support_engine_asks_for_a_dashboard(real: bool, tmp_path) -> None:
    transport, _ = demo_setup.report_transport(prefer_real=real)
    result = dashboard_example.run(demo_setup.NexDashClient(transport))
    assert "support_metrics" in result["models"]
    assert result["dashboard_html"].lower().startswith("<!doctype html>")
    assert result["invalid_period_refused_by"] == "ValidationError"
    args = ["--out", str(tmp_path)] + ([] if real else ["--stub"])
    assert dashboard_example.main(args) == 0
    assert (tmp_path / "support_metrics.html").is_file()


def test_nexhub_triggers_ingest() -> None:
    demo = demo_setup.build_demo(prefer_real_nexdash=False, tickets=[])
    result = ingest_example.run(demo)
    assert result["actions"] == ["nexlink.manifest", "support.ingest", "support.metrics"]
    assert result["status"] == "ok" and result["result"]["ingested"] == 3
    assert result["result"]["stats"]["pii_masked"]["EMAIL"] == 2
    assert result["nexcoder_ingest"] == "forbidden"
    text = str(result)
    assert "@example.com" not in text and "invoice export" not in text.lower()
    assert all("example.com" not in t.thread_text() for t in demo.workspace.tickets.values())


def test_examples_main_print_json(capsys) -> None:
    assert ingest_example.main() == 0
    assert metrics_example.main(["--stub"]) == 0
    out = capsys.readouterr().out
    assert '"status": "ok"' in out and "@example.com" not in out
