"""support-engine pipeline command."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from agent_data import tickets
from copilot_data import EMAIL, open_ticket

from support_engine.cli import main


def run(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def write_tickets(path: Path, items) -> Path:
    path.write_text(json.dumps({"tickets": [t.model_dump(mode="json") for t in items]}), encoding="utf-8")
    return path


def test_steps_and_presets_are_listed() -> None:
    code, out, _ = run(["pipeline", "steps"])
    assert code == 0
    for name in ("cleanup", "pii_mask", "chunking", "dedup", "curation", "index", "answer", "copilot", "analytics"):
        assert name in out
    code, out, _ = run(["pipeline", "presets"])
    assert code == 0
    assert "minimal: cleanup -> pii_mask -> chunking -> index -> answer" in out
    assert "full: cleanup -> pii_mask -> chunking -> dedup -> curation" in out


def test_show_prints_the_resolved_flow(tmp_path: Path) -> None:
    flow = tmp_path / "flow.yaml"
    flow.write_text("extends: full\nname: no-analytics\nsteps:\n  - use: analytics\n    enabled: false\n", "utf-8")
    code, out, _ = run(["pipeline", "show", "-f", str(flow)])
    assert code == 0
    assert "name: no-analytics" in out and "use: dedup" in out and "enabled: false" in out


def test_run_full_flow_writes_a_masked_report(tmp_path: Path) -> None:
    corpus = write_tickets(tmp_path / "tickets.json", tickets())
    open_file = write_tickets(tmp_path / "open.json", [open_ticket()])
    report = tmp_path / "out" / "report.json"
    code, out, err = run(
        [
            "pipeline", "run", "-f", "full", "-i", str(corpus), "--open-tickets", str(open_file),
            "-q", f"I am {EMAIL}: the invoice export fails with error 500", "--report-out", str(report),
        ]
    )  # fmt: skip
    assert code == 0, err
    assert "Flow 'full' completed" in out and "1 answer(s), 1 suggestion(s)" in out
    assert "curation" in out and "success" in out
    body = report.read_text(encoding="utf-8")
    assert EMAIL not in body and EMAIL not in out
    data = json.loads(body)
    assert data["answers"][0]["action"] == "answer"
    assert data["suggestions"][0]["ticket_id"] == "N-1"


def test_run_minimal_flow(tmp_path: Path) -> None:
    corpus = write_tickets(tmp_path / "tickets.json", tickets())
    code, out, _ = run(["pipeline", "run", "-f", "minimal", "-i", str(corpus), "-q", "invoice export error 500"])
    assert code == 0
    assert "answer     success: questions=1, review=1" in out


def test_errors(tmp_path: Path) -> None:
    corpus = write_tickets(tmp_path / "tickets.json", tickets())
    code, _, err = run(["pipeline", "run", "-f", "nope", "-i", str(corpus)])
    assert code == 1 and "not found" in err
    bad = tmp_path / "bad.yaml"
    bad.write_text("steps:\n  - use: chunking\n", encoding="utf-8")
    code, _, err = run(["pipeline", "run", "-f", str(bad), "-i", str(corpus)])
    assert code == 2 and "pii_mask" in err
    code, _, err = run(["pipeline", "run", "-f", "minimal", "-i", str(tmp_path / "missing.json")])
    assert code == 1 and "not found" in err
    code, _, err = run(["pipeline"])
    assert code == 2
