"""support-engine curate command."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from support_engine.cli import main
from support_engine.curation import FeedbackStore

FIXTURE = Path(__file__).parent / "fixtures" / "tickets_sample.json"


def run(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def write_feedback(path: Path) -> None:
    records = [
        {"ticket_id": "JSON-201", "rating": 0.0, "correction": "Set clock drift tolerance, mail ops@example.com",
         "created_at": "2026-03-01T10:00:00Z"},
        {"ticket_id": "JSON-201", "rating": 0.0, "query": "saml loop again", "created_at": "2026-03-01T11:00:00Z"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_curate_writes_chunks_and_report(tmp_path: Path) -> None:
    fb = tmp_path / "feedback.jsonl"
    write_feedback(fb)
    chunks_out, report_out = tmp_path / "chunks.jsonl", tmp_path / "report.json"
    code, out, _ = run([
        "curate", "-i", str(FIXTURE), "--feedback", str(fb), "--chunks-out", str(chunks_out),
        "--report-out", str(report_out), "--gap-min-occurrences", "1",
    ])
    assert code == 0
    assert "Curated 1 ticket(s)" in out and "1 correction(s) added" in out and "1 article(s) suggested" in out
    chunks = [json.loads(line) for line in chunks_out.read_text(encoding="utf-8").splitlines()]
    assert any(c["kind"] == "correction" for c in chunks)
    assert all("resolution_score" in c["metadata"] for c in chunks)
    body = chunks_out.read_text(encoding="utf-8") + report_out.read_text(encoding="utf-8")
    assert "example.com" not in body
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["metrics"]["feedback_count"] == 2 and len(report["suggestions"]) == 1


def test_curate_saves_and_reuses_stored_feedback(tmp_path: Path) -> None:
    db = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
    fb = tmp_path / "feedback.json"
    fb.write_text(json.dumps([{"ticket_id": "JSON-201", "rating": 1.0, "created_at": "2026-03-01T10:00:00Z"}]),
                  encoding="utf-8")
    code, _, _ = run(["--db-url", db, "curate", "-i", str(FIXTURE), "--feedback", str(fb), "--save-feedback"])
    assert code == 0
    with FeedbackStore(db, env_file=None) as store:
        assert store.count() == 1
    code, out, _ = run(["--db-url", db, "curate", "-i", str(FIXTURE), "--with-stored-feedback"])
    assert code == 0 and "Feedback 1 " in out


def test_curate_missing_input_and_bad_feedback(tmp_path: Path) -> None:
    code, _, err = run(["curate", "-i", str(tmp_path / "nope.json")])
    assert code == 1 and "not found" in err
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ticket_id": "x", "rating": 7, "correction": "mail ops@example.com"}', encoding="utf-8")
    code, _, err = run(["curate", "-i", str(FIXTURE), "--feedback", str(bad)])
    assert code == 2 and "example.com" not in err
