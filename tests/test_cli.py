"""Tests for the support-engine command line interface."""

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from support_engine.cli import main
from support_engine.storage import TicketStore


def test_cli_no_args_prints_help() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main([])
    assert exit_code == 0
    assert "support-engine" in buf.getvalue()


def test_cli_help_flag() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "support-engine" in buf.getvalue()


def test_cli_import_and_list_and_show_and_stats(tmp_path: Path) -> None:
    db_file = tmp_path / "cli_test.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    fixture_csv = Path(__file__).parent / "fixtures" / "tickets_sample.csv"

    # Import
    buf_out = io.StringIO()
    with redirect_stdout(buf_out):
        code = main(["--db-url", db_url, "import", "--input", str(fixture_csv), "--mask-pii"])
    assert code == 0
    assert "Imported" in buf_out.getvalue()
    assert "(PII masked)" in buf_out.getvalue()

    # List
    buf_list = io.StringIO()
    with redirect_stdout(buf_list):
        code = main(["--db-url", db_url, "list"])
    assert code == 0
    output = buf_list.getvalue()
    assert "ID" in output
    assert "STATUS" in output

    # Stats
    buf_stats = io.StringIO()
    with redirect_stdout(buf_stats):
        code = main(["--db-url", db_url, "stats"])
    assert code == 0
    assert "Total tickets:" in buf_stats.getvalue()

    # Show first ticket
    with TicketStore(db_url) as store:
        tickets = store.list_tickets(limit=1)
        assert len(tickets) > 0
        first_id = tickets[0].id

    buf_show = io.StringIO()
    with redirect_stdout(buf_show):
        code = main(["--db-url", db_url, "show", "--id", first_id])
    assert code == 0
    assert f"Ticket ID:    {first_id}" in buf_show.getvalue()


def test_cli_show_not_found(tmp_path: Path) -> None:
    db_file = tmp_path / "cli_test_err.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        code = main(["--db-url", db_url, "show", "--id", "NONEXISTENT"])
    assert code == 1
    assert "not found" in buf_err.getvalue()


def test_cli_import_file_not_found(tmp_path: Path) -> None:
    db_file = tmp_path / "cli_test_err.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        code = main(["--db-url", db_url, "import", "--input", "does_not_exist.csv"])
    assert code == 1
    assert "not found" in buf_err.getvalue()
