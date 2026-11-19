"""support-engine serve and mcp commands. Offline: no server is started, no model is called."""

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from copilot_data import TOKEN

from support_engine.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "tickets_sample.json"


def run(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def test_serve_refuses_to_start_without_token(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SUPPORT_ENGINE_API_TOKEN", raising=False)
    code, _, err = run(["serve", "--input", str(FIXTURE), "--env-file", str(tmp_path / "none.env")])
    assert code == 2 and "SUPPORT_ENGINE_API_TOKEN" in err


def test_serve_uses_token_from_env_file(tmp_path, monkeypatch) -> None:
    import uvicorn

    monkeypatch.delenv("SUPPORT_ENGINE_API_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"SUPPORT_ENGINE_API_TOKEN={TOKEN}\n", encoding="utf-8")
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(app=app, **kw))
    code, out, err = run(["serve", "--input", str(FIXTURE), "--env-file", str(env_file), "--port", "8123"])
    assert code == 0 and started["host"] == "127.0.0.1" and started["port"] == 8123
    assert TOKEN not in out + err

    from fastapi.testclient import TestClient

    c = TestClient(started["app"])
    assert c.get("/health").status_code == 200
    assert c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401


def test_serve_missing_input(monkeypatch) -> None:
    monkeypatch.setenv("SUPPORT_ENGINE_API_TOKEN", TOKEN)
    code, _, err = run(["serve", "--input", "missing.json"])
    assert code == 1 and "not found" in err


def test_mcp_over_stdio(monkeypatch) -> None:
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n"))
    code, out, _ = run(["mcp", "--input", str(FIXTURE)])
    assert code == 0
    replies = [json.loads(line) for line in out.splitlines()]
    assert replies[0]["result"]["protocolVersion"] == "2025-03-26"
    assert len(replies[1]["result"]["tools"]) == 4
