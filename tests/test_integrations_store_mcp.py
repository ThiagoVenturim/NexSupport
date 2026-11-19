"""Token storage rules, the integration MCP tools, the MCP client and the CLI commands. Offline."""

from __future__ import annotations

import asyncio
import io
import json
import re
import subprocess
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from copilot_data import EMAIL, make_copilot, open_ticket
from integrations_data import ACCESS_1, CLIENT_SECRET, CODE, REFRESH_1, SECRETS, Clock, TokenServer, env_for

from support_engine import cli
from support_engine.copilot import CopilotService
from support_engine.integrations import (
    EnvFileTokenStore,
    IntegrationsMCPServer,
    IntegrationsService,
    MCPError,
    MCPToolClient,
    OAuthError,
    OAuthToken,
    load_client,
)
from support_engine.integrations import providers as providers_module
from support_engine.models import TicketStatus
from support_engine.storage import TicketStore

ROOT = Path(__file__).resolve().parents[1]


# Token storage ----------------------------------------------------------------------------------


def test_env_file_round_trip_keeps_other_lines(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# local settings\nDB_URL=sqlite:///x.db\nSUPPORT_ENGINE_OAUTH_ZENDESK_ACCESS_TOKEN=old-value-0123456789\n",
        encoding="utf-8",
    )
    store = EnvFileTokenStore(env_file, environ={})
    store.save("zendesk", OAuthToken(ACCESS_1, REFRESH_1, expires_at=1_800_000_000.5, scopes=("tickets:read",)))
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["# local settings", "DB_URL=sqlite:///x.db"]
    assert lines.count(f"SUPPORT_ENGINE_OAUTH_ZENDESK_ACCESS_TOKEN={ACCESS_1}") == 1
    assert "old-value-0123456789" not in env_file.read_text(encoding="utf-8")
    loaded = store.load("zendesk")
    assert (loaded.access_token, loaded.refresh_token, loaded.expires_at) == (ACCESS_1, REFRESH_1, 1_800_000_000)
    assert loaded.scopes == ("tickets:read",)
    assert store.load("intercom") is None
    assert not list(tmp_path.glob("*.tmp"))


def test_environment_wins_over_the_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SUPPORT_ENGINE_OAUTH_INTERCOM_ACCESS_TOKEN=from-file-0123456789\n", encoding="utf-8")
    store = EnvFileTokenStore(env_file, environ={"SUPPORT_ENGINE_OAUTH_INTERCOM_ACCESS_TOKEN": "from-env-0123456789"})
    assert store.load("intercom").access_token == "from-env-0123456789"


@pytest.mark.parametrize("name", ["tokens.txt", "settings.json", ".env.example", "config.env"])
def test_tokens_only_in_env_files(tmp_path, name) -> None:
    with pytest.raises(OAuthError, match=".env"):
        EnvFileTokenStore(tmp_path / name)


@pytest.mark.parametrize("bad", ["two\nlines-0123456789", "with space 0123456789", "hash#0123456789", 'q"uote'])
def test_unsafe_token_values_are_refused(tmp_path, bad) -> None:
    store = EnvFileTokenStore(tmp_path / ".env", environ={})
    with pytest.raises(OAuthError) as info:
        store.save("zendesk", OAuthToken(bad))
    assert bad not in str(info.value)


def test_invalid_expiry(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SUPPORT_ENGINE_OAUTH_ZENDESK_ACCESS_TOKEN=abc0123456789\nSUPPORT_ENGINE_OAUTH_ZENDESK_EXPIRES_AT=later\n",
        encoding="utf-8",
    )
    with pytest.raises(OAuthError, match="UNIX time"):
        EnvFileTokenStore(env_file, environ={}).load("zendesk")


def test_load_client_reads_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in env_for("zendesk").items()) + "\n", encoding="utf-8")
    client = load_client("zendesk", environ={}, env_file=env_file)
    assert client.client_secret == CLIENT_SECRET
    assert CLIENT_SECRET not in repr(client)


def test_repo_keeps_secrets_out() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gitignore
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if line.startswith("SUPPORT_ENGINE_OAUTH_") and line.split("=", 1)[0].endswith(
            ("_SECRET", "_ACCESS_TOKEN", "_REFRESH_TOKEN")
        ):
            assert line.split("=", 1)[1] == "", line
    assert "SUPPORT_ENGINE_OAUTH_ZENDESK_CLIENT_SECRET=" in example
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    if tracked:
        assert not any(Path(p).name == ".env" for p in tracked.splitlines())


# MCP tools ------------------------------------------------------------------------------------


@pytest.fixture
def ticket_db(tmp_path):
    with TicketStore(target=f"sqlite:///{tmp_path / 'tickets.db'}") as store:
        base = open_ticket(tid="Z-1")
        store.save_ticket(base.model_copy(update={"source": "zendesk"}))
        later = base.created_at + timedelta(days=1)
        store.save_ticket(
            open_ticket(tid="J-1").model_copy(
                update={"source": "jira_sm", "created_at": later, "status": TicketStatus.SOLVED}
            )
        )
        yield store


def server_for(store, with_copilot=True):
    service = CopilotService(make_copilot()[0]) if with_copilot else None
    return IntegrationsMCPServer(IntegrationsService(store, service))


def tool(srv, name, arguments=None, msg_id=1):
    message = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    return asyncio.run(srv.handle(message))["result"]


def test_tools_list_and_initialize(ticket_db) -> None:
    srv = server_for(ticket_db, with_copilot=False)
    init = asyncio.run(srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))["result"]
    assert init["serverInfo"]["name"] == "support-engine-integrations"
    tools = asyncio.run(srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_tickets", "suggest_reply", "support_metrics"]
    assert all("inputSchema" in t for t in tools)


def test_list_tickets_filters_and_masks(ticket_db) -> None:
    srv = server_for(ticket_db, with_copilot=False)
    everything = tool(srv, "list_tickets")["structuredContent"]
    assert [t["id"] for t in everything["tickets"]] == ["J-1", "Z-1"]
    only = tool(srv, "list_tickets", {"source": "zendesk"})["structuredContent"]
    assert [t["id"] for t in only["tickets"]] == ["Z-1"]
    solved = tool(srv, "list_tickets", {"status": "solved"})["structuredContent"]
    assert [t["id"] for t in solved["tickets"]] == ["J-1"]
    assert EMAIL not in json.dumps(everything)
    stored = ticket_db.get_ticket("Z-1")
    assert all(EMAIL not in c.body for c in stored.comments)
    bad = tool(srv, "list_tickets", {"limit": 1000, "source": "Robert'); DROP"})
    assert bad["isError"] and "DROP" not in bad["content"][0]["text"]


def test_suggest_reply_by_ticket_id(ticket_db) -> None:
    srv = server_for(ticket_db)
    result = tool(srv, "suggest_reply", {"ticket_id": "Z-1"})
    assert result["isError"] is False
    data = result["structuredContent"]
    assert data["found"] is True and "suggestion" in data
    assert "next_step" in data["suggestion"]
    assert tool(srv, "suggest_reply", {"ticket_id": "nope"})["structuredContent"] == {
        "found": False,
        "ticket_id": "nope",
    }
    no_copilot = tool(server_for(ticket_db, with_copilot=False), "suggest_reply", {"ticket_id": "Z-1"})
    assert no_copilot["isError"] and "ToolUnavailable" in no_copilot["content"][0]["text"]


def test_support_metrics(ticket_db) -> None:
    data = tool(server_for(ticket_db, with_copilot=False), "support_metrics", {"top": 3})["structuredContent"]
    assert data["tickets"] == 2
    assert "sentiment" in data and "ticket_sentiment" not in data
    assert len(data["risk"]) <= 3


def test_mcp_client_consumes_a_server_over_http(ticket_db) -> None:
    srv = server_for(ticket_db, with_copilot=False)
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        answer = asyncio.run(srv.handle_json(request.content.decode("utf-8")))
        return httpx.Response(200, content=answer or "")

    client = MCPToolClient(
        "https://mcp.example/mcp",
        "client-token-0123456789",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert [t["name"] for t in client.list_tools()][0] == "list_tickets"
    data = client.call_tool("list_tickets", {"source": "jira_sm"})
    assert [t["id"] for t in data["tickets"]] == ["J-1"]
    with pytest.raises(MCPError, match="suggest_reply failed"):
        client.call_tool("suggest_reply", {"ticket_id": "Z-1"})
    assert set(seen) == {"Bearer client-token-0123456789"}
    assert "client-token" not in repr(client)
    client.close()


def test_mcp_client_rules() -> None:
    with pytest.raises(MCPError):
        MCPToolClient("http://mcp.example/mcp", "t")
    with pytest.raises(MCPError):
        MCPToolClient("https://mcp.example/mcp", "")
    with pytest.raises(MCPError, match="NEXHUB_MCP_TOKEN"):
        MCPToolClient.from_env("NEXHUB_MCP_URL", "NEXHUB_MCP_TOKEN", environ={}, env_file=None)
    denied = MCPToolClient(
        "https://mcp.example/mcp",
        "client-token-0123456789",
        http=httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(401))),
    )
    with pytest.raises(MCPError, match="401") as info:
        denied.list_tools()
    assert "client-token" not in str(info.value)


def test_rest_app_serves_integration_tools(ticket_db) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from support_engine.copilot.api import create_app

    copilot = make_copilot()[0]
    token = "rest-token-0123456789abcdef"
    app = create_app(copilot, token=token, integrations=server_for(ticket_db, with_copilot=False))
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    with TestClient(app) as http:
        assert http.post("/mcp/integrations", json=body).status_code == 401
        answer = http.post("/mcp/integrations", json=body, headers={"Authorization": f"Bearer {token}"})
        assert answer.status_code == 200
        assert len(answer.json()["result"]["tools"]) == 3
    plain = create_app(copilot, token=token)
    with TestClient(plain) as http:
        assert http.post("/mcp/integrations", json=body, headers={"X-API-Key": token}).status_code == 404


# CLI ------------------------------------------------------------------------------------------


class RedirectingStdin:
    """Plays the user: reads the printed authorization URL and pastes back the redirect."""

    def __init__(self, stdout: io.StringIO) -> None:
        self.stdout = stdout

    def readline(self) -> str:
        state = re.search(r"[?&]state=([^&\s]+)", self.stdout.getvalue()).group(1)
        return f"http://127.0.0.1:8765/callback?code={CODE}&state={state}\n"


def test_cli_oauth_login_and_sync(tmp_path, monkeypatch, capsys) -> None:
    env_file = tmp_path / ".env"
    settings = env_for("zendesk", SUBDOMAIN="acme")
    env_file.write_text("\n".join(f"{k}={v}" for k, v in settings.items()) + "\n", encoding="utf-8")
    for key in settings:
        monkeypatch.delenv(key, raising=False)
    server = TokenServer()
    clock = Clock()
    real_flow, real_integration = providers_module.build_flow, providers_module.build_integration

    def fake_flow(name, **kwargs):
        return real_flow(name, token_http=server.client(), clock=clock, **kwargs)

    def api(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {ACCESS_1}"
        return httpx.Response(
            200,
            json={
                "tickets": [
                    {
                        "id": 77,
                        "subject": "Login loop",
                        "description": f"Customer {EMAIL} cannot log in.",
                        "status": "pending",
                        "created_at": "2026-02-01T10:00:00Z",
                    }
                ]
            },
        )

    def fake_integration(name, **kwargs):
        return real_integration(name, transport=httpx.MockTransport(api), clock=clock, **kwargs)

    import support_engine.integrations as integrations_pkg

    monkeypatch.setattr(integrations_pkg, "build_flow", fake_flow)
    monkeypatch.setattr(integrations_pkg, "build_integration", fake_integration)

    out = io.StringIO()
    args = cli.build_parser().parse_args(["oauth-login", "zendesk", "--env-file", str(env_file)])
    assert cli.cmd_oauth_login(args, stdin=RedirectingStdin(out), stdout=out) == 0
    printed = out.getvalue()
    assert "https://acme.zendesk.com/oauth/authorizations/new?" in printed
    assert "Saved the zendesk token to .env (scopes: tickets:read)" in printed
    for secret in SECRETS:
        assert secret not in printed
    assert f"SUPPORT_ENGINE_OAUTH_ZENDESK_ACCESS_TOKEN={ACCESS_1}" in env_file.read_text(encoding="utf-8")

    db = f"sqlite:///{tmp_path / 'cli.db'}"
    assert cli.main(["--db-url", db, "sync", "zendesk", "--env-file", str(env_file), "--max-pages", "1"]) == 0
    captured = capsys.readouterr()
    assert "Synced 1 ticket(s) from zendesk. PII masked: EMAIL=1." in captured.out
    for secret in SECRETS:
        assert secret not in captured.out + captured.err
    with TicketStore(target=db) as store:
        ticket = store.get_ticket("77")
    assert ticket.status == TicketStatus.PENDING and EMAIL not in ticket.description


def test_cli_errors_are_clean(tmp_path, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    assert cli.main(["oauth-login", "intercom", "--env-file", str(env_file)]) == 2
    assert "SUPPORT_ENGINE_OAUTH_INTERCOM_CLIENT_ID" in capsys.readouterr().err
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    assert cli.main(["--db-url", db, "sync", "zendesk", "--env-file", str(env_file)]) == 2
    assert "SUPPORT_ENGINE_OAUTH_ZENDESK_SUBDOMAIN" in capsys.readouterr().err


def test_cli_integrations_mcp_stdio(tmp_path, ticket_db) -> None:
    args = cli.build_parser().parse_args(["integrations-mcp"])
    stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
    stdout = io.StringIO()
    assert cli.cmd_integrations_mcp(args, ticket_db, stdin=stdin, stdout=stdout) == 0
    names = [t["name"] for t in json.loads(stdout.getvalue())["result"]["tools"]]
    assert names == ["list_tickets", "suggest_reply", "support_metrics"]
