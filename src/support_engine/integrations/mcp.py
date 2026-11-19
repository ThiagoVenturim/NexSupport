"""MCP tools over the synced help desk tickets, for other assistants and for NexHub.

Tools:

- ``list_tickets``: tickets synced from Zendesk, Freshdesk, Intercom or Jira Service Management,
  in the common model, PII already masked.
- ``suggest_reply``: suggested reply for a stored ticket, through the copilot.
- ``support_metrics``: sentiment, trend alerts, risk and root causes over the stored tickets.

Same JSON-RPC handling as the copilot server (stdio, or ``POST`` behind the API token); tool
errors report a type only, never arguments or secrets. ``MCPToolClient`` consumes any MCP
server over HTTP with a bearer token read from the environment.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from support_engine.analytics import AnalyticsConfig, AnalyticsPipeline
from support_engine.copilot.mcp import CopilotMCPServer, Tool
from support_engine.copilot.service import NAMESPACE_PATTERN, CopilotService, SuggestRequest
from support_engine.integrations.tokens import merged_env
from support_engine.models import Ticket, TicketStatus
from support_engine.storage import TicketStore

ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,128}$"
SOURCE_PATTERN = r"^[a-z_]{1,32}$"


class ListTicketsRequest(BaseModel):
    status: TicketStatus | None = None
    source: str | None = Field(default=None, pattern=SOURCE_PATTERN, description="zendesk, freshdesk, ...")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=100_000)


class SuggestByIdRequest(BaseModel):
    ticket_id: str = Field(pattern=ID_PATTERN)
    namespace: str = Field(default="default", pattern=NAMESPACE_PATTERN)


class MetricsRequest(BaseModel):
    max_tickets: int = Field(default=1000, ge=1, le=10_000)
    top: int = Field(default=10, ge=1, le=50)


class ToolUnavailable(RuntimeError):
    """The tool needs a component that is not configured."""


def ticket_row(ticket: Ticket) -> dict[str, Any]:
    return {
        "id": ticket.id,
        "source": ticket.source,
        "subject": ticket.subject,
        "status": ticket.status.value,
        "priority": ticket.priority,
        "product": ticket.product,
        "tags": list(ticket.tags),
        "created_at": ticket.created_at.isoformat(),
        "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
        "comments": len(ticket.comments),
    }


class IntegrationsService:
    """Operations behind the integration MCP tools."""

    def __init__(self, store: TicketStore, copilot: CopilotService | None = None) -> None:
        self.store = store
        self.copilot = copilot

    async def list_tickets(self, request: ListTicketsRequest) -> dict[str, Any]:
        if request.source is None:
            tickets = self.store.list_tickets(status=request.status, limit=request.limit, offset=request.offset)
        else:
            matching = [
                t for t in self.store.list_tickets(status=request.status, limit=100_000) if t.source == request.source
            ]
            tickets = matching[request.offset : request.offset + request.limit]
        return {"tickets": [ticket_row(t) for t in tickets], "count": len(tickets)}

    async def suggest_reply(self, request: SuggestByIdRequest) -> dict[str, Any]:
        if self.copilot is None:
            raise ToolUnavailable("copilot not configured")
        ticket = self.store.get_ticket(request.ticket_id)
        if ticket is None:
            return {"found": False, "ticket_id": request.ticket_id}
        result = await self.copilot.suggest(SuggestRequest(ticket=ticket, namespace=request.namespace))
        return {"found": True, "ticket_id": request.ticket_id, "suggestion": result}

    async def support_metrics(self, request: MetricsRequest) -> dict[str, Any]:
        tickets = self.store.list_tickets(limit=request.max_tickets)
        report = AnalyticsPipeline(AnalyticsConfig()).run(tickets, None)
        data = report.model_dump(mode="json", exclude={"ticket_sentiment"})
        for key in ("trends", "alerts", "risk", "root_causes"):
            if isinstance(data.get(key), list):
                data[key] = data[key][: request.top]
        return data


class IntegrationsMCPServer(CopilotMCPServer):
    """MCP server with ``list_tickets``, ``suggest_reply`` and ``support_metrics``."""

    def __init__(
        self, service: IntegrationsService, *, name: str = "support-engine-integrations", version: str = "0.1.0"
    ):
        self.service = service  # type: ignore[assignment]
        self.name = name
        self.version = version
        self.tools = {
            t.name: t
            for t in (
                Tool(
                    "list_tickets",
                    "List help desk tickets synced from Zendesk, Freshdesk, Intercom or Jira Service Management, "
                    "newest first, in the common ticket model with PII masked. Filters: status, source.",
                    ListTicketsRequest,
                    service.list_tickets,
                ),
                Tool(
                    "suggest_reply",
                    "Suggested reply, summary, next step and tone for a stored ticket, grounded in curated "
                    "past answers.",
                    SuggestByIdRequest,
                    service.suggest_reply,
                ),
                Tool(
                    "support_metrics",
                    "Support metrics over the stored tickets: sentiment, trend alerts, risk radar, service "
                    "quality and root causes.",
                    MetricsRequest,
                    service.support_metrics,
                ),
            )
        }

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        result = super()._initialize(params)
        result["instructions"] = "Help desk tickets synced over OAuth. Ticket text is masked before storage."
        return result


class MCPError(RuntimeError):
    """A remote MCP call failed. The message never contains the token."""


class MCPToolClient:
    """Minimal MCP client over HTTP (``tools/list`` and ``tools/call``) with a bearer token."""

    def __init__(self, url: str, token: str, *, http: httpx.Client | None = None, timeout: float = 30.0) -> None:
        if not url.startswith("https://") and not url.startswith(("http://127.0.0.1", "http://localhost")):
            raise MCPError("The MCP URL must be https, or http on localhost.")
        if not token:
            raise MCPError("An access token is required for the MCP server.")
        self.url = url
        self._token = token
        self._http = http or httpx.Client(timeout=timeout)
        self._ids = itertools.count(1)
        self._initialized = False

    def __repr__(self) -> str:
        return f"MCPToolClient(url={self.url!r}, token=***)"

    @classmethod
    def from_env(
        cls,
        url_var: str,
        token_var: str,
        *,
        environ: Mapping[str, str] | None = None,
        env_file: str | Path | None = ".env",
        **kwargs: Any,
    ) -> MCPToolClient:
        values = merged_env(environ, env_file)
        url, token = values.get(url_var, "").strip(), values.get(token_var, "").strip()
        if not url or not token:
            raise MCPError(f"Set {url_var} and {token_var} in the environment or in the .env file.")
        return cls(url, token, **kwargs)

    def close(self) -> None:
        self._http.close()

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        message = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params or {}}
        try:
            response = self._http.post(self.url, json=message, headers={"Authorization": f"Bearer {self._token}"})
        except httpx.HTTPError as exc:
            raise MCPError(f"MCP request failed ({type(exc).__name__}).") from None
        if response.status_code >= 400:
            raise MCPError(f"MCP server answered {response.status_code}.")
        try:
            payload = response.json()
        except ValueError:
            raise MCPError("MCP server sent invalid JSON.") from None
        if not isinstance(payload, dict):
            raise MCPError("MCP server sent an unexpected answer.")
        if "error" in payload:
            error = payload["error"] if isinstance(payload["error"], dict) else {}
            raise MCPError(f"MCP error {error.get('code')}: {error.get('message')}")
        result = payload.get("result")
        return result if isinstance(result, dict) else {}

    def initialize(self) -> dict[str, Any]:
        result = self._rpc(
            "initialize",
            {"protocolVersion": "2025-06-18", "clientInfo": {"name": "support-engine", "version": "0.1.0"}},
        )
        self._initialized = True
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        if not self._initialized:
            self.initialize()
        tools = self._rpc("tools/list").get("tools", [])
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Structured result of the tool; raises ``MCPError`` when the tool reports an error."""
        if not self._initialized:
            self.initialize()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            raise MCPError(f"MCP tool {name} failed.")
        if "structuredContent" in result:
            return result["structuredContent"]
        content = result.get("content") or []
        if content and isinstance(content[0], dict) and content[0].get("type") == "text":
            try:
                return json.loads(content[0].get("text", ""))
            except ValueError:
                return content[0].get("text")
        return result


__all__ = [
    "IntegrationsMCPServer",
    "IntegrationsService",
    "ListTicketsRequest",
    "MCPError",
    "MCPToolClient",
    "MetricsRequest",
    "SuggestByIdRequest",
    "ticket_row",
]
