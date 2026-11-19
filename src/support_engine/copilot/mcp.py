"""Model Context Protocol server for the copilot (JSON-RPC 2.0, no extra dependency).

Tools: ``suggest_reply``, ``answer_question``, ``summarize_ticket`` and ``record_feedback``.

Transports:

- stdio (``support-engine mcp``): one JSON message per line on stdin and stdout. The MCP client
  starts the server as a child process under its own user, the usual trust model for stdio.
  Logs go to stderr so stdout carries protocol messages only.
- HTTP: ``POST /mcp`` on the REST app, behind the same access token as the rest of the API.

Errors never echo arguments: a validation error reports field locations and error types only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TextIO

from pydantic import BaseModel, ValidationError

from support_engine.copilot.service import (
    AnswerRequest,
    CopilotService,
    FeedbackRequest,
    SuggestRequest,
    SummaryRequest,
    validation_summary,
)

logger = logging.getLogger(__name__)

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    model: type[BaseModel]
    handler: Callable[[Any], Awaitable[dict[str, Any]]]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.model.model_json_schema()}


class CopilotMCPServer:
    """Dispatches MCP requests to the copilot service."""

    def __init__(self, service: CopilotService, *, name: str = "support-engine", version: str = "0.1.0") -> None:
        self.service = service
        self.name = name
        self.version = version
        self.tools: dict[str, Tool] = {
            t.name: t
            for t in (
                Tool(
                    "suggest_reply",
                    "Suggested reply, summary, next step and tone for an open support ticket. The reply is grounded "
                    "in curated past answers and lists the tickets it cites.",
                    SuggestRequest,
                    service.suggest,
                ),
                Tool(
                    "answer_question",
                    "Ask the support agent one question; returns the cited answer, the level (1 answer, 2 review, "
                    "3 escalate) and the reasons.",
                    AnswerRequest,
                    service.answer,
                ),
                Tool(
                    "summarize_ticket",
                    "Short summary of a support ticket: problem, latest messages and status.",
                    SummaryRequest,
                    service.summarize,
                ),
                Tool(
                    "record_feedback",
                    "Record whether the human agent accepted, edited or rejected a suggested reply; feeds curation.",
                    FeedbackRequest,
                    service.feedback,
                ),
            )
        }

    # JSON-RPC ---------------------------------------------------------------------------------

    @staticmethod
    def _error(msg_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": error}

    @staticmethod
    def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        return {
            "protocolVersion": version,
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {"tools": {"listChanged": False}},
            "instructions": "Support copilot over curated tickets. Ticket text is masked before use.",
        }

    async def _call_tool(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return self._error(msg_id, INVALID_PARAMS, "Unknown tool.")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return self._error(msg_id, INVALID_PARAMS, "Tool arguments must be an object.")
        try:
            request = tool.model.model_validate(arguments)
        except ValidationError as exc:
            errors = validation_summary(exc)
            text = json.dumps({"error": "invalid_arguments", "details": errors})
            return self._result(msg_id, {"content": [{"type": "text", "text": text}], "isError": True})
        try:
            data = await tool.handler(request)
        except Exception as exc:
            logger.warning("mcp tool %s failed: %s", tool.name, type(exc).__name__)
            text = json.dumps({"error": "tool_failed", "type": type(exc).__name__})
            return self._result(msg_id, {"content": [{"type": "text", "text": text}], "isError": True})
        text = json.dumps(data, ensure_ascii=False)
        return self._result(
            msg_id, {"content": [{"type": "text", "text": text}], "structuredContent": data, "isError": False}
        )

    async def handle(self, message: Any) -> dict[str, Any] | None:
        """Answer one JSON-RPC message; notifications (no id) get no answer."""
        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            return self._error(
                message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "Invalid request."
            )
        method = message["method"]
        is_notification = "id" not in message
        msg_id = message.get("id")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return None if is_notification else self._error(msg_id, INVALID_PARAMS, "Params must be an object.")
        if is_notification:
            return None

        if method == "initialize":
            return self._result(msg_id, self._initialize(params))
        if method == "ping":
            return self._result(msg_id, {})
        if method == "tools/list":
            return self._result(msg_id, {"tools": [t.schema() for t in self.tools.values()]})
        if method == "tools/call":
            return await self._call_tool(msg_id, params)
        return self._error(msg_id, METHOD_NOT_FOUND, "Method not found.")

    async def handle_payload(self, payload: Any) -> Any:
        """A single message or a batch (list); returns None when nothing needs an answer."""
        if isinstance(payload, list):
            if not payload:
                return self._error(None, INVALID_REQUEST, "Empty batch.")
            answers = [a for a in [await self.handle(m) for m in payload] if a is not None]
            return answers or None
        return await self.handle(payload)

    async def handle_json(self, raw: str) -> str | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps(self._error(None, PARSE_ERROR, "Parse error."))
        answer = await self.handle_payload(payload)
        return None if answer is None else json.dumps(answer, ensure_ascii=False)

    # stdio ------------------------------------------------------------------------------------

    async def serve_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Read one message per line until end of input."""
        reader = stdin or sys.stdin
        writer = stdout or sys.stdout
        while True:
            line = await asyncio.to_thread(reader.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            answer = await self.handle_json(line)
            if answer is not None:
                writer.write(answer + "\n")
                writer.flush()

    def run_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        asyncio.run(self.serve_stdio(stdin, stdout))
