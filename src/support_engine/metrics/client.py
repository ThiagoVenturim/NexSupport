"""Client for the askdash (NexDash) report engine over REST or MCP (JSON-RPC over HTTP).

Needs only ``httpx``. A request names a report model and passes period, team and
project; it never carries SQL. The token is read from the environment
(``NEXDASH_ENGINE_TOKEN``) or passed in, travels as ``Authorization: Bearer``, and is
never logged or shown in ``repr``. Error messages carry the HTTP status and the
server's client-safe message only.

    client = NexDashClient.from_env()          # NEXDASH_URL, NEXDASH_ENGINE_TOKEN
    report = client.run_report("support_insights", start="2026-04-01", end="2026-06-30", team="tier1")
    html = report["dashboard"]["html"]
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx

logger = logging.getLogger(__name__)

URL_ENV = "NEXDASH_URL"
TOKEN_ENV = "NEXDASH_ENGINE_TOKEN"
DEFAULT_URL = "http://127.0.0.1:8000"
MCP_PROTOCOL_VERSION = "2025-06-18"
MAX_TEXT_PARAM = 200
_MODEL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

Transport = Literal["rest", "mcp"]
Output = Literal["data", "dashboard", "both"]
Locale = Literal["en", "pt-BR"]


class NexDashError(RuntimeError):
    """A failed call to NexDash. ``status`` is the HTTP status (0 for transport or MCP tool errors)."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def _date(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    try:
        return dt.date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise ValueError(f"Parameter '{name}' must be a date in YYYY-MM-DD form") from None


def _text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > MAX_TEXT_PARAM:
        raise ValueError(f"Parameter '{name}' is longer than {MAX_TEXT_PARAM} characters")
    return text


def report_request(
    model: str,
    *,
    start: Any = None,
    end: Any = None,
    team: str | None = None,
    project: str | None = None,
    output: Output = "both",
    locale: Locale = "en",
    compare: bool = True,
) -> dict[str, Any]:
    """Build and check a report request before it leaves the process (NexDash checks it again)."""
    if not isinstance(model, str) or not _MODEL.match(model):
        raise ValueError("Report model must be a lowercase identifier")
    if output not in ("data", "dashboard", "both"):
        raise ValueError("Output must be data, dashboard or both")
    if locale not in ("en", "pt-BR"):
        raise ValueError("Locale must be en or pt-BR")
    params = {
        "start": _date(start, "start"),
        "end": _date(end, "end"),
        "team": _text(team, "team"),
        "project": _text(project, "project"),
    }
    if params["start"] and params["end"] and params["end"] < params["start"]:
        raise ValueError("Parameter 'end' must not be earlier than 'start'")
    return {"model": model, "parameters": params, "output": output, "locale": locale, "compare": bool(compare)}


def _check_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("NexDash URL must be an http or https URL with a host")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("NexDash URL must not carry credentials, a query or a fragment")
    return url.rstrip("/")


class NexDashClient:
    """Talks to a NexDash server started with ``--report-models`` (or ``support-engine metrics-server``)."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        token: str | None = None,
        *,
        transport: Transport = "rest",
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ) -> None:
        if transport not in ("rest", "mcp"):
            raise ValueError("transport must be rest or mcp")
        self.url = _check_url(url)
        self.transport = transport
        self._token = token or None
        self._http = http or httpx.Client(timeout=timeout)
        self._owns_http = http is None
        self._mcp_ready = False
        self._next_id = 0

    @classmethod
    def from_env(cls, *, transport: Transport = "rest", **kwargs: Any) -> NexDashClient:
        """URL from ``NEXDASH_URL`` (default local), token from ``NEXDASH_ENGINE_TOKEN``."""
        return cls(os.environ.get(URL_ENV) or DEFAULT_URL, os.environ.get(TOKEN_ENV), transport=transport, **kwargs)

    def __repr__(self) -> str:
        return (
            f"NexDashClient(url={self.url!r}, transport={self.transport!r}, token={'set' if self._token else 'unset'})"
        )

    def __enter__(self) -> NexDashClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    # -- operations ----------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        if self.transport == "mcp":
            return self._tool("nexdash_list_report_models", {})["models"]
        return self._rest("GET", "/api/reports/models")["models"]

    def describe_model(self, model: str) -> dict[str, Any]:
        if not isinstance(model, str) or not _MODEL.match(model):
            raise ValueError("Report model must be a lowercase identifier")
        if self.transport == "mcp":
            return self._tool("nexdash_describe_report_model", {"model": model})
        return self._rest("GET", f"/api/reports/models/{quote(model)}")

    def run_report(self, model: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a report: ``{"data": {"panels": ...}, "dashboard": {"html": ...}, ...}``."""
        request = report_request(model, **kwargs)
        if self.transport == "mcp":
            return self._tool("nexdash_run_report", request)
        return self._rest("POST", "/api/reports/run", request)

    def dashboard(self, model: str, **kwargs: Any) -> str:
        """Only the HTML dashboard of a report."""
        kwargs["output"] = "dashboard"
        if self.transport == "mcp":
            return self.run_report(model, **kwargs)["dashboard"]["html"]
        response = self._send("POST", "/api/reports/dashboard", report_request(model, **kwargs))
        return response.text

    # -- transports ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, text/html"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _send(self, method: str, path: str, body: Any = None) -> httpx.Response:
        try:
            response = self._http.request(method, self.url + path, json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            logger.warning("NexDash call failed: %s", type(exc).__name__)
            raise NexDashError(f"Could not reach NexDash ({type(exc).__name__})") from None
        if response.status_code >= 400:
            message = ""
            try:
                message = str(response.json().get("error") or "")
            except (ValueError, AttributeError):
                message = ""
            logger.warning("NexDash answered %d on %s", response.status_code, path)
            raise NexDashError(
                f"NexDash answered {response.status_code}: {message or 'no detail'}", response.status_code
            )
        return response

    def _rest(self, method: str, path: str, body: Any = None) -> Any:
        response = self._send(method, path, body)
        try:
            return response.json()
        except ValueError:
            raise NexDashError("NexDash answered with a body that is not JSON") from None

    def _rpc(self, method: str, params: Mapping[str, Any] | None = None, *, notify: bool = False) -> Any:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        if notify:
            self._send("POST", "/mcp", message)
            return None
        self._next_id += 1
        message["id"] = self._next_id
        reply = self._rest("POST", "/mcp", message)
        if not isinstance(reply, dict):
            raise NexDashError("NexDash MCP answered with an unexpected message")
        if "error" in reply:
            error = reply["error"] if isinstance(reply["error"], dict) else {}
            raise NexDashError(f"NexDash MCP error: {error.get('message', 'no detail')}")
        return reply.get("result")

    def _initialize(self) -> None:
        if self._mcp_ready:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "support-engine", "version": "1"},
            },
        )
        self._rpc("notifications/initialized", notify=True)
        self._mcp_ready = True

    def _tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        self._initialize()
        result = self._rpc("tools/call", {"name": name, "arguments": dict(arguments)})
        if not isinstance(result, dict):
            raise NexDashError("NexDash MCP answered with an unexpected result")
        payload = result.get("structuredContent")
        if payload is None:
            texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
            try:
                payload = json.loads(texts[0]) if texts else None
            except ValueError:
                payload = None
        if result.get("isError"):
            message = payload.get("error") if isinstance(payload, dict) else None
            raise NexDashError(f"NexDash tool '{name}' failed: {message or 'no detail'}")
        if payload is None:
            raise NexDashError(f"NexDash tool '{name}' returned no data")
        return payload


__all__ = ["DEFAULT_URL", "TOKEN_ENV", "URL_ENV", "NexDashClient", "NexDashError", "report_request"]
