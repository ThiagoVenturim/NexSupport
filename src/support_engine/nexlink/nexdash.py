"""Client of the nexdash (askdash) report engine: the Support Engine asks for dashboards.

nexdash serves report models over REST and MCP (see its ``docs/REPORT_ENGINE.md``). A request
names a model and passes a few validated parameters; it never carries SQL. This client builds
that request, sends it over a transport and returns the answer:

- ``RestReportTransport``: ``POST /api/reports/run`` with ``Authorization: Bearer <token>``.
  The token comes from ``NEXDASH_ENGINE_TOKEN`` (environment or ``.env``), never from code.
- ``EngineReportTransport``: an in-process ``nexdash.engine.ReportEngine`` (no network), used by
  the offline examples and tests when nexdash is installed.

The token is never logged. Error messages carry the nexdash status and its client-safe message,
never the request.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from support_engine.nexlink.security import MIN_TOKEN_CHARS, outgoing_token

logger = logging.getLogger(__name__)

TOKEN_ENV = "NEXDASH_ENGINE_TOKEN"
SUPPORT_MODEL = "support_metrics"


class ReportError(RuntimeError):
    """nexdash refused or failed a request. ``status`` follows the REST codes of the engine."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"nexdash answered {status}: {message}")
        self.status = status
        self.message = message


class ReportParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: dt.date | None = None
    end: dt.date | None = None
    team: str | None = Field(default=None, max_length=200)
    project: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _period(self) -> ReportParameters:
        if self.start and self.end and self.end < self.start:
            raise ValueError("'end' must not be earlier than 'start'")
        return self


class ReportRequest(BaseModel):
    """Same object as the nexdash ``RunRequest``; unknown fields are refused here too."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default=SUPPORT_MODEL, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    parameters: ReportParameters = Field(default_factory=ReportParameters)
    output: Literal["data", "dashboard", "both"] = "both"
    locale: Literal["en", "pt-BR"] = "en"
    compare: bool = True


class ReportTransport(Protocol):
    def run(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def list_models(self) -> list[dict[str, Any]]: ...


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return "no details"
    message = body.get("error") if isinstance(body, dict) else None
    return str(message)[:300] if message else "no details"


class RestReportTransport:
    """nexdash over REST. Pass ``client`` to reuse a pool or, in tests, an ``httpx.MockTransport``."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        env_file: str | Path | None = ".env",
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("The nexdash URL must start with http:// or https://")
        secret = token or outgoing_token("nexdash", env_var=TOKEN_ENV, environ=environ, env_file=env_file)
        if len(secret) < MIN_TOKEN_CHARS:
            raise ValueError(f"The nexdash token must have at least {MIN_TOKEN_CHARS} characters.")
        self._headers = {"Authorization": f"Bearer {secret}"}
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout, trust_env=False)

    def __repr__(self) -> str:
        return f"RestReportTransport(base_url={self._base_url!r}, token=***)"

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.request(method, f"{self._base_url}{path}", json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            logger.warning("nexdash request failed: %s", type(exc).__name__)
            raise ReportError(503, "nexdash is not reachable") from exc
        if response.status_code >= 400:
            raise ReportError(response.status_code, _error_message(response))
        return response.json()

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/api/reports/run", request)

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/api/reports/models").get("models", []))


class EngineReportTransport:
    """An in-process nexdash ``ReportEngine`` (or any object with ``run`` and ``list_models``)."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    @classmethod
    def demo(cls) -> EngineReportTransport:
        """nexdash engine over in-memory SQLite with its synthetic report data. Needs nexdash installed."""
        try:
            from nexdash.engine import ReportEngine
            from nexdash.executor import get_executor
            from nexdash.report_models.synthetic import seed_model_data
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("The in-process engine needs nexdash: pip install nexdash") from exc
        executor = get_executor("sqlite:///:memory:")
        seed_model_data(executor)
        return cls(ReportEngine(executor))

    def _call(self, fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as exc:
            status = getattr(exc, "status", None)
            if isinstance(status, int):
                raise ReportError(status, str(exc)[:300]) from exc
            logger.warning("nexdash engine failed: %s", type(exc).__name__)
            raise ReportError(500, "report engine failure") from exc

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call(self.engine.run, request)

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._call(self.engine.list_models))


class NexDashClient:
    """Validates report requests locally, then sends them to nexdash."""

    def __init__(self, transport: ReportTransport) -> None:
        self.transport = transport

    @classmethod
    def from_env(cls, base_url: str, **kwargs: Any) -> NexDashClient:
        return cls(RestReportTransport(base_url, **kwargs))

    def list_models(self) -> list[str]:
        return [m["name"] for m in self.transport.list_models() if isinstance(m, dict) and "name" in m]

    def run_report(self, request: ReportRequest | Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
        """Run a report model; ``fields`` override the request (``model``, ``parameters``, ``output``...)."""
        if isinstance(request, ReportRequest):
            base: dict[str, Any] = request.model_dump()
        else:
            base = dict(request or {})
        req = ReportRequest.model_validate({**base, **fields})
        payload = req.model_dump(mode="json")
        answer = self.transport.run(payload)
        if not isinstance(answer, dict) or answer.get("model") != req.model:
            raise ReportError(502, "unexpected answer from nexdash")
        return answer

    def support_dashboard(
        self,
        *,
        start: dt.date | str | None = None,
        end: dt.date | str | None = None,
        team: str | None = None,
        locale: Literal["en", "pt-BR"] = "en",
        output: Literal["data", "dashboard", "both"] = "both",
    ) -> dict[str, Any]:
        """The ``support_metrics`` report: volume, first response, resolution, deflection, CSAT, sentiment."""
        return self.run_report(
            ReportRequest(
                model=SUPPORT_MODEL,
                parameters=ReportParameters(start=start, end=end, team=team),
                output=output,
                locale=locale,
            )
        )
