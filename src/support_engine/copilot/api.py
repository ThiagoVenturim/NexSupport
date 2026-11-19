"""REST API of the copilot (FastAPI, optional extra ``api``).

Endpoints (all but ``/health`` need the access token)::

    GET  /health          liveness, no token
    POST /v1/suggest      suggested reply, summary, next step and tone for a ticket
    POST /v1/answer       ask the support agent one question
    POST /v1/summary      summary of a ticket
    POST /v1/feedback     the human agent's verdict on a suggestion (feeds curation)
    POST /mcp             MCP over HTTP (JSON-RPC 2.0), same tools as the stdio server
    POST /mcp/integrations  MCP tools over the synced help desk tickets (only when `integrations` is given)

The token comes from ``SUPPORT_ENGINE_API_TOKEN`` or the ``.env`` file (see ``auth``); the app
refuses to start without one. Validation errors list field locations and error types only, so
ticket text never comes back in an error. Bodies above ``max_body_bytes`` are refused.
"""

import json
import logging
from typing import Any

from support_engine.copilot.assist import Copilot
from support_engine.copilot.auth import TokenAuth, load_token
from support_engine.copilot.config import InterfaceConfig
from support_engine.copilot.mcp import CopilotMCPServer
from support_engine.copilot.service import (
    AnswerRequest,
    CopilotService,
    FeedbackRequest,
    SuggestRequest,
    SummaryRequest,
)

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({"/health"})


def create_app(
    copilot: Copilot,
    *,
    token: str | None = None,
    config: InterfaceConfig | None = None,
    integrations: Any = None,
) -> Any:
    """FastAPI app over ``copilot``. Without ``token`` it is read with ``load_token`` (env, then .env)."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise RuntimeError("The REST API needs FastAPI: pip install 'support-intelligence-engine[api]'") from exc

    cfg = config or InterfaceConfig()
    secret = token or load_token(cfg.token_env, cfg.env_file, min_chars=cfg.min_token_chars)
    auth = TokenAuth(secret, min_chars=cfg.min_token_chars)
    service = CopilotService(copilot)
    mcp = CopilotMCPServer(service)

    app = FastAPI(
        title="Support Intelligence Engine copilot",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.service = service
    app.state.mcp = mcp

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path not in PUBLIC_PATHS:
            if not auth.check(request.headers):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "unauthorized"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if request.method in ("POST", "PUT", "PATCH"):
                length = request.headers.get("content-length")
                if length is None:
                    return JSONResponse(status_code=411, content={"detail": "content_length_required"})
                if not length.isdigit() or int(length) > cfg.max_body_bytes:
                    return JSONResponse(status_code=413, content={"detail": "body_too_large"})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid(_request: Request, exc: RequestValidationError):
        errors = [{"loc": [str(p) for p in e.get("loc", ())], "type": e.get("type", "invalid")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def failed(_request: Request, exc: Exception):
        logger.warning("copilot api failed: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal_error"})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/suggest")
    async def suggest(body: SuggestRequest) -> dict[str, Any]:
        return await service.suggest(body)

    @app.post("/v1/answer")
    async def answer(body: AnswerRequest) -> dict[str, Any]:
        return await service.answer(body)

    @app.post("/v1/summary")
    async def summary(body: SummaryRequest) -> dict[str, Any]:
        return await service.summarize(body)

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackRequest) -> dict[str, Any]:
        return await service.feedback(body)

    @app.post("/mcp")
    async def mcp_http(request: Request):
        raw = await request.body()
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(status_code=400, content=CopilotMCPServer._error(None, -32700, "Parse error."))
        answer = await mcp.handle_payload(payload)
        if answer is None:
            return JSONResponse(status_code=202, content=None)
        return JSONResponse(content=answer)

    if integrations is not None:

        @app.post("/mcp/integrations")
        async def mcp_integrations(request: Request):
            raw = await request.body()
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return JSONResponse(status_code=400, content=CopilotMCPServer._error(None, -32700, "Parse error."))
            answer = await integrations.handle_payload(payload)
            if answer is None:
                return JSONResponse(status_code=202, content=None)
            return JSONResponse(content=answer)

    return app
