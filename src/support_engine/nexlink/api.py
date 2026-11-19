"""NexLink over HTTP (FastAPI, optional extra ``api``).

``POST /nexlink`` takes one envelope and answers one reply. The token goes in
``Authorization: Bearer <token>`` (or ``X-API-Key``). The HTTP status follows the reply::

    ok 200, invalid_envelope / invalid_params / wrong_target 400, unauthorized 401,
    forbidden 403, unknown_action 404, failed 500, upstream_error 502, unavailable 503

Bodies above ``max_body_bytes`` are refused before they are parsed.
"""

# No ``from __future__ import annotations``: FastAPI must see the real ``Request`` type of the
# endpoint, which is imported inside ``create_nexlink_app``.
import json
from typing import Any

from support_engine.copilot.auth import TokenAuth
from support_engine.nexlink.protocol import ErrorCode, Reply
from support_engine.nexlink.server import NexLinkServer

STATUS = {
    ErrorCode.INVALID_ENVELOPE: 400,
    ErrorCode.INVALID_PARAMS: 400,
    ErrorCode.WRONG_TARGET: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.UNKNOWN_ACTION: 404,
    ErrorCode.FAILED: 500,
    ErrorCode.UPSTREAM_ERROR: 502,
    ErrorCode.UNAVAILABLE: 503,
}


def http_status(reply: Reply) -> int:
    return 200 if reply.ok or reply.error is None else STATUS.get(reply.error.code, 500)


def create_nexlink_app(server: NexLinkServer, *, max_body_bytes: int = 5_000_000) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise RuntimeError("NexLink over HTTP needs FastAPI: pip install 'support-intelligence-engine[api]'") from exc

    app = FastAPI(title="Support Engine NexLink", version=server.version, docs_url=None, redoc_url=None)
    app.state.nexlink = server

    def answer(reply: Reply) -> Any:
        return JSONResponse(status_code=http_status(reply), content=reply.model_dump(mode="json"))

    def refuse(code: ErrorCode, message: str, status: int) -> Any:
        reply = Reply(sender=server.project, status="error", error={"code": code, "message": message})
        return JSONResponse(status_code=status, content=reply.model_dump(mode="json"))

    @app.post("/nexlink")
    async def nexlink(request: Request):
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            return refuse(ErrorCode.INVALID_ENVELOPE, "Content-Length is required.", 411)
        if int(length) > max_body_bytes:
            return refuse(ErrorCode.INVALID_ENVELOPE, "The message is too large.", 413)
        raw = await request.body()
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return refuse(ErrorCode.INVALID_ENVELOPE, "The body is not JSON.", 400)
        token = TokenAuth.credential(request.headers)
        return answer(await server.handle(payload, token=token))

    return app
