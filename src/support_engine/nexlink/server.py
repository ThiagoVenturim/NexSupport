"""NexLink server: checks an envelope, its token and the action scope, then runs the action.

The server knows nothing about support tickets. ``support_engine.nexlink.support`` registers
the Support Engine actions on it; any other project could reuse it with its own catalog.

Logs carry the sender, the action, the outcome and the correlation id. They never carry
parameters, results or tokens.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from support_engine.copilot.service import validation_summary
from support_engine.nexlink.protocol import PROTOCOL, Envelope, ErrorCode, Reply
from support_engine.nexlink.security import ClientRegistry

logger = logging.getLogger(__name__)

MANIFEST_ACTION = "nexlink.manifest"

Handler = Callable[[Any, Envelope], Awaitable[dict[str, Any] | None] | dict[str, Any] | None]


class ActionError(Exception):
    """Raised by a handler to answer with a specific code. The message must be safe to show."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ActionSpec(BaseModel):
    """One entry of a manifest."""

    name: str
    description: str
    scope: str | None = Field(description="scope the caller token needs; None for any authenticated caller")
    params_schema: dict[str, Any]


class Manifest(BaseModel):
    """What a project exposes over NexLink. NexHub collects the manifests of every project."""

    protocol: str = PROTOCOL
    project: str
    version: str
    actions: list[ActionSpec]

    def action(self, name: str) -> ActionSpec | None:
        return next((a for a in self.actions if a.name == name), None)


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    scope: str | None
    params: type[BaseModel]
    handler: Handler

    def spec(self) -> ActionSpec:
        return ActionSpec(
            name=self.name,
            description=self.description,
            scope=self.scope,
            params_schema=self.params.model_json_schema(),
        )


class NexLinkServer:
    """Dispatches NexLink envelopes addressed to ``project``."""

    def __init__(self, project: str, registry: ClientRegistry, *, version: str = "0.1.0") -> None:
        self.project = project
        self.registry = registry
        self.version = version
        self.actions: dict[str, Action] = {}
        self.register(
            MANIFEST_ACTION,
            "Actions this project exposes and the scope each one needs.",
            None,
            _NoParams,
            lambda _params, _envelope: self.manifest().model_dump(mode="json"),
        )

    def register(
        self,
        name: str,
        description: str,
        scope: str | None,
        params: type[BaseModel],
        handler: Handler,
    ) -> None:
        if name in self.actions:
            raise ValueError(f"Action '{name}' is registered twice")
        if scope is not None and scope not in self.registry.allowed_scopes:
            raise ValueError(f"Action '{name}' uses an unknown scope")
        self.actions[name] = Action(name, description, scope, params, handler)

    def manifest(self) -> Manifest:
        return Manifest(
            project=self.project,
            version=self.version,
            actions=[self.actions[name].spec() for name in sorted(self.actions)],
        )

    def _log(self, envelope: Envelope, outcome: str) -> None:
        logger.info(
            "nexlink %s sender=%s action=%s outcome=%s correlation=%s",
            envelope.kind,
            envelope.sender,
            envelope.action,
            outcome,
            envelope.conversation,
        )

    async def handle(self, message: Envelope | Mapping[str, Any] | Any, token: str | None = None) -> Reply:
        """Answer one message. ``token`` (from the transport) wins over the token in the envelope."""
        if isinstance(message, Envelope):
            envelope = message
        else:
            try:
                envelope = Envelope.model_validate(message)
            except ValidationError as exc:
                logger.info("nexlink invalid envelope")
                return Reply(
                    sender=self.project,
                    status="error",
                    error={
                        "code": ErrorCode.INVALID_ENVELOPE,
                        "message": "The message is not a valid NexLink envelope.",
                        "details": validation_summary(exc),
                    },
                )

        if envelope.target != self.project:
            self._log(envelope, "wrong_target")
            return envelope.reply_error(self.project, ErrorCode.WRONG_TARGET, "This project is not the target.")

        candidate = token if token is not None else (envelope.token.get_secret_value() if envelope.token else None)
        client = self.registry.authenticate(envelope.sender, candidate)
        if client is None:
            self._log(envelope, "unauthorized")
            return envelope.reply_error(self.project, ErrorCode.UNAUTHORIZED, "Missing or invalid token.")

        action = self.actions.get(envelope.action)
        if action is None:
            self._log(envelope, "unknown_action")
            return envelope.reply_error(self.project, ErrorCode.UNKNOWN_ACTION, "Unknown action.")
        if not client.allows(action.scope):
            self._log(envelope, "forbidden")
            return envelope.reply_error(self.project, ErrorCode.FORBIDDEN, "The token lacks the scope of this action.")

        try:
            params = action.params.model_validate(envelope.params)
        except ValidationError as exc:
            self._log(envelope, "invalid_params")
            return envelope.reply_error(
                self.project, ErrorCode.INVALID_PARAMS, "Invalid parameters.", validation_summary(exc)
            )

        try:
            result = action.handler(params, envelope)
            if inspect.isawaitable(result):
                result = await result
        except ActionError as exc:
            self._log(envelope, exc.code.value)
            return envelope.reply_error(self.project, exc.code, exc.message)
        except Exception as exc:
            logger.warning("nexlink action %s failed: %s", envelope.action, type(exc).__name__)
            return envelope.reply_error(self.project, ErrorCode.FAILED, "The action failed.")

        self._log(envelope, "ok")
        return envelope.reply_ok(self.project, None if envelope.kind == "notice" else result)

    def handle_sync(self, message: Envelope | Mapping[str, Any] | Any, token: str | None = None) -> Reply:
        """``handle`` for callers without an event loop."""
        return asyncio.run(self.handle(message, token))
