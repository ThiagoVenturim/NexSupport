"""NexLink message envelope: one contract for a project to ask another one for something.

A request names the sender, the target project, the action and its parameters, and carries a
correlation id so a chain of calls (NexHub -> Support Engine -> nexdash) can be followed. The
reply echoes the correlation id and either a result or an error with a stable code.

The token of a request is part of the envelope model (``token``) but never part of its JSON:
it is excluded from ``model_dump``, hidden from ``repr`` and, over HTTP, travels in the
``Authorization`` header. Error messages are generic and never echo parameter values.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROTOCOL = "nexlink/1"
PROJECT_PATTERN = r"^[a-z][a-z0-9_-]{1,31}$"
ACTION_PATTERN = r"^[a-z][a-z0-9_]{0,31}\.[a-z][a-z0-9_]{0,63}$"
ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"
MAX_CONTEXT_KEYS = 20
MAX_CONTEXT_CHARS = 200


def new_id() -> str:
    return uuid.uuid4().hex


class ErrorCode(StrEnum):
    """Stable error codes of a NexLink reply."""

    INVALID_ENVELOPE = "invalid_envelope"
    WRONG_TARGET = "wrong_target"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    UNKNOWN_ACTION = "unknown_action"
    INVALID_PARAMS = "invalid_params"
    UNAVAILABLE = "unavailable"
    UPSTREAM_ERROR = "upstream_error"
    FAILED = "failed"


class Envelope(BaseModel):
    """A NexLink request (``kind="request"``) or a one-way notice (``kind="notice"``)."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["nexlink/1"] = PROTOCOL
    id: str = Field(default_factory=new_id, pattern=ID_PATTERN, description="id of this message")
    correlation_id: str | None = Field(
        default=None, pattern=ID_PATTERN, description="id shared by every message of one conversation"
    )
    kind: Literal["request", "notice"] = "request"
    sender: str = Field(pattern=PROJECT_PATTERN, description="project that sends the message")
    target: str = Field(pattern=PROJECT_PATTERN, description="project that should run the action")
    action: str = Field(pattern=ACTION_PATTERN, description="'<namespace>.<name>', listed in the target manifest")
    params: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, str] = Field(
        default_factory=dict,
        description="small references shared between projects (run id, floor, ticket id); never ticket text",
    )
    sent_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @field_validator("context")
    @classmethod
    def _small_context(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > MAX_CONTEXT_KEYS:
            raise ValueError(f"context holds at most {MAX_CONTEXT_KEYS} keys")
        for key, item in value.items():
            if not key or len(key) > 64 or len(item) > MAX_CONTEXT_CHARS:
                raise ValueError(f"context keys have 1-64 characters and values at most {MAX_CONTEXT_CHARS}")
        return value

    @property
    def conversation(self) -> str:
        """Correlation id of the conversation; a first request starts one with its own id."""
        return self.correlation_id or self.id

    def reply_ok(self, sender: str, result: dict[str, Any] | None) -> Reply:
        return Reply(
            in_reply_to=self.id,
            correlation_id=self.conversation,
            sender=sender,
            target=self.sender,
            action=self.action,
            status="ok",
            result=result,
        )

    def reply_error(self, sender: str, code: ErrorCode, message: str, details: Any = None) -> Reply:
        return Reply(
            in_reply_to=self.id,
            correlation_id=self.conversation,
            sender=sender,
            target=self.sender,
            action=self.action,
            status="error",
            error=ErrorInfo(code=code, message=message, details=details),
        )


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str
    details: Any = None


class Reply(BaseModel):
    """Answer to an envelope."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["nexlink/1"] = PROTOCOL
    id: str = Field(default_factory=new_id, pattern=ID_PATTERN)
    in_reply_to: str | None = Field(default=None, description="id of the request; None when it was unreadable")
    correlation_id: str | None = None
    sender: str
    target: str | None = None
    action: str | None = None
    status: Literal["ok", "error"]
    result: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    sent_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def ok(self) -> bool:
        return self.status == "ok"
