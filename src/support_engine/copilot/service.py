"""Request models and operations shared by the REST API and the MCP server.

Both interfaces validate input with the same models and call the same operations, so a client
gets the same answer whichever door it uses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from support_engine.copilot.assist import Copilot
from support_engine.models import Ticket

NAMESPACE_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"


class SuggestRequest(BaseModel):
    ticket: Ticket
    namespace: str = Field(default="default", pattern=NAMESPACE_PATTERN, description="cache partition")
    now: datetime | None = Field(default=None, description="time the risk is computed for (default: latest event)")


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    namespace: str = Field(default="default", pattern=NAMESPACE_PATTERN)


class SummaryRequest(BaseModel):
    ticket: Ticket


class FeedbackRequest(BaseModel):
    ticket_ids: list[str] = Field(min_length=1, max_length=20, description="tickets the suggestion cited")
    accepted: bool
    correction: str | None = Field(default=None, max_length=8_000, description="reply as the agent sent it")
    query: str | None = Field(default=None, max_length=8_000, description="question the suggestion answered")
    reviewer_role: Literal["agent", "reviewer", "customer"] = "agent"


class FeedbackResponse(BaseModel):
    feedback: list[str] = Field(default_factory=list, description="ids of the feedback records")
    rating: float | None = None
    stored: bool = False


def validation_summary(exc: ValidationError) -> list[dict[str, Any]]:
    """Field locations and error types only; pydantic's default output echoes the input values."""
    return [{"loc": [str(p) for p in err.get("loc", ())], "type": err.get("type", "invalid")} for err in exc.errors()]


class CopilotService:
    """The operations exposed over REST and MCP."""

    def __init__(self, copilot: Copilot) -> None:
        self.copilot = copilot

    async def suggest(self, request: SuggestRequest) -> dict[str, Any]:
        result = await self.copilot.asuggest(request.ticket, namespace=request.namespace, now=request.now)
        return result.model_dump(mode="json")

    async def answer(self, request: AnswerRequest) -> dict[str, Any]:
        result = await self.copilot.aanswer(request.question, namespace=request.namespace)
        return result.model_dump(mode="json")

    async def summarize(self, request: SummaryRequest) -> dict[str, Any]:
        result = await self.copilot.asummarize(request.ticket)
        return result.model_dump(mode="json")

    async def feedback(self, request: FeedbackRequest) -> dict[str, Any]:
        items = self.copilot.record_feedback(
            request.ticket_ids,
            accepted=request.accepted,
            correction=request.correction,
            query=request.query,
            reviewer_role=request.reviewer_role,
        )
        response = FeedbackResponse(
            feedback=[f.id for f in items],
            rating=items[0].rating if items else None,
            stored=bool(items) and self.copilot.feedback_store is not None,
        )
        return response.model_dump(mode="json")
