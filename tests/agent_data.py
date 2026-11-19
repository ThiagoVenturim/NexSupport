"""Synthetic tickets and helpers shared by the support agent tests. All data is invented."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from support_engine.agent import Source
from support_engine.curation import CurationPipeline, CurationResult, Feedback
from support_engine.ingestion import IngestionPipeline
from support_engine.models import Comment, Ticket, TicketStatus

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def _ticket(tid: str, subject: str, problem: str, answer: str, **kwargs) -> Ticket:
    comments = [
        Comment(id=f"{tid}-c1", author_role="customer", body=problem, created_at=T0),
        Comment(id=f"{tid}-c2", author_role="agent", body=answer, created_at=T0 + timedelta(hours=1)),
    ]
    base = {
        "id": tid,
        "source": "csv",
        "subject": subject,
        "status": TicketStatus.SOLVED,
        "csat": 1.0,
        "created_at": T0,
        "comments": comments,
    }
    base.update(kwargs)
    return Ticket(**base)


def tickets() -> list[Ticket]:
    return [
        _ticket(
            "T-1",
            "Invoice export fails with error 500",
            "The monthly invoice export fails with error 500 when I click Export PDF.",
            "Clear the export cache in Settings, then run the invoice export again.",
        ),
        _ticket(
            "T-2",
            "SSO login redirect loop",
            "Users are redirected back to the login screen after the SAML assertion.",
            "Enable the five minute clock drift tolerance on the SAML verifier, then sign in again.",
        ),
        _ticket(
            "T-3",
            "Password reset email never arrives",
            "The password reset email never arrives in my inbox.",
            "Check the spam folder and add the sender to the allow list, then request a new reset link.",
            status=TicketStatus.OPEN,
            csat=None,
        ),
    ]


def curated(feedback: Sequence[Feedback] = ()) -> tuple[CurationPipeline, CurationResult]:
    ingested = IngestionPipeline().process(tickets())
    pipeline = CurationPipeline()
    return pipeline, pipeline.curate(ingested.tickets, ingested.chunks, feedback)


class RecordingGenerator:
    """Scripted answers; records what the agent sent to the model."""

    def __init__(self, answers: Sequence[str | Exception]) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []

    async def agenerate(self, question: str, sources: Sequence[Source], feedback: Sequence[str] = ()) -> str:
        self.calls.append({"question": question, "sources": list(sources), "feedback": list(feedback)})
        reply = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply
