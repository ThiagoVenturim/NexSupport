"""Synthetic open tickets and a copilot factory for the copilot tests. All data is invented."""

from __future__ import annotations

import json
from datetime import timedelta

from agent_data import T0, RecordingGenerator, tickets
from nexcoder_core.llm.fake import FakeChatModel

from support_engine.agent import AgentConfig
from support_engine.copilot import Copilot, CopilotConfig
from support_engine.models import Comment, Ticket, TicketStatus

GOOD = "Clear the export cache in Settings, then run the invoice export again [S1, S2]."
EMAIL = "maria.example@example.com"
TOKEN = "test-token-0123456789abcdef"


def verdict(score: float, reason: str = "ok", grounded: bool = True) -> str:
    return json.dumps({"score": score, "grounded": grounded, "reason": reason})


def open_ticket(body: str | None = None, *, tid: str = "N-1", **kwargs) -> Ticket:
    first = f"Hello, I am writing from {EMAIL}. The invoice export fails with error 500 when I click Export PDF."
    last = body or "This is terrible, the invoice export still fails with error 500. What should I do?"
    comments = [
        Comment(id=f"{tid}-c1", author_role="customer", body=first, created_at=T0),
        Comment(
            id=f"{tid}-c2", author_role="agent", body="We are looking into it.", created_at=T0 + timedelta(hours=1)
        ),
        Comment(id=f"{tid}-c3", author_role="customer", body=last, created_at=T0 + timedelta(hours=2)),
    ]
    base = {
        "id": tid,
        "source": "csv",
        "subject": "Invoice export broken",
        "status": TicketStatus.OPEN,
        "created_at": T0,
        "comments": comments,
    }
    base.update(kwargs)
    return Ticket(**base)


def make_copilot(answers=(GOOD,), judge_replies=(verdict(0.95),), *, config: CopilotConfig | None = None, **kwargs):
    generator = RecordingGenerator(list(answers))
    judge_llm = FakeChatModel(responses=list(judge_replies))
    cfg = config or CopilotConfig(agent=AgentConfig(max_retries=1))
    copilot = Copilot.from_tickets(tickets(), config=cfg, generator=generator, judge_llm=judge_llm, **kwargs)
    return copilot, generator, judge_llm
