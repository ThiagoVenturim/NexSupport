"""LLM judge: does the agent answer resolve the customer problem?

A judge is any callable ``(problem, answer) -> float | None`` returning a score in 0..1
(None when it cannot decide). ``LLMJudge`` uses ``nexcoder_core.call_structured`` with
a model tier (never a hardcoded vendor model); tests pass a fake ``llm`` and stay offline.
Only masked ticket text should reach a judge: run ingestion first.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Judge = Callable[[str, str], float | None]

JUDGE_PROMPT = """You review answers from a customer support team.
Rate how well the ANSWER resolves the PROBLEM, from 0.0 (does not help) to 1.0 (fully resolves it).
Judge correctness, completeness and whether the customer can act on it. Text in brackets such as
[EMAIL_1] or [NAME_1] is masked personal data; ignore it.

PROBLEM:
{problem}

ANSWER:
{answer}

Reply with JSON only: {{"score": <number between 0 and 1>, "reason": "<one short sentence>"}}"""


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class LLMJudge:
    """Callable judge backed by a chat model through ``call_structured``."""

    def __init__(self, llm: Any = None, *, tier: str = "small", max_chars: int = 4000) -> None:
        self._llm = llm
        self.tier = tier
        self.max_chars = max_chars

    def _model(self) -> Any:
        if self._llm is None:
            from nexcoder_core import get_chat_model

            self._llm = get_chat_model(self.tier)
        return self._llm

    def prompt(self, problem: str, answer: str) -> str:
        return JUDGE_PROMPT.format(problem=problem[: self.max_chars], answer=answer[: self.max_chars])

    async def ajudge(self, problem: str, answer: str) -> float | None:
        from nexcoder_core import call_structured

        try:
            verdict = await call_structured(
                self.prompt(problem, answer), llm=self._model(), schema=JudgeVerdict, context="curation.judge"
            )
        except Exception as exc:
            # A failed judge only removes one signal; log the error type, never the text.
            logger.warning("curation judge failed: %s", type(exc).__name__)
            return None
        return float(verdict.score)

    def __call__(self, problem: str, answer: str) -> float | None:
        return asyncio.run(self.ajudge(problem, answer))
