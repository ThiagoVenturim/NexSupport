"""Deflection and ROI of the support agent.

Input: the outcomes of the agent (``support_engine.agent.AgentResult`` or ``Outcome``,
anything with ``action``, ``level``, ``cached`` and ``attempts``). Per question:

- deflected: level 1 answers sent to the customer (cache hits too, with ``deflect_cached``),
  times ``confirmed_rate`` for the share that really resolves the question;
- assisted: level 2 drafts a person reviews instead of writing the answer;
- escalated and refused: no saving.

``hours_saved = deflected * handle_minutes / 60 + assisted * (handle_minutes - review_minutes) / 60``,
``gross_savings = hours_saved * agent_hourly_cost``, ``llm_cost = calls * llm_cost_per_call``
where ``calls = attempts * llm_calls_per_attempt`` for each answer that was not served from
the cache, ``net_savings = gross_savings - llm_cost - fixed_cost`` and
``roi = net_savings / (llm_cost + fixed_cost)`` (None when nothing was spent).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from support_engine.analytics.config import ROIConfig


class Outcome(BaseModel):
    """What the agent did with one question; the numbers ROI needs and nothing else."""

    action: str = Field(description="answer, review, escalate or refuse")
    level: int | None = None
    cached: bool = False
    attempts: int = Field(default=1, ge=0)

    @classmethod
    def of(cls, result: Any) -> Outcome:
        if isinstance(result, Outcome):
            return result
        if isinstance(result, dict):
            return cls.model_validate({k: result[k] for k in ("action", "level", "cached", "attempts") if k in result})
        return cls(
            action=str(result.action),
            level=getattr(result, "level", None),
            cached=bool(getattr(result, "cached", False)),
            attempts=int(getattr(result, "attempts", 1) or 0),
        )


class ROIReport(BaseModel):
    """Deflection rates and money. Money values are in ``currency``."""

    currency: str
    questions: int = 0
    answered: int = Field(default=0, description="level 1 answers")
    deflected: float = Field(default=0.0, description="answered times confirmed_rate")
    assisted: int = 0
    escalated: int = 0
    refused: int = 0
    cached: int = 0
    deflection_rate: float = 0.0
    assist_rate: float = 0.0
    escalation_rate: float = 0.0
    llm_calls: float = 0.0
    hours_saved: float = 0.0
    gross_savings: float = 0.0
    llm_cost: float = 0.0
    fixed_cost: float = 0.0
    net_savings: float = 0.0
    roi: float | None = Field(default=None, description="net savings per unit of cost")
    cost_per_question: float = 0.0
    baseline_cost: float = Field(default=0.0, description="cost if people handled every question")


def compute_roi(results: Iterable[Any], config: ROIConfig | None = None) -> ROIReport:
    """Deflection and ROI over agent outcomes."""
    cfg = config or ROIConfig()
    outcomes = [Outcome.of(r) for r in results]
    report = ROIReport(currency=cfg.currency, questions=len(outcomes), fixed_cost=round(cfg.fixed_cost, 4))
    calls = 0.0
    for o in outcomes:
        if o.cached:
            report.cached += 1
        else:
            calls += o.attempts * cfg.llm_calls_per_attempt
        if o.action == "answer" and o.level == 1 and (cfg.deflect_cached or not o.cached):
            report.answered += 1
        elif o.action == "review" or (o.action == "answer" and o.level == 2):
            report.assisted += 1
        elif o.action == "escalate":
            report.escalated += 1
        elif o.action == "refuse":
            report.refused += 1
    n = report.questions
    report.deflected = round(report.answered * cfg.confirmed_rate, 4)
    if n:
        report.deflection_rate = round(report.deflected / n, 4)
        report.assist_rate = round(report.assisted / n, 4)
        report.escalation_rate = round(report.escalated / n, 4)
    saved_minutes = report.deflected * cfg.handle_minutes
    saved_minutes += report.assisted * max(0.0, cfg.handle_minutes - cfg.review_minutes)
    report.hours_saved = round(saved_minutes / 60, 4)
    report.gross_savings = round(report.hours_saved * cfg.agent_hourly_cost, 4)
    report.llm_calls = round(calls, 4)
    report.llm_cost = round(calls * cfg.llm_cost_per_call, 4)
    spent = report.llm_cost + cfg.fixed_cost
    report.net_savings = round(report.gross_savings - spent, 4)
    report.roi = round(report.net_savings / spent, 4) if spent > 0 else None
    report.cost_per_question = round(spent / n, 4) if n else 0.0
    handled = n - report.refused
    report.baseline_cost = round(handled * cfg.handle_minutes / 60 * cfg.agent_hourly_cost, 4)
    return report
