"""The support agent as a LangGraph state graph.

::

    guard_input --refuse--> refuse --> END
        |  \\--escalate--> escalate --> END
        v
    cache_lookup --hit--> END
        v
    retrieve --no documents--> escalate
        v
    generate <--------------------------+
        v                               | explanatory retry (problems or judge reason)
    check_citations --fails, retries----+
        v                               |
    judge --score below level 1, retries+
        v
    guard_output --> decide --> END   (level 1 answers are stored in the semantic cache)

Levels: 1 sends the answer to the customer, 2 holds it as a draft for a support agent,
3 escalates to a person. The graph state and the logs carry the masked question only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from support_engine.agent.cache import SemanticCache
from support_engine.agent.citations import Citation, CitationReport, check_citations, citations_for
from support_engine.agent.config import AgentConfig
from support_engine.agent.guards import Guard
from support_engine.agent.llm import AnswerGenerator, AnswerJudge, JudgeResult, LLMAnswerGenerator, LLMAnswerJudge
from support_engine.agent.retrieval import LexicalRetriever, Retriever, Source
from support_engine.curation import CurationPipeline
from support_engine.ingestion import Chunk

logger = logging.getLogger(__name__)

Action = Literal["answer", "review", "escalate", "refuse"]


class AgentState(TypedDict, total=False):
    question: str
    namespace: str
    route: str
    flags: list[str]
    reasons: list[str]
    blocked: str | None
    sources: list[Source]
    dropped_sources: int
    answer: str
    feedback: list[str]
    attempts: int
    report: CitationReport | None
    judge: JudgeResult | None
    citations: list[Citation]
    cached: bool
    action: Action
    level: int | None
    reply: str | None


class AgentResult(BaseModel):
    """Outcome of one question. Text fields hold masked text only."""

    question: str = Field(description="masked question")
    action: Action
    level: int | None = Field(default=None, description="1 answer, 2 review by an agent, 3 escalated; None if refused")
    reply: str | None = Field(default=None, description="message that can go to the customer now")
    answer: str | None = Field(default=None, description="generated answer; a draft when level is 2 or 3")
    citations: list[Citation] = Field(default_factory=list)
    citation_report: CitationReport | None = None
    judge: JudgeResult | None = None
    attempts: int = 0
    cached: bool = False
    sources: int = 0
    dropped_sources: int = 0
    flags: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @property
    def escalated(self) -> bool:
        return self.action == "escalate"


class SupportAgent:
    """Answers customer questions from curated tickets, with citations, a judge, guards and levels."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        llm: Any = None,
        judge_llm: Any = None,
        generator: AnswerGenerator | None = None,
        judge: AnswerJudge | None = None,
        config: AgentConfig | None = None,
        guard: Guard | None = None,
        cache: SemanticCache | None = None,
        curation: CurationPipeline | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.retriever = retriever
        self.generator = generator or LLMAnswerGenerator(llm, tier=self.config.generator_tier)
        self.judge = judge or LLMAnswerJudge(judge_llm if judge_llm is not None else llm, tier=self.config.judge_tier)
        self.guard = guard or Guard(self.config.guards)
        self.cache = cache if cache is not None else SemanticCache(self.config.cache)
        self.curation = curation
        self.graph = self._build().compile()

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk], *, curation: CurationPipeline | None = None, **kwargs: Any):
        """Agent over curated chunks with the offline lexical retriever."""
        pipeline = curation or CurationPipeline()
        return cls(LexicalRetriever(chunks, curation=pipeline), curation=pipeline, **kwargs)

    # Nodes ----------------------------------------------------------------------------------

    def _guard_input(self, state: AgentState) -> AgentState:
        verdict = self.guard.check_input(state.get("question", ""))
        route = "refuse" if verdict.blocked else ("escalate" if verdict.escalate else "continue")
        return {
            "question": verdict.question,
            "flags": list(verdict.flags),
            "reasons": list(verdict.escalate),
            "blocked": verdict.blocked,
            "route": route,
        }

    def _cache_lookup(self, state: AgentState) -> AgentState:
        hit = self.cache.get(state["question"], state.get("namespace", "default"))
        if hit is None:
            return {"route": "miss"}
        entry = hit.entry
        return {
            "route": "hit",
            "cached": True,
            "answer": entry.answer,
            "citations": list(entry.citations),
            "judge": JudgeResult(score=entry.score, reason="cached") if entry.score is not None else None,
            "action": "answer",
            "level": 1,
            "reply": entry.answer,
            "flags": [*state.get("flags", []), "cache_hit"],
        }

    def _retrieve(self, state: AgentState) -> AgentState:
        cfg = self.config.retrieval
        try:
            hits = self.retriever(state["question"], cfg.k)
        except Exception as exc:
            logger.warning("agent retrieval failed: %s", type(exc).__name__)
            return {"sources": [], "route": "none", "reasons": [*state.get("reasons", []), "retrieval_failed"]}
        sources: list[Source] = []
        dropped = 0
        for chunk, relevance in hits:
            if relevance < cfg.min_relevance:
                continue
            if not self.guard.is_safe_source(chunk.text):
                dropped += 1
                continue
            sources.append(Source.from_chunk(f"S{len(sources) + 1}", chunk, relevance, cfg.max_source_chars))
        flags = list(state.get("flags", []))
        if dropped:
            flags.append("unsafe_source_dropped")
        update: AgentState = {"sources": sources, "dropped_sources": dropped, "flags": flags}
        if not sources:
            update["reasons"] = [*state.get("reasons", []), "no_documents"]
            update["route"] = "none"
        else:
            update["route"] = "found"
        return update

    async def _generate(self, state: AgentState) -> AgentState:
        attempts = state.get("attempts", 0) + 1
        try:
            answer = await self.generator.agenerate(state["question"], state["sources"], state.get("feedback", []))
        except Exception as exc:
            logger.warning("agent generation failed: %s", type(exc).__name__)
            return {
                "attempts": attempts,
                "route": "failed",
                "reasons": [*state.get("reasons", []), "generation_failed"],
            }
        return {"attempts": attempts, "answer": answer, "route": "ok"}

    def _retries_left(self, state: AgentState) -> bool:
        return state.get("attempts", 0) <= self.config.max_retries

    def _check_citations(self, state: AgentState) -> AgentState:
        report = check_citations(state.get("answer", ""), state["sources"], self.config.citations)
        update: AgentState = {"report": report}
        if not report.ok and self._retries_left(state):
            update["feedback"] = [*state.get("feedback", []), report.explanation()]
            update["route"] = "retry"
        else:
            update["route"] = "judge"
        return update

    async def _judge(self, state: AgentState) -> AgentState:
        result = await self.judge.ajudge(state["question"], state.get("answer", ""), state["sources"])
        update: AgentState = {"judge": result, "route": "output"}
        report = state.get("report")
        if (
            result is not None
            and result.score < self.config.levels.level1_min_score
            and report is not None
            and report.ok
            and self._retries_left(state)
        ):
            reason = result.reason.strip() or "The answer does not fully resolve the question."
            update["feedback"] = [*state.get("feedback", []), f"Reviewer: {reason}"]
            update["route"] = "retry"
        return update

    def _guard_output(self, state: AgentState) -> AgentState:
        verdict = self.guard.check_output(state.get("answer", ""))
        report = state.get("report") or CitationReport()
        return {
            "answer": verdict.answer,
            "flags": [*state.get("flags", []), *verdict.flags],
            "citations": citations_for(report, state["sources"]),
        }

    def _decide(self, state: AgentState) -> AgentState:
        levels = self.config.levels
        flags = state.get("flags", [])
        reasons = list(state.get("reasons", []))
        report = state.get("report")
        result = state.get("judge")
        answer = state.get("answer", "")

        if "injection_in_answer" in flags:
            level, reasons = 3, [*reasons, "unsafe_answer"]
        elif not answer or report is None or not report.ok:
            level, reasons = 3, [*reasons, "ungrounded"]
        elif result is None:
            level, reasons = 2, [*reasons, "judge_unavailable"]
        elif result.score >= levels.level1_min_score and result.grounded and "pii_in_answer" not in flags:
            level = 1
        elif result.score >= levels.level2_min_score:
            level = 2
            if "pii_in_answer" in flags:
                reasons.append("pii_in_answer")
            if not result.grounded:
                reasons.append("judge_not_grounded")
            if result.score < levels.level1_min_score:
                reasons.append("needs_review")
        else:
            level, reasons = 3, [*reasons, "low_score"]

        if level == 1:
            self.cache.put(
                state["question"],
                answer,
                state.get("namespace", "default"),
                citations=state.get("citations", []),
                score=result.score if result else None,
            )
            return {"action": "answer", "level": 1, "reply": answer, "reasons": reasons}
        if level == 2:
            return {"action": "review", "level": 2, "reply": None, "reasons": reasons}
        self._record_gap(state["question"], reasons)
        return {"action": "escalate", "level": 3, "reply": self.config.messages.escalate, "reasons": reasons}

    def _escalate(self, state: AgentState) -> AgentState:
        reasons = state.get("reasons", [])
        self._record_gap(state.get("question", ""), reasons)
        return {"action": "escalate", "level": 3, "reply": self.config.messages.escalate}

    def _refuse(self, state: AgentState) -> AgentState:
        messages = self.config.messages
        reply = {
            "injection": messages.refuse_injection,
            "too_long": messages.refuse_too_long,
        }.get(state.get("blocked") or "", messages.refuse_out_of_scope)
        return {"action": "refuse", "level": None, "reply": reply, "reasons": [state.get("blocked") or "refused"]}

    def _record_gap(self, question: str, reasons: Sequence[str]) -> None:
        """Questions the knowledge base could not answer feed curation's gap detector."""
        if self.curation is not None and question and {"no_documents", "low_score"} & set(reasons):
            self.curation.gaps.add(question)

    # Graph ----------------------------------------------------------------------------------

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("guard_input", self._guard_input)
        g.add_node("cache_lookup", self._cache_lookup)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_node("check_citations", self._check_citations)
        g.add_node("judge", self._judge)
        g.add_node("guard_output", self._guard_output)
        g.add_node("decide", self._decide)
        g.add_node("escalate", self._escalate)
        g.add_node("refuse", self._refuse)

        def route(state: AgentState) -> str:
            return state.get("route", "")

        g.set_entry_point("guard_input")
        g.add_conditional_edges(
            "guard_input", route, {"refuse": "refuse", "escalate": "escalate", "continue": "cache_lookup"}
        )
        g.add_conditional_edges("cache_lookup", route, {"hit": END, "miss": "retrieve"})
        g.add_conditional_edges("retrieve", route, {"none": "escalate", "found": "generate"})
        g.add_conditional_edges("generate", route, {"failed": "escalate", "ok": "check_citations"})
        g.add_conditional_edges("check_citations", route, {"retry": "generate", "judge": "judge"})
        g.add_conditional_edges("judge", route, {"retry": "generate", "output": "guard_output"})
        g.add_edge("guard_output", "decide")
        g.add_edge("decide", END)
        g.add_edge("escalate", END)
        g.add_edge("refuse", END)
        return g

    # Entry points ---------------------------------------------------------------------------

    async def aanswer(self, question: str, *, namespace: str = "default") -> AgentResult:
        """Run the graph for one question."""
        state: AgentState = await self.graph.ainvoke(
            {"question": question, "namespace": namespace, "attempts": 0, "feedback": [], "flags": [], "reasons": []}
        )
        answer = state.get("answer") or None
        result = AgentResult(
            question=state.get("question", ""),
            action=state.get("action", "escalate"),
            level=state.get("level"),
            reply=state.get("reply"),
            answer=answer,
            citations=state.get("citations", []),
            citation_report=state.get("report"),
            judge=state.get("judge"),
            attempts=state.get("attempts", 0),
            cached=state.get("cached", False),
            sources=len(state.get("sources", [])),
            dropped_sources=state.get("dropped_sources", 0),
            flags=list(dict.fromkeys(state.get("flags", []))),
            reasons=list(dict.fromkeys(state.get("reasons", []))),
        )
        # Fixed labels and counts only: no question or answer text in the logs.
        logger.info(
            "agent: action=%s level=%s attempts=%d sources=%d cached=%s flags=%s reasons=%s",
            result.action,
            result.level,
            result.attempts,
            result.sources,
            result.cached,
            ",".join(result.flags) or "-",
            ",".join(result.reasons) or "-",
        )
        return result

    def answer(self, question: str, *, namespace: str = "default") -> AgentResult:
        """Synchronous wrapper around ``aanswer`` (not for use inside a running event loop)."""
        return asyncio.run(self.aanswer(question, namespace=namespace))
