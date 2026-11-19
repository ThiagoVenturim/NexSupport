"""Built-in steps: the support flow ingestion -> cleanup/PII -> chunking/dedup -> curation -> RAG -> answer.

Each step wraps a module that already exists (``ingestion``, ``curation``, ``agent``,
``copilot``, ``analytics``) and takes that module's own configuration model as its
parameters, so a flow file uses the same names as the Python API.

Context keys a step can read or write:

``tickets``      tickets of the run (cleaned and masked once those steps ran)
``chunks``       retrievable pieces of the tickets
``scores``       resolution score per ticket (curation)
``retriever``    the knowledge base the answer steps search (RAG)
``answers``      support agent results for the questions of the run
``suggestions``  copilot suggestions for open tickets
``analytics``    the analytics report

Counters returned by the steps hold numbers and labels only, never ticket text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from support_engine.agent import AgentConfig, JudgeResult, LexicalRetriever, Source, SupportAgent, labels_in
from support_engine.agent.citations import split_sentences
from support_engine.agent.llm import LLMAnswerGenerator, LLMAnswerJudge
from support_engine.analytics import AnalyticsConfig, AnalyticsPipeline
from support_engine.copilot import Copilot, CopilotConfig
from support_engine.curation import CurationConfig, CurationPipeline
from support_engine.ingestion import (
    ChunkingConfig,
    CleanupConfig,
    DedupConfig,
    IngestionConfig,
    IngestionPipeline,
    IngestionStats,
    NearDuplicateIndex,
    PrivacyConfig,
    chunk_ticket,
)
from support_engine.pipelines.catalog import StepSpec

if TYPE_CHECKING:
    from support_engine.pipelines.engine import PipelineContext

ANSWER_ROLES: frozenset[str] = frozenset({"agent", "reviewer", "system"})
_RESOLUTION = re.compile(r"^Resolution:\s*(.+)$", re.MULTILINE)


# Offline answer generator and judge --------------------------------------------------------


class ExtractiveAnswerGenerator:
    """Answers with sentences copied from the best answer-like sources, each one cited.

    Needs no model, so a flow runs offline. Answer-like sources are human corrections, the
    resolution line of case summaries and interactions written by an agent (``roles`` maps a
    chunk id to its author role). Customer messages are never quoted as answers. When no
    source holds an answer it returns an empty text, which the agent escalates.
    """

    def __init__(self, roles: Mapping[str, str] | None = None, *, max_sources: int = 2, max_sentences: int = 2) -> None:
        self.roles = dict(roles or {})
        self.max_sources = max_sources
        self.max_sentences = max_sentences

    def answer_text(self, source: Source) -> str:
        if source.kind == "correction":
            return source.text
        if source.kind == "summary":
            match = _RESOLUTION.search(source.text)
            return match.group(1) if match else ""
        return source.text if self.roles.get(source.chunk_id) in ANSWER_ROLES else ""

    async def agenerate(self, question: str, sources: Sequence[Source], feedback: Sequence[str] = ()) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        used = 0
        for source in sources:
            if used >= self.max_sources:
                break
            sentences = [s for s in split_sentences(self.answer_text(source)) if s.strip()]
            fresh = [s for s in sentences if " ".join(s.lower().split()) not in seen][: self.max_sentences]
            if not fresh:
                continue
            used += 1
            for sentence in fresh:
                seen.add(" ".join(sentence.lower().split()))
                lines.append(f"{sentence.rstrip(' .!?')} [{source.label}].")
        return " ".join(lines)


class PriorityJudge:
    """Scores an answer with the curation priority of the sources it cites (no model call).

    Curated, proven answers and human corrections carry a high priority, so they reach level 1;
    uncurated sources sit at the neutral 0.5 and go to review.
    """

    async def ajudge(self, question: str, answer: str, sources: Sequence[Source]) -> JudgeResult | None:
        by_label = {s.label.upper(): s for s in sources}
        cited = [by_label[label] for label in labels_in(answer) if label in by_label]
        if not cited:
            return JudgeResult(score=0.0, grounded=False, reason="The answer cites no retrieved document.")
        score = max(min(1.0, max(0.0, s.priority)) for s in cited)
        return JudgeResult(
            score=round(score, 6), grounded=True, reason="Score is the curation priority of the sources."
        )


# Parameters ------------------------------------------------------------------------------


class ChunkingParams(ChunkingConfig):
    """``ChunkingConfig`` plus the interaction filters that run before chunking."""

    drop_empty_comments: bool = True
    dedup_comments: bool = Field(default=True, description="drop repeated interactions inside a ticket")


class DedupParams(DedupConfig):
    """``DedupConfig`` plus the scope of the near-duplicate search."""

    scope: Literal["ticket", "corpus"] = "corpus"


class IndexParams(BaseModel):
    k1: float = Field(default=1.5, gt=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


Generator = Literal["extractive", "llm"]
JudgeKind = Literal["priority", "llm"]


class AnswerParams(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    generator: Generator = Field(default="extractive", description="extractive runs offline; llm calls a model tier")
    judge: JudgeKind = Field(default="priority", description="priority runs offline; llm calls a model tier")
    max_sources: int = Field(default=2, ge=1, le=10, description="extractive generator: sources quoted")
    max_sentences: int = Field(default=2, ge=1, le=10, description="extractive generator: sentences per source")
    open_tickets: bool = Field(default=False, description="also answer the latest customer message of open tickets")
    namespace: str = Field(default="default", pattern=r"^[A-Za-z0-9_.:-]{1,64}$")


class CopilotParams(BaseModel):
    copilot: CopilotConfig = Field(default_factory=CopilotConfig)
    generator: Generator = "extractive"
    judge: JudgeKind = "priority"
    max_sources: int = Field(default=2, ge=1, le=10)
    max_sentences: int = Field(default=2, ge=1, le=10)
    include_unresolved: bool = Field(
        default=True, description="when the run has no open tickets, suggest for unresolved tickets of the corpus"
    )
    namespace: str = Field(default="default", pattern=r"^[A-Za-z0-9_.:-]{1,64}$")


# Helpers ---------------------------------------------------------------------------------


def _counters(stats: IngestionStats) -> dict[str, int]:
    return {f"pii_{k}": v for k, v in sorted(stats.pii_masked.items())}


def _build_agent(ctx: PipelineContext, config: AgentConfig, params: AnswerParams | CopilotParams) -> SupportAgent:
    services = ctx.services
    llm = services.get("llm")
    if services.get("generator") is not None:
        generator = services["generator"]
    elif params.generator == "llm":
        generator = LLMAnswerGenerator(llm, tier=config.generator_tier)
    else:
        roles = {c.id: str(c.metadata.get("author_role", "")) for c in ctx.chunks or []}
        generator = ExtractiveAnswerGenerator(roles, max_sources=params.max_sources, max_sentences=params.max_sentences)
    if services.get("judge") is not None:
        judge = services["judge"]
    elif params.judge == "llm":
        judge = LLMAnswerJudge(services.get("judge_llm", llm), tier=config.judge_tier)
    else:
        judge = PriorityJudge()
    return SupportAgent(
        ctx.retriever,
        generator=generator,
        judge=judge,
        config=config,
        curation=ctx.curation or CurationPipeline(),
    )


def _action_counts(actions: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for action in actions:
        out[action] = out.get(action, 0) + 1
    return out


# Steps -----------------------------------------------------------------------------------


def run_cleanup(ctx: PipelineContext, params: CleanupConfig) -> dict[str, Any]:
    pipeline = IngestionPipeline(IngestionConfig(cleanup=params))
    ctx.tickets = [pipeline.clean_ticket(t) for t in ctx.tickets]
    return {"tickets": len(ctx.tickets)}


def run_pii_mask(ctx: PipelineContext, params: PrivacyConfig) -> dict[str, Any]:
    pipeline = IngestionPipeline(IngestionConfig(privacy=params), name_detectors=ctx.services.get("name_detectors", ()))
    stats = IngestionStats()
    masked = []
    for ticket in ctx.tickets:
        out, counts = pipeline.mask_ticket(ticket)
        for kind, n in counts.items():
            stats.pii_masked[kind] = stats.pii_masked.get(kind, 0) + n
        masked.append(out)
    ctx.tickets = masked
    if params.mask_pii:
        ctx.masked = True
    return {"tickets": len(masked), **_counters(stats)}


def run_chunking(ctx: PipelineContext, params: ChunkingParams) -> dict[str, Any]:
    chunking = ChunkingConfig.model_validate(params.model_dump(include=set(ChunkingConfig.model_fields)))
    pipeline = IngestionPipeline(
        IngestionConfig(
            chunking=chunking, drop_empty_comments=params.drop_empty_comments, dedup_comments=params.dedup_comments
        ),
        summarizer=ctx.services.get("summarizer"),
    )
    stats = IngestionStats()
    tickets, chunks = [], []
    for ticket in ctx.tickets:
        stats.comments_in += len(ticket.comments)
        kept = pipeline._dedup_comments(ticket, stats)
        tickets.append(kept)
        chunks.extend(chunk_ticket(kept, chunking, pipeline.summarizer))
    ctx.tickets, ctx.chunks = tickets, chunks
    return {
        "tickets": len(tickets),
        "comments_kept": stats.comments_kept,
        "comments_dropped": stats.comments_empty + stats.comments_duplicate,
        "chunks": len(chunks),
    }


def run_dedup(ctx: PipelineContext, params: DedupParams) -> dict[str, Any]:
    config = DedupConfig.model_validate(params.model_dump(include=set(DedupConfig.model_fields)))
    chunks = list(ctx.chunks or [])
    if not config.enabled:
        return {"chunks": len(chunks), "duplicates": 0}
    kept, duplicates = [], 0
    indexes: dict[str, NearDuplicateIndex] = {}
    for chunk in chunks:
        if chunk.kind != "interaction":
            kept.append(chunk)
            continue
        key = "corpus" if params.scope == "corpus" else chunk.ticket_id
        index = indexes.setdefault(key, NearDuplicateIndex(config))
        if index.add(chunk.id, chunk.text) is not None:
            duplicates += 1
            continue
        kept.append(chunk)
    ctx.chunks = kept
    return {"chunks": len(kept), "duplicates": duplicates}


def run_curation(ctx: PipelineContext, params: CurationConfig) -> dict[str, Any]:
    pipeline = CurationPipeline(params, judge=ctx.services.get("curation_judge"))
    result = pipeline.curate(ctx.tickets, ctx.chunks or [], ctx.feedback)
    ctx.curation, ctx.curation_result = pipeline, result
    ctx.chunks, ctx.scores = result.chunks, result.scores
    stats = result.stats
    return {
        "chunks": stats.chunks_out,
        "below_min": stats.chunks_below_min,
        "superseded": stats.chunks_superseded,
        "corrections": stats.corrections_added,
        "suggested_articles": len(result.suggestions),
    }


def run_index(ctx: PipelineContext, params: IndexParams) -> dict[str, Any]:
    ctx.retriever = LexicalRetriever(
        ctx.chunks or [], curation=ctx.curation or CurationPipeline(), k1=params.k1, b=params.b
    )
    return {"documents": len(ctx.retriever)}


def run_answer(ctx: PipelineContext, params: AnswerParams) -> dict[str, Any]:
    agent = _build_agent(ctx, params.agent, params)
    ctx.agent = agent
    questions = list(ctx.questions)
    if params.open_tickets:
        copilot = Copilot(agent)
        questions += [copilot.question_for(t) for t in ctx.open_tickets]
    results = [agent.answer(q, namespace=params.namespace) for q in questions if q.strip()]
    ctx.answers.extend(results)
    return {"questions": len(results), **_action_counts([r.action for r in results])}


def run_copilot(ctx: PipelineContext, params: CopilotParams) -> dict[str, Any]:
    agent = _build_agent(ctx, params.copilot.agent, params)
    copilot = Copilot(agent, config=params.copilot, summarizer=ctx.services.get("copilot_summarizer"))
    targets = list(ctx.open_tickets)
    if not targets and params.include_unresolved:
        targets = [t for t in ctx.tickets if not t.is_resolved]
    suggestions = [copilot.suggest(t, namespace=params.namespace, now=ctx.now) for t in targets]
    ctx.suggestions.extend(suggestions)
    return {"tickets": len(suggestions), **_action_counts([s.action for s in suggestions])}


def run_analytics(ctx: PipelineContext, params: AnalyticsConfig) -> dict[str, Any]:
    pipeline = AnalyticsPipeline(params, llm=ctx.services.get("llm"))
    report = pipeline.run(ctx.tickets, ctx.answers or None, now=ctx.now)
    ctx.analytics = report
    return {
        "tickets": report.tickets,
        "negative": report.sentiment.negative,
        "alerts": len(report.alerts),
        "high_risk": len(report.high_risk()),
    }


def builtin_steps() -> list[StepSpec]:
    """The steps that ship with the engine, in the order of the full flow."""
    return [
        StepSpec(
            name="cleanup",
            run=run_cleanup,
            description="Normalise text, strip HTML, quoted replies, signatures and disclaimers.",
            category="ingestion",
            params=CleanupConfig,
            requires=("tickets",),
            provides=("tickets",),
        ),
        StepSpec(
            name="pii_mask",
            run=run_pii_mask,
            description="Mask e-mails, phones, documents, cards and personal names in every ticket field.",
            category="ingestion",
            params=PrivacyConfig,
            requires=("tickets",),
            provides=("tickets",),
        ),
        StepSpec(
            name="chunking",
            run=run_chunking,
            description="Drop empty and repeated interactions, then split tickets into interaction and summary chunks.",
            category="ingestion",
            params=ChunkingParams,
            requires=("tickets",),
            provides=("tickets", "chunks"),
            needs_masked=True,
        ),
        StepSpec(
            name="dedup",
            run=run_dedup,
            description="Remove near-duplicate interaction chunks (MinHash) per ticket or across the corpus.",
            category="ingestion",
            params=DedupParams,
            requires=("chunks",),
            provides=("chunks",),
        ),
        StepSpec(
            name="curation",
            run=run_curation,
            description="Score answers, apply human feedback and corrections, drop weak chunks, find gaps.",
            category="curation",
            params=CurationConfig,
            requires=("tickets", "chunks"),
            provides=("chunks", "scores"),
            needs_masked=True,
        ),
        StepSpec(
            name="index",
            run=run_index,
            description="Build the knowledge base (BM25 reranked by curation priority) the answer steps search.",
            category="rag",
            params=IndexParams,
            requires=("chunks",),
            provides=("retriever",),
        ),
        StepSpec(
            name="answer",
            run=run_answer,
            description="Answer the questions of the run with the support agent: citations, judge, guards, levels.",
            category="response",
            params=AnswerParams,
            requires=("retriever",),
            provides=("answers",),
        ),
        StepSpec(
            name="copilot",
            run=run_copilot,
            description="Suggest replies, summaries, next steps and tone for open tickets.",
            category="response",
            params=CopilotParams,
            requires=("retriever",),
            provides=("suggestions",),
        ),
        StepSpec(
            name="analytics",
            run=run_analytics,
            description="Sentiment, trends, risk radar, service quality, root cause and ROI of the run.",
            category="analytics",
            params=AnalyticsConfig,
            requires=("tickets",),
            provides=("analytics",),
            needs_masked=True,
        ),
    ]
