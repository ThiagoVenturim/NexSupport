"""Run a configured flow over tickets.

Two engines run the same steps and give the same result:

``nexcoder``  every step becomes a block of the LangGraph pipeline engine of
              ``nexcoder-agents`` (``PipelineBuilder``), the engine shared with the
              other projects of the suite;
``local``     a plain loop, for installs without ``nexcoder-agents``.

``auto`` (the default) picks ``nexcoder`` when the package is installed.

Objects (tickets, chunks, the knowledge base) stay in a ``PipelineContext`` owned by the
run; the graph state only carries step counters. Log lines and the step history hold
step names, counters and error types, never ticket text or exception messages.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from support_engine.agent import AgentResult, LexicalRetriever, SupportAgent
from support_engine.analytics import AnalyticsReport
from support_engine.copilot import CopilotSuggestion
from support_engine.curation import CurationPipeline, CurationResult, Feedback, ResolutionScore
from support_engine.ingestion import Chunk
from support_engine.models import Ticket
from support_engine.pipelines.catalog import StepCatalog, StepSpec, default_catalog
from support_engine.pipelines.config import PipelineConfig, check_flow, load_config

logger = logging.getLogger(__name__)

StepStatus = Literal["success", "failure", "skipped"]


@dataclass
class PipelineContext:
    """Everything the steps of one run share."""

    tickets: list[Ticket]
    feedback: list[Feedback] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    open_tickets: list[Ticket] = field(default_factory=list)
    now: datetime | None = None
    services: dict[str, Any] = field(default_factory=dict)
    masked: bool = False
    chunks: list[Chunk] | None = None
    scores: dict[str, ResolutionScore] = field(default_factory=dict)
    curation: CurationPipeline | None = None
    curation_result: CurationResult | None = None
    retriever: LexicalRetriever | None = None
    agent: SupportAgent | None = None
    answers: list[AgentResult] = field(default_factory=list)
    suggestions: list[CopilotSuggestion] = field(default_factory=list)
    analytics: AnalyticsReport | None = None


class StepRecord(BaseModel):
    """What one step did. Names, counters and an error type only."""

    name: str
    use: str
    status: StepStatus
    seconds: float = 0.0
    counters: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PipelineResult(BaseModel):
    """Outputs of a run. Ticket text in here is masked when the flow masks PII."""

    model_config = {"arbitrary_types_allowed": True}

    pipeline: str
    engine: str
    status: Literal["completed", "failed"]
    steps: list[StepRecord] = Field(default_factory=list)
    tickets: list[Ticket] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    curation: CurationResult | None = None
    answers: list[AgentResult] = Field(default_factory=list)
    suggestions: list[CopilotSuggestion] = Field(default_factory=list)
    analytics: AnalyticsReport | None = None
    retriever: Any = Field(default=None, exclude=True, description="knowledge base built by the index step")

    @property
    def errors(self) -> list[str]:
        return [f"{s.name}: {s.error}" for s in self.steps if s.error]

    def step(self, name: str) -> StepRecord:
        for record in self.steps:
            if record.name == name:
                return record
        raise KeyError(name)

    def summary(self) -> dict[str, Any]:
        """Counters of the run, safe to print or log."""
        return {
            "pipeline": self.pipeline,
            "engine": self.engine,
            "status": self.status,
            "steps": [r.model_dump(mode="json") for r in self.steps],
            "tickets": len(self.tickets),
            "chunks": len(self.chunks),
            "answers": len(self.answers),
            "suggestions": len(self.suggestions),
        }


def nexcoder_available() -> bool:
    """True when the ``nexcoder-agents`` pipeline engine can be imported."""
    try:
        return importlib.util.find_spec("nexcoder_agents.engine") is not None
    except (ImportError, ValueError):
        return False


class Pipeline:
    """A checked flow, ready to run any number of times."""

    def __init__(self, config: PipelineConfig, *, catalog: StepCatalog | None = None) -> None:
        self.catalog = catalog or default_catalog()
        self.config = config
        self.params = check_flow(config, self.catalog)
        self.steps: list[tuple[str, StepSpec]] = [(s.id, self.catalog.get(s.use)) for s in config.enabled_steps()]
        if config.engine == "nexcoder" and not nexcoder_available():
            raise ValueError("engine 'nexcoder' needs the nexcoder-agents package; install it or use engine: local")
        self.engine = (
            "nexcoder" if config.engine == "nexcoder" or (config.engine == "auto" and nexcoder_available()) else "local"
        )

    @classmethod
    def from_config(
        cls, source: str | Path | Mapping[str, Any] | PipelineConfig, *, catalog: StepCatalog | None = None
    ) -> Pipeline:
        """Flow from a preset name, a YAML or JSON file, a dict or a ``PipelineConfig``."""
        return cls(load_config(source), catalog=catalog)

    @property
    def name(self) -> str:
        return self.config.name

    # One step -----------------------------------------------------------------------------

    def _run_step(self, name: str, spec: StepSpec, ctx: PipelineContext) -> StepRecord:
        started = time.perf_counter()
        try:
            counters = spec.run(ctx, self.params[name]) or {}
            record = StepRecord(name=name, use=spec.name, status="success", counters=dict(counters))
        except Exception as exc:
            # The type only: messages can quote ticket text.
            logger.warning("pipeline %s: step %s failed: %s", self.name, name, type(exc).__name__)
            record = StepRecord(name=name, use=spec.name, status="failure", error=type(exc).__name__)
        record.seconds = round(time.perf_counter() - started, 4)
        logger.info("pipeline %s: step %s %s %s", self.name, name, record.status, record.counters)
        return record

    # Engines ------------------------------------------------------------------------------

    def _run_local(self, ctx: PipelineContext) -> list[StepRecord]:
        records: list[StepRecord] = []
        failed = False
        for name, spec in self.steps:
            if failed and self.config.stop_on_error:
                records.append(StepRecord(name=name, use=spec.name, status="skipped"))
                continue
            record = self._run_step(name, spec, ctx)
            failed = failed or record.status == "failure"
            records.append(record)
        return records

    def _run_nexcoder(self, ctx: PipelineContext) -> list[StepRecord]:
        from nexcoder_agents.engine import BaseBlock, BlockResult, BlockStatus, PipelineBuilder

        records: dict[str, StepRecord] = {}
        runner = self

        class StepBlock(BaseBlock):
            def __init__(self, name: str, spec: StepSpec) -> None:
                super().__init__(name=name, block_type=f"support:{spec.name}")
                self.spec = spec

            def execute(self, state: dict[str, Any], context: dict[str, Any] | None = None) -> BlockResult:
                record = runner._run_step(self.name, self.spec, ctx)
                records[self.name] = record
                ok = record.status == "success" or not runner.config.stop_on_error
                return BlockResult(
                    block_name=self.name,
                    status=BlockStatus.SUCCESS if ok else BlockStatus.FAILURE,
                    updates={f"step_{self.name}": record.counters},
                    error=None if ok else f"{self.name}: {record.error}",
                )

        builder = PipelineBuilder(self.name)
        builder.add_sequence([StepBlock(name, spec) for name, spec in self.steps])
        builder.compile().run({})
        return [
            records.get(name) or StepRecord(name=name, use=spec.name, status="skipped") for name, spec in self.steps
        ]

    # Run ----------------------------------------------------------------------------------

    def run(
        self,
        tickets: Iterable[Ticket],
        *,
        feedback: Iterable[Feedback] = (),
        questions: Iterable[str] = (),
        open_tickets: Iterable[Ticket] = (),
        now: datetime | None = None,
        **services: Any,
    ) -> PipelineResult:
        """Run the flow.

        ``questions`` go to the ``answer`` step, ``open_tickets`` to the ``copilot`` step.
        ``services`` replace the offline defaults of the steps: ``llm``, ``judge_llm``,
        ``generator``, ``judge`` (support agent), ``curation_judge``, ``summarizer``
        (ingestion case summary), ``copilot_summarizer`` and ``name_detectors``.
        Synchronous: do not call it inside a running event loop.
        """
        ctx = PipelineContext(
            tickets=list(tickets),
            feedback=list(feedback),
            questions=[q for q in questions if q],
            open_tickets=list(open_tickets),
            now=now,
            services=services,
        )
        records = self._run_nexcoder(ctx) if self.engine == "nexcoder" else self._run_local(ctx)
        status = "failed" if any(r.status == "failure" for r in records) else "completed"
        logger.info("pipeline %s: %s with engine %s, %d step(s)", self.name, status, self.engine, len(records))
        return PipelineResult(
            pipeline=self.name,
            engine=self.engine,
            status=status,
            steps=records,
            tickets=ctx.tickets,
            chunks=ctx.chunks or [],
            curation=ctx.curation_result,
            answers=ctx.answers,
            suggestions=ctx.suggestions,
            analytics=ctx.analytics,
            retriever=ctx.retriever,
        )


def run_pipeline(
    source: str | Path | Mapping[str, Any] | PipelineConfig, tickets: Iterable[Ticket], **kwargs: Any
) -> PipelineResult:
    """Convenience wrapper around ``Pipeline.from_config(source).run(tickets, **kwargs)``."""
    return Pipeline.from_config(source).run(tickets, **kwargs)
