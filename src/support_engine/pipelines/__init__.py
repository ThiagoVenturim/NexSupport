"""Adjustable support flows: pluggable steps, order and parameters in YAML, with presets.

Typical use::

    from support_engine.pipelines import Pipeline

    flow = Pipeline.from_config("full")               # a preset, or a path to a YAML file
    result = flow.run(tickets, feedback=feedback, questions=["The invoice export fails"])
    result.answers       # support agent results (offline extractive answers by default)
    result.suggestions   # copilot suggestions for open tickets
    result.summary()     # per-step counters, no ticket text

The steps come from a catalog (``default_catalog()``) that other packages can extend with
``register_step`` or the ``support_engine.pipeline_steps`` entry point group.
"""

from support_engine.pipelines.catalog import (
    ENTRY_POINT_GROUP,
    NoParams,
    StepCatalog,
    StepSpec,
    default_catalog,
    register_step,
)
from support_engine.pipelines.config import (
    PipelineConfig,
    StepConfig,
    check_flow,
    dump_config,
    list_presets,
    load_config,
    validate_params,
)
from support_engine.pipelines.engine import (
    Pipeline,
    PipelineContext,
    PipelineResult,
    StepRecord,
    nexcoder_available,
    run_pipeline,
)
from support_engine.pipelines.steps import (
    AnswerParams,
    ChunkingParams,
    CopilotParams,
    DedupParams,
    ExtractiveAnswerGenerator,
    IndexParams,
    PriorityJudge,
    builtin_steps,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "AnswerParams",
    "ChunkingParams",
    "CopilotParams",
    "DedupParams",
    "ExtractiveAnswerGenerator",
    "IndexParams",
    "NoParams",
    "Pipeline",
    "PipelineConfig",
    "PipelineContext",
    "PipelineResult",
    "PriorityJudge",
    "StepCatalog",
    "StepConfig",
    "StepRecord",
    "StepSpec",
    "builtin_steps",
    "check_flow",
    "default_catalog",
    "dump_config",
    "list_presets",
    "load_config",
    "nexcoder_available",
    "register_step",
    "run_pipeline",
    "validate_params",
]
