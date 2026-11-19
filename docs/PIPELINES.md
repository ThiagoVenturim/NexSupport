# Adjustable support flows

`support_engine.pipelines` runs the support flow from configuration instead of code:

```
ingestion -> cleanup/PII -> chunking/dedup -> curation -> knowledge base (RAG) -> answer/copilot -> analytics
```

Each stage is a step from a catalog. A flow file (YAML or JSON) says which steps run,
in which order, with which parameters. Every step can be switched off. Two presets
ship with the package (`minimal` and `full`), and a flow can extend a preset and change
only what it needs.

```python
from support_engine.pipelines import Pipeline

flow = Pipeline.from_config("full")              # preset name or path to a flow file
result = flow.run(
    tickets,                                      # history to learn from
    feedback=feedback,                            # optional human ratings and corrections
    questions=["The invoice export fails"],       # for the answer step
    open_tickets=open_tickets,                    # for the copilot step
)
result.answers       # support agent results
result.suggestions   # copilot suggestions
result.analytics     # analytics report
result.chunks        # curated chunks of the knowledge base
result.summary()     # per-step status and counters, no ticket text
```

The run is synchronous; do not call it inside a running event loop.

## Steps

| Step | Category | Reads | Writes | Parameters |
| --- | --- | --- | --- | --- |
| `cleanup` | ingestion | tickets | tickets | `CleanupConfig` |
| `pii_mask` | ingestion | tickets | tickets | `PrivacyConfig` |
| `chunking` | ingestion | tickets | tickets, chunks | `ChunkingConfig` plus `drop_empty_comments`, `dedup_comments` |
| `dedup` | ingestion | chunks | chunks | `DedupConfig` plus `scope` (`ticket` or `corpus`) |
| `curation` | curation | tickets, chunks | chunks, scores | `CurationConfig` |
| `index` | rag | chunks | retriever | `k1`, `b` (BM25) |
| `answer` | response | retriever | answers | `agent` (`AgentConfig`), `generator`, `judge`, `max_sources`, `max_sentences`, `open_tickets`, `namespace` |
| `copilot` | response | retriever | suggestions | `copilot` (`CopilotConfig`), `generator`, `judge`, `max_sources`, `max_sentences`, `include_unresolved`, `namespace` |
| `analytics` | analytics | tickets | analytics | `AnalyticsConfig` |

The parameters are the configuration models of the modules the steps wrap, with the
same field names as the Python API (see [INGESTION.md](INGESTION.md),
[CURATION.md](CURATION.md), [AGENT.md](AGENT.md), [COPILOT.md](COPILOT.md) and
[ANALYTICS.md](ANALYTICS.md)). `support-engine pipeline steps` lists the catalog, and
`default_catalog().manifest()` returns it as data, with the JSON schema of every
parameter model.

### Offline answers

The `answer` and `copilot` steps use the support agent. By default they need no model:

- `generator: extractive` answers with sentences copied from the best answer-like
  sources (human corrections, the resolution line of case summaries, replies written by
  an agent), each one cited. Customer messages are never quoted as answers.
- `judge: priority` scores the answer with the curation priority of the sources it
  cites. After the `curation` step, proven answers score high and reach level 1; without
  curation every source sits at the neutral 0.5 and the answer goes to review (level 2).

`generator: llm` and `judge: llm` call `nexcoder_core.get_chat_model` with the tiers of
`AgentConfig`. Objects passed to `run` replace the defaults: `llm`, `judge_llm`,
`generator`, `judge`, `curation_judge`, `summarizer`, `copilot_summarizer` and
`name_detectors`.

## Flow files

```yaml
name: support-tier1
extends: full                  # start from a preset (optional)
engine: auto                   # auto, nexcoder or local
order: [pii_mask, cleanup]     # optional: move steps to the front, the rest keep their order
steps:
  - use: dedup
    params:
      threshold: 0.9           # merged into the preset parameters key by key
  - use: curation
    params:
      min_score: 0.3
      feedback:
        superseded_policy: drop
  - use: answer
    params:
      agent:
        levels:
          level1_min_score: 0.85
  - use: analytics
    enabled: false
```

Rules:

- Without `extends`, `steps` is the whole flow, in order.
- With `extends`, an entry patches the preset step with the same `name` (`name`
  defaults to `use`); entries with a new name are added at the end. Use `name` to run
  the same catalog step twice.
- Unknown keys, at the top level or in any parameter model, are errors.
- Every step must find its inputs in the output of an earlier step, so a flow with
  `index` before `chunking` is rejected before anything runs.
- Steps that read ticket text (`chunking`, `curation`, `analytics`) need an enabled
  `pii_mask` step before them. To run without masking, set `allow_unmasked: true` on the
  flow (and, to switch masking off inside `pii_mask`, `mask_pii: false` with
  `allow_unmasked: true` in its parameters).
- `stop_on_error: true` (default) skips the remaining steps after a failure; the result
  has `status: failed` and the step records the error type only.

`examples/pipelines/support-tier1.yaml` is a complete example.

## Presets

- `minimal`: `cleanup -> pii_mask -> chunking -> index -> answer`. The smallest safe
  flow; answers are drafts for review because nothing is curated.
- `full`: `cleanup -> pii_mask -> chunking -> dedup -> curation -> index -> answer ->
  copilot -> analytics`, with the same defaults as calling the modules directly.

## Engines

- `nexcoder`: every step becomes a block of the LangGraph pipeline engine of
  `nexcoder-agents` (`PipelineBuilder`), the engine the other projects of the suite use.
  Install it with the `pipelines` extra.
- `local`: a plain loop, for installs without `nexcoder-agents`.
- `auto` (default): `nexcoder` when the package is installed, `local` otherwise.

Both engines run the same step functions and give the same result. Objects (tickets,
chunks, the knowledge base) stay in the run context; the graph state carries step
counters only.

## Adding steps

A step is a function `run(context, params) -> counters` plus a `StepSpec`:

```python
from pydantic import BaseModel
from support_engine.pipelines import Pipeline, StepCatalog, StepSpec, builtin_steps

class TagParams(BaseModel):
    tag: str = "reviewed"

def add_tag(ctx, params: TagParams):
    ctx.tickets = [t.model_copy(update={"tags": [*t.tags, params.tag]}) for t in ctx.tickets]
    return {"tagged": len(ctx.tickets)}

catalog = StepCatalog(builtin_steps())
catalog.add(StepSpec(name="tag", run=add_tag, description="Tag every ticket.", params=TagParams,
                     requires=("tickets",), provides=("tickets",)))
flow = Pipeline.from_config({"extends": "minimal", "steps": [{"use": "tag"}]}, catalog=catalog)
```

`register_step(spec)` adds a step to the default catalog. Another package can publish
steps without any code here through the `support_engine.pipeline_steps` entry point
group; each entry points to a `StepSpec` or to a function that returns a list of them:

```toml
[project.entry-points."support_engine.pipeline_steps"]
my_steps = "my_package.steps:steps"
```

Flow files can only name steps that are in the catalog. They never import code from a
module path, so a flow file is configuration and cannot run arbitrary code.

## Command line

```bash
support-engine pipeline steps                       # the catalog
support-engine pipeline presets                     # ready-made flows
support-engine pipeline show -f examples/pipelines/support-tier1.yaml
support-engine pipeline run -f full -i tickets.json --open-tickets open.json \
    -q "The invoice export fails" --report-out run.json
```

`run` prints one line per step with its counters and exits with 0 when the flow
completes, 1 or 2 for missing files or an invalid flow, and 3 when a step failed.

## Privacy

- Masking runs by default in both presets, and a flow that reads ticket text without it
  is rejected unless it opts out explicitly.
- Logs and step records carry step names, counters and error types; they never carry
  ticket text, questions, answers or exception messages.
- Error messages about a flow name the steps and parameters, not the values.
