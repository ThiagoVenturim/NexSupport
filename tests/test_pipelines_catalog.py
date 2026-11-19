"""Step catalog: built-in steps, custom steps, entry points and the offline answer helpers."""

from __future__ import annotations

import asyncio

import pytest
from agent_data import tickets
from pydantic import BaseModel

from support_engine.agent import Source
from support_engine.pipelines import (
    ExtractiveAnswerGenerator,
    Pipeline,
    PriorityJudge,
    StepCatalog,
    StepSpec,
    builtin_steps,
    catalog,
    default_catalog,
)

FLOW = ["cleanup", "pii_mask", "chunking", "dedup", "curation", "index", "answer", "copilot", "analytics"]


def test_builtin_catalog_covers_the_support_flow():
    assert default_catalog().names()[: len(FLOW)] == FLOW
    manifest = {entry["name"]: entry for entry in default_catalog().manifest()}
    assert manifest["chunking"]["provides"] == ["tickets", "chunks"]
    assert manifest["chunking"]["needs_masked"] is True
    assert "max_chars" in manifest["chunking"]["params"]["properties"]
    assert "threshold" in manifest["dedup"]["params"]["properties"]
    assert {e["name"] for e in default_catalog().manifest("ingestion")} == {"cleanup", "pii_mask", "chunking", "dedup"}


class TagParams(BaseModel):
    tag: str = "reviewed"


def add_tag(ctx, params: TagParams):
    ctx.tickets = [t.model_copy(update={"tags": [*t.tags, params.tag]}) for t in ctx.tickets]
    return {"tagged": len(ctx.tickets)}


def test_custom_step_in_a_private_catalog():
    custom = StepCatalog(builtin_steps())
    custom.add(
        StepSpec(
            name="tag",
            run=add_tag,
            description="Add a tag to every ticket.",
            params=TagParams,
            requires=("tickets",),
            provides=("tickets",),
        )
    )
    flow = Pipeline.from_config(
        {"engine": "local", "steps": [{"use": "tag", "params": {"tag": "vip"}}, {"use": "pii_mask"}]}, catalog=custom
    )
    result = flow.run(tickets())
    assert all("vip" in t.tags for t in result.tickets)
    assert result.step("tag").counters == {"tagged": 3}
    assert "tag" not in default_catalog()


def test_decorator_registration_and_name_rules():
    custom = StepCatalog()

    @custom.register("noop", description="Does nothing.")
    def noop(ctx, params):
        return None

    assert custom.get("noop").run is noop
    with pytest.raises(ValueError, match="already"):
        custom.add(custom.get("noop"))
    custom.add(custom.get("noop"), replace=True)
    with pytest.raises(ValueError, match="invalid step name"):
        custom.add(StepSpec(name="bad name", run=noop, description="x"))
    with pytest.raises(KeyError, match="unknown step"):
        custom.get("missing")


def test_entry_points_publish_steps(monkeypatch):
    spec = StepSpec(name="external", run=add_tag, description="From another package.", params=TagParams)

    class FakeEntryPoint:
        def __init__(self, name, obj):
            self.name, self._obj = name, obj

        def load(self):
            if isinstance(self._obj, Exception):
                raise self._obj
            return self._obj

    points = [
        FakeEntryPoint("one", spec),
        FakeEntryPoint("many", lambda: [StepSpec(name="external2", run=add_tag, description="x")]),
        FakeEntryPoint("clash", StepSpec(name="cleanup", run=add_tag, description="x")),
        FakeEntryPoint("broken", ImportError("missing dependency")),
    ]
    monkeypatch.setattr(catalog, "entry_points", lambda group: points if group == catalog.ENTRY_POINT_GROUP else [])
    custom = StepCatalog(builtin_steps())
    assert custom.load_entry_points() == 2
    assert "external" in custom and "external2" in custom
    assert custom.get("cleanup").run is not add_tag


def _source(label, kind, text, chunk_id, priority=0.5):
    return Source(label=label, chunk_id=chunk_id, ticket_id="T-1", kind=kind, text=text, priority=priority)


def test_extractive_generator_quotes_answers_not_customers():
    sources = [
        _source("S1", "interaction", "The export fails with error 500.", "T-1:i1:0"),
        _source("S2", "summary", "Subject: Export\nStatus: solved\nProblem: fails\nResolution: Clear the cache.", "s"),
        _source("S3", "interaction", "Clear the cache. Then retry the export.", "T-1:i2:0"),
        _source("S4", "correction", "Use the new export page.", "c"),
    ]
    roles = {"T-1:i1:0": "customer", "T-1:i2:0": "agent"}
    text = asyncio.run(ExtractiveAnswerGenerator(roles, max_sources=3).agenerate("q", sources))
    assert "error 500" not in text
    # Repeated sentences are quoted once; each sentence cites its source.
    assert text == "Clear the cache [S2]. Then retry the export [S3]. Use the new export page [S4]."
    assert asyncio.run(ExtractiveAnswerGenerator(roles).agenerate("q", sources[:1])) == ""


def test_priority_judge_uses_curation_priority_of_cited_sources():
    sources = [_source("S1", "summary", "x", "a", priority=0.9), _source("S2", "summary", "y", "b", priority=0.3)]
    judge = PriorityJudge()
    assert asyncio.run(judge.ajudge("q", "Do it [S2].", sources)).score == 0.3
    assert asyncio.run(judge.ajudge("q", "Do it [S1, S2].", sources)).score == 0.9
    verdict = asyncio.run(judge.ajudge("q", "No citation.", sources))
    assert verdict.score == 0.0 and not verdict.grounded
