"""Two flows, minimal and full, run offline on synthetic tickets and give the expected results."""

from __future__ import annotations

import logging

import pytest
from agent_data import T0, tickets
from copilot_data import EMAIL, open_ticket

from support_engine.curation import CurationPipeline, Feedback
from support_engine.ingestion import IngestionPipeline
from support_engine.models import Comment, Ticket, TicketStatus
from support_engine.pipelines import Pipeline, nexcoder_available

QUESTION = "The invoice export fails with error 500"
ENGINES = ["local", pytest.param("nexcoder", marks=pytest.mark.skipif(not nexcoder_available(), reason="no engine"))]


def corpus() -> list[Ticket]:
    """The agent tickets plus one with personal data and a repeated interaction."""
    leaky = Ticket(
        id="T-4",
        source="csv",
        subject="Invoice export error 500 again",
        status=TicketStatus.SOLVED,
        csat=1.0,
        created_at=T0,
        extra={"requester_name": "Joana Exemplo"},
        comments=[
            Comment(
                id="T-4-c1",
                author_role="customer",
                body=f"Hi, Joana Exemplo here ({EMAIL}). The invoice export fails with error 500.",
                created_at=T0,
            ),
            Comment(
                id="T-4-c2",
                author_role="agent",
                body="Clear the export cache in Settings, then run the invoice export again.",
                created_at=T0,
            ),
            Comment(
                id="T-4-c3",
                author_role="agent",
                body="Clear the export cache in Settings, then run the invoice export again.",
                created_at=T0,
            ),
        ],
    )
    return [*tickets(), leaky]


@pytest.mark.parametrize("engine", ENGINES)
def test_minimal_flow_drafts_answers_for_review(engine):
    flow = Pipeline.from_config({"extends": "minimal", "engine": engine})
    result = flow.run(corpus(), questions=[QUESTION])

    assert result.status == "completed"
    assert result.engine == engine
    assert [s.name for s in result.steps] == ["cleanup", "pii_mask", "chunking", "index", "answer"]
    assert all(s.status == "success" for s in result.steps)
    # No curation: chunks carry no score, sources sit at the neutral priority, the answer goes to review.
    assert result.curation is None
    assert all("resolution_score" not in c.metadata for c in result.chunks)
    [answer] = result.answers
    assert answer.action == "review" and answer.level == 2
    assert "Clear the export cache" in answer.answer
    assert answer.citations and answer.citations[0].ticket_id in {"T-1", "T-4"}
    assert result.suggestions == [] and result.analytics is None
    assert result.step("answer").counters == {"questions": 1, "review": 1}


@pytest.mark.parametrize("engine", ENGINES)
def test_full_flow_answers_suggests_and_analyses(engine):
    feedback = [Feedback.vote("T-2", helpful=False, correction="Raise the clock drift tolerance to five minutes.")]
    flow = Pipeline.from_config({"extends": "full", "engine": engine})
    result = flow.run(corpus(), feedback=feedback, questions=[QUESTION], open_tickets=[open_ticket()])

    assert result.status == "completed"
    assert [s.name for s in result.steps] == [
        "cleanup",
        "pii_mask",
        "chunking",
        "dedup",
        "curation",
        "index",
        "answer",
        "copilot",
        "analytics",
    ]
    # Curated: proven answers rank high, the agent answers on its own (level 1) with a citation.
    [answer] = result.answers
    assert answer.action == "answer" and answer.level == 1
    assert answer.judge is not None and answer.judge.score >= 0.8
    assert {c.ticket_id for c in answer.citations} <= {"T-1", "T-4"}
    # The human correction became a chunk of the knowledge base.
    assert any(c.kind == "correction" and c.ticket_id == "T-2" for c in result.chunks)
    assert result.step("curation").counters["corrections"] == 1
    # The repeated agent message of T-4 was dropped before chunking.
    assert result.step("chunking").counters["comments_dropped"] == 1
    # Copilot: grounded suggestion for the open ticket, empathetic tone for an upset customer.
    [suggestion] = result.suggestions
    assert suggestion.ticket_id == "N-1"
    assert suggestion.reply and "Clear the export cache" in suggestion.reply
    assert suggestion.tone.tone == "empathetic"
    assert result.analytics is not None and result.analytics.tickets == 4
    # ROI is computed from the answers of the run.
    assert result.analytics.roi is not None and result.analytics.roi.questions == 1


def test_engines_give_the_same_result():
    if not nexcoder_available():
        pytest.skip("nexcoder-agents is not installed")
    outputs = []
    for engine in ("local", "nexcoder"):
        result = Pipeline.from_config({"extends": "full", "engine": engine}).run(corpus(), questions=[QUESTION])
        outputs.append(
            (
                [(s.name, s.status, s.counters) for s in result.steps],
                [c.id for c in result.chunks],
                [(a.action, a.answer) for a in result.answers],
            )
        )
    assert outputs[0] == outputs[1]


def test_full_flow_matches_the_python_api():
    """The flow reuses ingestion and curation: same chunks, same scores as calling them directly."""
    ingested = IngestionPipeline().process(corpus())
    curated = CurationPipeline().curate(ingested.tickets, ingested.chunks)
    flow = Pipeline.from_config({"extends": "full", "engine": "local", "steps": [{"use": "copilot", "enabled": False}]})
    result = flow.run(corpus())

    assert [c.id for c in result.chunks] == [c.id for c in curated.chunks]
    assert [c.metadata.get("curation_priority") for c in result.chunks] == [
        c.metadata.get("curation_priority") for c in curated.chunks
    ]
    assert [t.model_dump() for t in result.tickets] == [t.model_dump() for t in ingested.tickets]


def test_pii_is_masked_in_outputs_and_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    result = Pipeline.from_config({"extends": "full", "engine": "local"}).run(
        corpus(), questions=[f"I am {EMAIL}, the invoice export fails"], open_tickets=[open_ticket()]
    )
    dumped = result.model_dump_json()
    assert EMAIL not in dumped and "Joana" not in dumped
    assert "[EMAIL_" in dumped
    for record in caplog.records:
        text = record.getMessage()
        assert EMAIL not in text and "Joana" not in text and "invoice" not in text.lower()


def test_disabled_and_reordered_steps():
    flow = Pipeline.from_config(
        {
            "extends": "full",
            "engine": "local",
            "order": ["pii_mask", "cleanup"],
            "steps": [
                {"use": "dedup", "params": {"threshold": 0.95}},
                {"use": "copilot", "enabled": False},
                {"use": "analytics", "enabled": False},
            ],
        }
    )
    assert [s.id for s in flow.config.enabled_steps()][:3] == ["pii_mask", "cleanup", "chunking"]
    assert flow.params["dedup"].threshold == 0.95
    result = flow.run(corpus(), questions=[QUESTION])
    assert [s.name for s in result.steps][-2:] == ["index", "answer"]
    assert result.suggestions == [] and result.analytics is None


def test_failed_step_stops_the_flow_and_reports_the_type_only(monkeypatch):
    from support_engine.pipelines import steps

    secret = f"boom {EMAIL}"

    def explode(ctx, params):
        raise RuntimeError(secret)

    monkeypatch.setattr(steps.IngestionPipeline, "_dedup_comments", lambda self, t, s: explode(None, None))
    result = Pipeline.from_config({"extends": "minimal", "engine": "local"}).run(corpus(), questions=[QUESTION])
    assert result.status == "failed"
    assert result.step("chunking").status == "failure"
    assert result.step("chunking").error == "RuntimeError"
    assert [result.step(n).status for n in ("index", "answer")] == ["skipped", "skipped"]
    assert secret not in result.model_dump_json()
    assert result.errors == ["chunking: RuntimeError"]


@pytest.mark.skipif(not nexcoder_available(), reason="nexcoder-agents is not installed")
def test_failed_step_on_the_nexcoder_engine(monkeypatch):
    from support_engine.pipelines import steps

    monkeypatch.setattr(steps.IngestionPipeline, "_dedup_comments", lambda self, t, s: 1 / 0)
    result = Pipeline.from_config({"extends": "minimal", "engine": "nexcoder"}).run(corpus())
    assert result.status == "failed"
    assert result.step("chunking").error == "ZeroDivisionError"
    assert result.step("answer").status == "skipped"


def test_injected_generator_replaces_the_offline_default():
    from agent_data import RecordingGenerator

    generator = RecordingGenerator(["Clear the export cache in Settings, then run the invoice export again [S1]."])
    result = Pipeline.from_config({"extends": "minimal", "engine": "local"}).run(
        corpus(), questions=[QUESTION], generator=generator
    )
    assert generator.calls and result.answers[0].answer.startswith("Clear the export cache")
