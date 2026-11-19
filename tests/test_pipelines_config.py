"""Flow files: YAML loading, presets, patches, strict parameters and the checks before a run."""

from __future__ import annotations

import pytest
import yaml
from agent_data import tickets

from support_engine.pipelines import (
    Pipeline,
    PipelineConfig,
    check_flow,
    dump_config,
    list_presets,
    load_config,
)


def test_presets_ship_with_the_package():
    assert {"minimal", "full"} <= set(list_presets())
    for name in list_presets():
        config = load_config(name)
        assert config.name == name
        check_flow(config)


def test_yaml_file_defines_the_whole_flow(tmp_path):
    path = tmp_path / "flow.yaml"
    path.write_text(
        """
name: kb-only
engine: local
steps:
  - use: pii_mask
  - use: cleanup
    params:
      strip_signatures: false
  - use: chunking
    params:
      max_chars: 300
      overlap_chars: 30
      include_summary: false
  - use: index
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert [s.use for s in config.steps] == ["pii_mask", "cleanup", "chunking", "index"]
    result = Pipeline.from_config(path).run(tickets())
    assert result.status == "completed"
    assert result.chunks and all(c.kind == "interaction" and len(c.text) <= 300 for c in result.chunks)
    assert result.retriever is not None and len(result.retriever) == len(result.chunks)


def test_json_file_is_accepted(tmp_path):
    path = tmp_path / "flow.json"
    path.write_text('{"extends": "minimal", "name": "from-json"}', encoding="utf-8")
    assert load_config(path).name == "from-json"


def test_extends_merges_params_key_by_key():
    config = load_config(
        {
            "extends": "full",
            "name": "strict",
            "steps": [
                {"use": "curation", "params": {"min_score": 0.4}},
                {"use": "curation", "params": {"feedback": {"superseded_policy": "drop"}}},
                {"use": "dedup", "params": {"threshold": 0.9}},
            ],
        }
    )
    assert config.name == "strict"
    curation = next(s for s in config.steps if s.id == "curation")
    assert curation.params == {"min_score": 0.4, "feedback": {"superseded_policy": "drop"}}
    dedup = next(s for s in config.steps if s.id == "dedup")
    assert dedup.params == {"scope": "corpus", "threshold": 0.9}
    params = check_flow(config)
    assert params["curation"].feedback.superseded_policy == "drop"


def test_new_step_names_are_appended_and_can_repeat_a_step():
    config = load_config({"extends": "minimal", "steps": [{"use": "cleanup", "name": "cleanup_again"}]})
    assert [s.id for s in config.steps][-1] == "cleanup_again"
    check_flow(config)


def test_unknown_parameter_is_an_error():
    with pytest.raises(ValueError, match="unknown parameter.*max_char"):
        Pipeline.from_config({"extends": "minimal", "steps": [{"use": "chunking", "params": {"max_char": 10}}]})
    with pytest.raises(ValueError, match="agent.retrieval.kk"):
        Pipeline.from_config(
            {"extends": "minimal", "steps": [{"use": "answer", "params": {"agent": {"retrieval": {"kk": 1}}}}]}
        )


def test_invalid_parameter_value_names_the_field_only():
    with pytest.raises(ValueError, match="threshold") as info:
        Pipeline.from_config({"extends": "full", "steps": [{"use": "dedup", "params": {"threshold": 7.5}}]})
    assert "7.5" not in str(info.value)


def test_unknown_step_and_unknown_top_level_key():
    with pytest.raises(KeyError, match="unknown step 'translate'"):
        Pipeline.from_config({"steps": [{"use": "translate"}]})
    with pytest.raises(ValueError, match="stepz"):
        load_config({"stepz": []})


def test_order_must_give_each_step_its_inputs():
    with pytest.raises(ValueError, match="'index' needs chunks"):
        Pipeline.from_config({"steps": [{"use": "pii_mask"}, {"use": "index"}, {"use": "chunking"}]})
    with pytest.raises(ValueError, match="'answer' needs retriever"):
        Pipeline.from_config({"extends": "minimal", "steps": [{"use": "index", "enabled": False}]})


def test_text_steps_need_masking_first():
    with pytest.raises(ValueError, match="no 'pii_mask' step runs before it"):
        Pipeline.from_config({"steps": [{"use": "cleanup"}, {"use": "chunking"}]})
    with pytest.raises(ValueError, match="pii_mask"):
        Pipeline.from_config({"extends": "minimal", "order": ["chunking", "pii_mask"]})
    # Turning masking off needs the explicit opt-out of the privacy settings as well.
    with pytest.raises(ValueError, match="allow_unmasked"):
        Pipeline.from_config({"extends": "minimal", "steps": [{"use": "pii_mask", "params": {"mask_pii": False}}]})
    flow = Pipeline.from_config(
        {"allow_unmasked": True, "engine": "local", "steps": [{"use": "chunking"}, {"use": "index"}]}
    )
    assert flow.run(tickets()).status == "completed"


def test_duplicate_step_names_and_empty_flows_are_rejected():
    with pytest.raises(ValueError, match="used twice"):
        load_config({"steps": [{"use": "cleanup"}, {"use": "cleanup"}]})
    with pytest.raises(ValueError, match="no enabled step"):
        Pipeline.from_config(
            {
                "extends": "minimal",
                "steps": [{"use": n, "enabled": False} for n in ("cleanup", "pii_mask", "chunking", "index", "answer")],
            }
        )


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("steps: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_config(bad)
    listing = tmp_path / "list.yaml"
    listing.write_text("- use: cleanup\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(listing)


def test_dump_round_trip():
    config = load_config({"extends": "full", "steps": [{"use": "analytics", "enabled": False}]})
    again = load_config(yaml.safe_load(dump_config(config)))
    assert again.model_dump() == config.model_dump()


def test_nexcoder_engine_requires_the_package(monkeypatch):
    from support_engine.pipelines import engine

    monkeypatch.setattr(engine, "nexcoder_available", lambda: False)
    with pytest.raises(ValueError, match="nexcoder-agents"):
        engine.Pipeline(load_config({"extends": "minimal", "engine": "nexcoder"}))
    assert engine.Pipeline(load_config("minimal")).engine == "local"


def test_pipeline_config_object_is_accepted():
    config = PipelineConfig(extends="minimal", engine="local")
    flow = Pipeline.from_config(config)
    assert flow.name == "minimal" and flow.engine == "local"


def test_example_flow_file_is_valid():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "pipelines" / "support-tier1.yaml"
    flow = Pipeline.from_config(path)
    assert flow.name == "support-tier1"
    assert "analytics" not in [s.id for s in flow.config.enabled_steps()]
    assert flow.params["curation"].feedback.superseded_policy == "drop"
    assert flow.params["answer"].agent.levels.level1_min_score == 0.85
    result = flow.run(tickets(), questions=["The invoice export fails with error 500"])
    assert result.status == "completed" and result.answers[0].action == "answer"
