"""Flow configuration: which steps run, in which order and with which parameters.

A flow lives in YAML (or JSON, or a dict)::

    name: my-flow
    extends: full            # optional: start from a preset
    steps:
      - use: dedup
        params: {threshold: 0.9}
      - use: analytics
        enabled: false

Without ``extends`` the ``steps`` list is the whole flow, in order. With ``extends`` each
entry patches the preset step of the same ``name`` (``params`` are merged key by key) and
entries with a new name are added at the end; ``order`` can list the step names again to
reorder them. ``name`` defaults to ``use``, so a step can appear twice under two names.

Parameters are checked against the step's own model before anything runs, and unknown
keys are errors so that a typo never goes unnoticed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from support_engine.pipelines.catalog import StepCatalog, default_catalog

STEP_NAME = r"^[A-Za-z0-9_-]{1,64}$"
Engine = Literal["auto", "nexcoder", "local"]


class StepConfig(BaseModel):
    """One step of a flow."""

    use: str = Field(pattern=STEP_NAME, description="catalog step to run")
    name: str | None = Field(default=None, pattern=STEP_NAME, description="unique name in the flow (default: use)")
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @property
    def id(self) -> str:
        return self.name or self.use


class PipelineConfig(BaseModel):
    """A whole flow."""

    name: str = Field(default="custom", pattern=STEP_NAME)
    description: str = ""
    extends: str | None = Field(default=None, pattern=STEP_NAME, description="preset this flow starts from")
    engine: Engine = Field(default="auto", description="nexcoder (LangGraph engine of nexcoder-agents) or local loop")
    allow_unmasked: bool = Field(default=False, description="required to run text steps without PII masking first")
    stop_on_error: bool = True
    order: list[str] | None = Field(default=None, description="step names in the order to run (with extends)")
    steps: list[StepConfig] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _unique_names(self) -> PipelineConfig:
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen and self.extends is None:
                raise ValueError(f"step name {step.id!r} is used twice; give one of them a 'name'")
            seen.add(step.id)
        return self

    def enabled_steps(self) -> list[StepConfig]:
        return [s for s in self.steps if s.enabled]


# Strict parameter check -----------------------------------------------------------------


def _unknown_keys(model: type[BaseModel], data: Any, prefix: str = "") -> list[str]:
    """Keys in ``data`` that ``model`` (and its nested models) does not define."""
    if not isinstance(data, Mapping):
        return []
    out: list[str] = []
    fields = model.model_fields
    for key, value in data.items():
        if key not in fields:
            out.append(f"{prefix}{key}")
            continue
        annotation = fields[key].annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            out += _unknown_keys(annotation, value, f"{prefix}{key}.")
    return out


def validate_params(model: type[BaseModel], params: Mapping[str, Any], step: str) -> BaseModel:
    """Build the parameter model of ``step``; unknown keys and invalid values raise ``ValueError``."""
    unknown = _unknown_keys(model, params)
    if unknown:
        raise ValueError(f"step {step!r}: unknown parameter(s) {', '.join(unknown)}")
    try:
        return model.model_validate(dict(params))
    except ValidationError as exc:
        raise ValueError(f"step {step!r}: invalid parameters: {_errors(exc)}") from None


# Presets and loading --------------------------------------------------------------------


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def list_presets() -> list[str]:
    """Names of the flows that ship with the engine."""
    folder = resources.files("support_engine.pipelines") / "presets"
    return sorted(p.name[: -len(".yaml")] for p in folder.iterdir() if p.name.endswith(".yaml"))


def preset_data(name: str) -> dict[str, Any]:
    if name not in list_presets():
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(list_presets())}")
    text = (resources.files("support_engine.pipelines") / "presets" / f"{name}.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def _resolve(data: Mapping[str, Any], depth: int = 0) -> PipelineConfig:
    data = dict(data)
    if data.get("extends") is None:
        return PipelineConfig.model_validate(data)
    head = PipelineConfig.model_validate({k: v for k, v in data.items() if k != "steps"})
    if depth >= 5:
        raise ValueError("preset chain is too deep (more than 5 'extends')")
    base = _resolve(preset_data(head.extends or ""), depth + 1)
    steps = {s.id: s.model_dump() for s in base.steps}
    order = [s.id for s in base.steps]
    patches = data.get("steps") or []
    if not isinstance(patches, list):
        raise ValueError("'steps' must be a list")
    for patch in patches:
        if not isinstance(patch, Mapping):
            raise ValueError("every entry of 'steps' must be a mapping")
        key = patch.get("name") or patch.get("use")
        if key in steps:
            # Patch of a preset step: 'use' may be left out, params are merged key by key.
            merged = _deep_merge(steps[key], {k: v for k, v in patch.items() if k != "name"})
            steps[key] = StepConfig.model_validate(merged).model_dump()
        else:
            step = StepConfig.model_validate(dict(patch))
            steps[step.id] = step.model_dump()
            order.append(step.id)
    if head.order is not None:
        missing = [n for n in head.order if n not in steps]
        if missing:
            raise ValueError(f"order names unknown step(s): {', '.join(missing)}")
        order = list(dict.fromkeys(head.order)) + [n for n in order if n not in head.order]
    fields = {k: v for k, v in data.items() if k not in ("steps", "order", "extends")}
    merged_cfg = {**base.model_dump(exclude={"steps", "order", "extends"}), **fields}
    if "name" not in fields:
        merged_cfg["name"] = base.name
    return PipelineConfig.model_validate({**merged_cfg, "steps": [steps[n] for n in order]})


def _errors(exc: ValidationError) -> str:
    # Field locations and messages only: pydantic's default text echoes input values.
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors())


def _load(source: str | Path | Mapping[str, Any] | PipelineConfig) -> PipelineConfig:
    if isinstance(source, PipelineConfig):
        return _resolve(source.model_dump(exclude_unset=True)) if source.extends else source
    if isinstance(source, Mapping):
        return _resolve(source)
    if isinstance(source, str) and source in list_presets() and not Path(source).is_file():
        return _resolve(preset_data(source))
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"flow file '{path}' not found and no preset is called {str(source)!r}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError):
        raise ValueError(f"flow file '{path}' is not valid YAML or JSON") from None
    if not isinstance(data, Mapping):
        raise ValueError(f"flow file '{path}' must hold a mapping at the top level")
    return _resolve(data)


def load_config(source: str | Path | Mapping[str, Any] | PipelineConfig) -> PipelineConfig:
    """Flow from a preset name, a YAML or JSON file, a dict or a ``PipelineConfig``; presets are resolved."""
    try:
        return _load(source)
    except ValidationError as exc:
        raise ValueError(f"invalid flow: {_errors(exc)}") from None


def dump_config(config: PipelineConfig) -> str:
    """YAML text of a resolved flow."""
    data = config.model_dump(exclude_none=True, exclude_defaults=False)
    data.pop("extends", None)
    data.pop("order", None)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def check_flow(config: PipelineConfig, catalog: StepCatalog | None = None) -> dict[str, BaseModel]:
    """Validate steps, parameters, the order of inputs and outputs and the masking rule.

    Returns the parameter model of every enabled step, by step name. Raises ``ValueError``
    (or ``KeyError`` for an unknown step) when the flow cannot run.
    """
    catalog = catalog or default_catalog()
    available = {"tickets"}
    masked = False
    params: dict[str, BaseModel] = {}
    enabled = config.enabled_steps()
    if not enabled:
        raise ValueError(f"flow {config.name!r} has no enabled step")
    for step in enabled:
        spec = catalog.get(step.use)
        values = validate_params(spec.params, step.params, step.id)
        missing = [key for key in spec.requires if key not in available]
        if missing:
            raise ValueError(
                f"step {step.id!r} needs {', '.join(missing)}, which no earlier step provides; "
                "check the order of the flow"
            )
        if spec.needs_masked and not masked and not config.allow_unmasked:
            raise ValueError(
                f"step {step.id!r} reads ticket text but no 'pii_mask' step runs before it; "
                "add one earlier or set allow_unmasked: true to confirm"
            )
        if spec.name == "pii_mask" and getattr(values, "mask_pii", False):
            masked = True
        available.update(spec.provides)
        params[step.id] = values
    return params
