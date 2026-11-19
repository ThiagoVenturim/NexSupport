"""Catalog of reusable pipeline steps.

A step is a plain function ``run(context, params) -> counters`` plus a description, a
pydantic model for its parameters and the context keys it reads (``requires``) and
writes (``provides``). The catalog checks a flow before it runs: every step must find
its inputs in the output of an earlier step.

Other packages can publish steps without touching this one, either by calling
``register_step`` on a catalog or through the ``support_engine.pipeline_steps`` entry
point group, whose entries point to a ``StepSpec`` or to a function returning a list of
them. Steps are never imported from a module path written in a flow file: a flow is
configuration, not code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from support_engine.pipelines.engine import PipelineContext

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "support_engine.pipeline_steps"

Counters = Mapping[str, int | float | str | bool | None]
StepFn = Callable[["PipelineContext", Any], Counters | None]


class NoParams(BaseModel):
    """Parameters of a step that takes none."""


@dataclass(frozen=True)
class StepSpec:
    """A reusable step: what it does, its parameters and its inputs and outputs."""

    name: str
    run: StepFn
    description: str
    category: str = "general"
    params: type[BaseModel] = NoParams
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    needs_masked: bool = False  # reads ticket text, so PII masking must run before it

    def describe(self) -> dict[str, Any]:
        """Catalog entry as plain data (name, description, inputs, outputs and the parameter JSON schema)."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "requires": list(self.requires),
            "provides": list(self.provides),
            "needs_masked": self.needs_masked,
            "params": self.params.model_json_schema(),
        }


class StepCatalog:
    """Named steps a flow can use."""

    def __init__(self, steps: Iterable[StepSpec] = ()) -> None:
        self._steps: dict[str, StepSpec] = {}
        for spec in steps:
            self.add(spec)

    def add(self, spec: StepSpec, *, replace: bool = False) -> StepSpec:
        if not spec.name or not spec.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid step name {spec.name!r}: use letters, digits, '_' or '-'")
        if spec.name in self._steps and not replace:
            raise ValueError(f"step {spec.name!r} is already in the catalog")
        self._steps[spec.name] = spec
        return spec

    def get(self, name: str) -> StepSpec:
        try:
            return self._steps[name]
        except KeyError:
            raise KeyError(f"unknown step {name!r}; available: {', '.join(sorted(self._steps))}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._steps

    def __iter__(self):
        return iter(self._steps.values())

    def __len__(self) -> int:
        return len(self._steps)

    def names(self) -> list[str]:
        return list(self._steps)

    def manifest(self, category: str | None = None) -> list[dict[str, Any]]:
        """Every step as plain data, for listings and for other projects that reuse the catalog."""
        return [s.describe() for s in self._steps.values() if category is None or s.category == category]

    def copy(self) -> StepCatalog:
        return StepCatalog(self._steps.values())

    def register(
        self,
        name: str,
        *,
        description: str,
        category: str = "general",
        params: type[BaseModel] = NoParams,
        requires: Iterable[str] = (),
        provides: Iterable[str] = (),
        needs_masked: bool = False,
        replace: bool = False,
    ) -> Callable[[StepFn], StepFn]:
        """Decorator form of ``add``."""

        def decorator(fn: StepFn) -> StepFn:
            self.add(
                StepSpec(
                    name=name,
                    run=fn,
                    description=description,
                    category=category,
                    params=params,
                    requires=tuple(requires),
                    provides=tuple(provides),
                    needs_masked=needs_masked,
                ),
                replace=replace,
            )
            return fn

        return decorator

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> int:
        """Add the steps published by installed packages. Returns how many were added."""
        added = 0
        for ep in entry_points(group=group):
            try:
                obj = ep.load()
                specs = obj() if callable(obj) and not isinstance(obj, StepSpec) else obj
                for spec in [specs] if isinstance(specs, StepSpec) else list(specs):
                    if spec.name in self._steps:
                        logger.warning("pipeline step from entry point %s skipped: name already used", ep.name)
                        continue
                    self.add(spec)
                    added += 1
            except Exception as exc:
                logger.warning("pipeline step entry point %s failed: %s", ep.name, type(exc).__name__)
        return added


_DEFAULT: StepCatalog | None = None


def default_catalog() -> StepCatalog:
    """The built-in steps plus those published through entry points (built once per process)."""
    global _DEFAULT
    if _DEFAULT is None:
        from support_engine.pipelines.steps import builtin_steps

        catalog = StepCatalog(builtin_steps())
        catalog.load_entry_points()
        _DEFAULT = catalog
    return _DEFAULT


def register_step(spec: StepSpec, *, replace: bool = False) -> StepSpec:
    """Add a step to the default catalog."""
    return default_catalog().add(spec, replace=replace)
