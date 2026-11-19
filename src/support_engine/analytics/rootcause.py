"""Root cause: what a group of problem tickets has in common that the other tickets do not.

Every ticket becomes a set of features: content words and word pairs of its subject,
description and customer messages, its tags, product, priority, source, language, chosen
``Ticket.extra`` keys (environment, version, plan, ...) and change mentions such as
"after the update", "since version 2.3", "depois da atualizacao" or "desde la
actualizacion". For each feature shared by at least ``min_support`` target tickets:

``lift = target_share / baseline_share``, with ``share = (count + s) / (n + 2s)``
(additive smoothing ``s``), and ``score = lift * log(1 + support)``.

Features with ``lift >= min_lift`` are returned, best score first, as root cause
candidates. Words and word pairs come from masked text with placeholders removed, so a
feature never carries personal data.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, Field

from support_engine.analytics.config import RootCauseConfig
from support_engine.analytics.text import content_tokens, customer_messages, fold, ticket_field, topic_text
from support_engine.models import Ticket

FeatureKind = Literal["term", "phrase", "field", "change"]

_VERSION = re.compile(r"\b(?:v|version|versao|release|build)\s*(\d+(?:\.\d+){1,3})\b")
CHANGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "change:update",
        re.compile(
            r"\b(?:after|since)\s+(?:the\s+|last\s+|latest\s+|today'?s\s+)?(?:update|upgrade|release|deploy(?:ment)?|"
            r"patch|migration|change)\b"
        ),
    ),
    (
        "change:update",
        re.compile(
            r"\b(?:depois|desde|apos)\s+(?:d[ao]s?\s+|a\s+|o\s+|ultima\s+|ultimo\s+)*(?:atualizacao|update|"
            r"versao|mudanca|migracao|deploy)\b"
        ),
    ),
    (
        "change:update",
        re.compile(
            r"\b(?:despues|desde)\s+(?:de\s+)?(?:la\s+|el\s+|ultima\s+|ultimo\s+)*(?:actualizacion|update|"
            r"version|cambio|migracion)\b"
        ),
    ),
)


class RootCause(BaseModel):
    """A feature over-represented in the target tickets."""

    feature: str
    kind: FeatureKind
    support: int = Field(description="target tickets with the feature")
    target_share: float
    baseline_share: float
    lift: float
    score: float
    ticket_ids: list[str] = Field(default_factory=list)


class RootCauseReport(BaseModel):
    """Root cause candidates of one target group. Feature names and ids only."""

    name: str
    target: int
    baseline: int
    causes: list[RootCause] = Field(default_factory=list)


def _kind(feature: str) -> FeatureKind:
    if feature.startswith("change:") or feature.startswith("version:"):
        return "change"
    if ":" in feature:
        return "field"
    return "phrase" if " " in feature else "term"


class RootCauseAnalyzer:
    """Ranks the features that separate a target group of tickets from a baseline."""

    def __init__(self, config: RootCauseConfig | None = None) -> None:
        self.config = config or RootCauseConfig()

    def features(self, ticket: Ticket) -> set[str]:
        cfg = self.config
        text = "\n".join([topic_text(ticket), *customer_messages(ticket)])
        out: set[str] = set()
        tokens = content_tokens(text, cfg.min_token_chars)
        if cfg.use_terms:
            out |= set(tokens)
        if cfg.use_bigrams:
            out |= {f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False) if a != b}
        for name in cfg.fields:
            label = "tag" if name in ("tag", "tags") else name
            out |= {f"{label}:{v}" for v in ticket_field(ticket, name)}
        for key in cfg.extra_keys:
            out |= {f"{key}:{v}" for v in ticket_field(ticket, key)}
        if cfg.detect_changes:
            out |= change_features(text)
        return out

    def analyze(
        self, target: Iterable[Ticket], baseline: Iterable[Ticket], *, name: str = "target"
    ) -> RootCauseReport:
        """Compare ``target`` with ``baseline``. Tickets present in both count only as target."""
        cfg = self.config
        target_rows = list(target)
        target_ids = {t.id for t in target_rows}
        baseline_rows = [t for t in baseline if t.id not in target_ids]
        report = RootCauseReport(name=name, target=len(target_rows), baseline=len(baseline_rows))
        if not target_rows:
            return report
        target_sets = {t.id: self.features(t) for t in target_rows}
        target_counts = Counter(f for fs in target_sets.values() for f in fs)
        baseline_counts = Counter(f for t in baseline_rows for f in self.features(t))
        s, nt, nb = cfg.smoothing, len(target_rows), len(baseline_rows)
        causes = []
        for feature, support in target_counts.items():
            if support < cfg.min_support:
                continue
            t_share = (support + s) / (nt + 2 * s) if nt + 2 * s else 0.0
            b_share = (baseline_counts.get(feature, 0) + s) / (nb + 2 * s) if nb + 2 * s else 0.0
            lift = t_share / b_share if b_share else float(support)
            if lift < cfg.min_lift:
                continue
            ids = [tid for tid, fs in target_sets.items() if feature in fs][: cfg.max_ticket_ids]
            causes.append(
                RootCause(
                    feature=feature,
                    kind=_kind(feature),
                    support=support,
                    target_share=round(support / nt, 4),
                    baseline_share=round(baseline_counts.get(feature, 0) / nb, 4) if nb else 0.0,
                    lift=round(lift, 4),
                    score=round(lift * math.log1p(support), 4),
                    ticket_ids=ids,
                )
            )
        causes = _drop_redundant(causes)
        causes.sort(key=lambda c: (-c.score, -c.support, c.feature))
        report.causes = causes[: cfg.max_causes]
        return report

    def analyze_where(
        self, tickets: Iterable[Ticket], predicate: Callable[[Ticket], bool], *, name: str = "target"
    ) -> RootCauseReport:
        """Target = tickets matching ``predicate``; baseline = the rest."""
        rows = list(tickets)
        target = [t for t in rows if predicate(t)]
        return self.analyze(target, rows, name=name)


def change_features(text: str) -> set[str]:
    """``change:update`` for change mentions and ``version:<n>`` for version numbers, from folded text."""
    folded = fold(text)
    out = {name for name, pattern in CHANGE_PATTERNS if pattern.search(folded)}
    out |= {f"version:{m}" for m in _VERSION.findall(folded)}
    return out


def _drop_redundant(causes: list[RootCause]) -> list[RootCause]:
    """Drop a single word when a word pair containing it covers the same tickets with at least its lift."""
    pairs = [c for c in causes if c.kind == "phrase"]
    out = []
    for cause in causes:
        if cause.kind == "term" and any(
            cause.feature in p.feature.split() and p.support == cause.support and p.lift >= cause.lift for p in pairs
        ):
            continue
        out.append(cause)
    return out
