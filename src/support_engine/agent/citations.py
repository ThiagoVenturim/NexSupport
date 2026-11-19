"""Citation check: every claim in an answer must point at a retrieved document that supports it.

The model cites documents with their labels in brackets, for example ``[S1]`` or ``[S1, S3]``.
The check verifies that

1. the answer cites at least one document (when ``CitationConfig.required``);
2. every cited label belongs to a document that was actually retrieved;
3. each cited sentence is supported: at least ``min_support`` of its content words appear in
   the documents it cites;
4. no more than ``max_uncited_share`` of the content sentences go without a citation.

``CitationReport.explanation()`` turns the problems into a short instruction for the retry.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from support_engine.agent.config import CitationConfig
from support_engine.agent.retrieval import Source, tokens

_CITATION = re.compile(r"\[\s*((?:S\d+)(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class Citation(BaseModel):
    """A document the final answer cites."""

    label: str
    chunk_id: str
    ticket_id: str
    kind: str


class SentenceCheck(BaseModel):
    index: int
    labels: list[str] = Field(default_factory=list)
    support: float | None = Field(default=None, description="share of content words found in the cited documents")


class CitationReport(BaseModel):
    """Result of checking an answer against its sources. Holds labels and numbers, no text."""

    ok: bool = False
    cited: list[str] = Field(default_factory=list, description="valid labels, in order of first use")
    unknown: list[str] = Field(default_factory=list, description="labels that match no retrieved document")
    unsupported: list[int] = Field(default_factory=list, description="indexes of cited sentences not supported")
    uncited: list[int] = Field(default_factory=list, description="indexes of content sentences without citation")
    sentences: int = 0
    support: float = Field(default=0.0, description="mean support of the cited sentences")
    problems: list[str] = Field(default_factory=list)

    def explanation(self) -> str:
        """Short instruction for the next attempt."""
        return " ".join(self.problems)


def labels_in(text: str) -> list[str]:
    """Citation labels in order of appearance, upper-cased and without repeats."""
    seen: dict[str, None] = {}
    for match in _CITATION.finditer(text):
        for label in re.split(r"\s*[,;]\s*", match.group(1)):
            seen.setdefault(label.strip().upper(), None)
    return list(seen)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]


def check_citations(answer: str, sources: Sequence[Source], config: CitationConfig | None = None) -> CitationReport:
    """Check that ``answer`` cites retrieved ``sources`` and that the cited sources support it."""
    cfg = config or CitationConfig()
    by_label = {s.label.upper(): s for s in sources}
    source_tokens = {label: set(tokens(s.text)) for label, s in by_label.items()}
    report = CitationReport()

    supports: list[float] = []
    checked = 0
    for idx, sentence in enumerate(split_sentences(answer)):
        content = set(tokens(_CITATION.sub(" ", sentence)))
        labels = labels_in(sentence)
        for label in labels:
            if label in by_label:
                if label not in report.cited:
                    report.cited.append(label)
            elif label not in report.unknown:
                report.unknown.append(label)
        if len(content) < cfg.min_sentence_tokens:
            continue
        checked += 1
        valid = [label for label in labels if label in by_label]
        if not valid:
            report.uncited.append(idx)
            continue
        evidence: set[str] = set().union(*(source_tokens[label] for label in valid))
        support = len(content & evidence) / len(content)
        supports.append(support)
        if support < cfg.min_support:
            report.unsupported.append(idx)

    report.sentences = checked
    report.support = round(sum(supports) / len(supports), 6) if supports else 0.0

    if report.unknown:
        report.problems.append(
            f"You cited {', '.join(report.unknown)}, which are not among the documents; cite only the given labels."
        )
    if cfg.required and not report.cited:
        report.problems.append("Cite the document that supports each statement with its label, for example [S1].")
    if report.unsupported:
        report.problems.append(
            f"{len(report.unsupported)} cited sentence(s) are not backed by the cited document; "
            "state only what the documents say."
        )
    if checked and len(report.uncited) / checked > cfg.max_uncited_share:
        report.problems.append(f"{len(report.uncited)} sentence(s) have no citation; add one or remove the claim.")
    report.ok = not report.problems
    return report


def citations_for(report: CitationReport, sources: Sequence[Source]) -> list[Citation]:
    by_label = {s.label.upper(): s for s in sources}
    return [
        Citation(label=label, chunk_id=by_label[label].chunk_id, ticket_id=by_label[label].ticket_id,
                 kind=by_label[label].kind)
        for label in report.cited
        if label in by_label
    ]
