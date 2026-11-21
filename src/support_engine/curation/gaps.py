"""Knowledge gaps: questions without a good answer, grouped so repeats ask for a new article.

A question is a gap when the best answer found for it scores below ``gap_threshold``
(or nothing was found), or when a person rated the shown answer as not helpful and
wrote no correction. Similar questions join one cluster by token Jaccard; a cluster
with ``min_occurrences`` questions becomes an ``ArticleSuggestion``. Questions are
masked with the default PII rules before they are kept.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, Field

from support_engine.curation.config import GapConfig
from support_engine.curation.feedback import Feedback
from support_engine.ingestion.dedup import jaccard
from support_engine.pii import mask_pii

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")

STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "how", "what", "why", "when", "can", "does", "not", "are", "was", "you",
        "your", "our", "this", "that", "have", "from", "into", "after", "before", "get", "cannot", "can't",
        "como", "para", "com", "que", "nao", "uma", "por", "meu", "minha", "esta", "isso", "ser", "sem",
        "depois", "antes", "quando", "porque", "qual", "onde", "fazer", "consigo",
    }
)


class GapCluster(BaseModel):
    """Similar unanswered questions."""

    id: str
    tokens: list[str] = Field(default_factory=list, description="token set of the first question")
    count: int = 0
    samples: list[str] = Field(default_factory=list, description="masked questions")
    ticket_ids: list[str] = Field(default_factory=list)
    term_counts: dict[str, int] = Field(default_factory=dict)


class ArticleSuggestion(BaseModel):
    """A repeated gap that should get a knowledge base article."""

    cluster_id: str
    title: str
    keywords: list[str]
    occurrences: int
    sample_queries: list[str] = Field(default_factory=list)
    ticket_ids: list[str] = Field(default_factory=list)


def query_tokens(text: str, min_chars: int = 3) -> set[str]:
    """Lowercase content words, without masked placeholders and stopwords."""
    text = _PLACEHOLDER.sub(" ", text)
    return {t for t in (m.group(0).lower() for m in _TOKEN.finditer(text)) if len(t) >= min_chars} - STOPWORDS


class GapDetector:
    """Collects gaps and turns repeated ones into article suggestions."""

    def __init__(self, config: GapConfig | None = None) -> None:
        self.config = config or GapConfig()
        self.clusters: list[GapCluster] = []

    def __len__(self) -> int:
        return sum(c.count for c in self.clusters)

    def is_gap(self, best_score: float | None) -> bool:
        return best_score is None or best_score < self.config.gap_threshold

    def observe(self, query: str, best_score: float | None, *, ticket_id: str | None = None) -> GapCluster | None:
        """Record ``query`` if its best answer is a gap; return the cluster it joined."""
        if not self.config.enabled or not self.is_gap(best_score):
            return None
        return self.add(query, ticket_id=ticket_id)

    def observe_feedback(self, feedback: Iterable[Feedback], helpful_threshold: float = 0.5) -> int:
        """Record queries of unhelpful ratings that came without a correction. Returns how many were added."""
        added = 0
        for item in feedback:
            if item.query and item.rating < helpful_threshold and not item.correction:
                if self.add(item.query, ticket_id=item.ticket_id) is not None:
                    added += 1
        return added

    def add(self, query: str, *, ticket_id: str | None = None) -> GapCluster | None:
        """Add a question to the closest cluster (or a new one). Empty questions are ignored."""
        if not self.config.enabled:
            return None
        masked = " ".join(mask_pii(query).split())
        tokens = query_tokens(masked, self.config.min_token_chars)
        if not tokens:
            return None
        cluster = self._closest(tokens)
        if cluster is None:
            cluster = GapCluster(id=f"gap-{len(self.clusters) + 1}", tokens=sorted(tokens))
            self.clusters.append(cluster)
        cluster.count += 1
        if len(cluster.samples) < self.config.max_samples and masked not in cluster.samples:
            cluster.samples.append(masked)
        if ticket_id and ticket_id not in cluster.ticket_ids:
            cluster.ticket_ids.append(ticket_id)
        for token in tokens:
            cluster.term_counts[token] = cluster.term_counts.get(token, 0) + 1
        return cluster

    def _closest(self, tokens: set[str]) -> GapCluster | None:
        best, best_sim = None, 0.0
        for cluster in self.clusters:
            sim = jaccard(tokens, set(cluster.tokens))
            if sim >= self.config.similarity and sim > best_sim:
                best, best_sim = cluster, sim
        return best

    def suggestions(self) -> list[ArticleSuggestion]:
        """Clusters seen at least ``min_occurrences`` times, most frequent first."""
        out = []
        for cluster in self.clusters:
            if cluster.count < self.config.min_occurrences:
                continue
            ranked = Counter(cluster.term_counts).most_common()
            keywords = [term for term, _ in sorted(ranked, key=lambda kv: (-kv[1], kv[0]))][: self.config.max_keywords]
            title = cluster.samples[0] if cluster.samples else " ".join(keywords)
            out.append(
                ArticleSuggestion(
                    cluster_id=cluster.id,
                    title=title[:120],
                    keywords=keywords,
                    occurrences=cluster.count,
                    sample_queries=list(cluster.samples),
                    ticket_ids=list(cluster.ticket_ids),
                )
            )
        return sorted(out, key=lambda s: (-s.occurrences, s.cluster_id))
