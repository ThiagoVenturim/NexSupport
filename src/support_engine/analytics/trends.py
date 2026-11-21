"""Live trends: topic clusters counted per time window and compared with their own baseline.

``TrendDetector`` is incremental: call ``observe`` for each ticket as it arrives (in any
order) and ``report`` whenever you need the current picture. Tickets join a topic cluster
by token Jaccard of their subject, description, tags and product (leader clustering, so
the result is deterministic for a given arrival order). For each cluster the detector
counts tickets per window of ``window_hours`` and compares the window that contains
``now`` with the ``baseline_windows`` before it:

- ``z = (current - mean) / max(std, min_std)`` and ``ratio = current / mean``;
- ``new``: no tickets in the baseline and at least ``min_count`` now;
- ``spike``: at least ``min_count`` now and ``z >= z_threshold`` or ``ratio >= ratio_threshold``;
- ``rising`` / ``falling``: above or below the baseline mean without being a spike;
- ``stable`` otherwise.

``new`` and ``spike`` raise an alert. Only tokens and ticket ids are kept, never text.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from support_engine.analytics.config import TrendConfig
from support_engine.analytics.text import content_tokens, fold, topic_text
from support_engine.ingestion.dedup import jaccard
from support_engine.models import Ticket

TrendStatus = Literal["new", "spike", "rising", "stable", "falling"]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class TopicCluster(BaseModel):
    """Tickets about the same topic. Tokens and ids only."""

    id: str
    tokens: list[str] = Field(default_factory=list, description="token set of the first ticket")
    term_counts: dict[str, int] = Field(default_factory=dict)
    windows: dict[int, list[str]] = Field(default_factory=dict, description="window index to ticket ids")

    @property
    def size(self) -> int:
        return sum(len(ids) for ids in self.windows.values())


class Trend(BaseModel):
    """Current window of one topic against its baseline."""

    cluster_id: str
    keywords: list[str]
    current: int
    baseline_mean: float
    baseline_std: float
    z_score: float
    ratio: float | None = Field(default=None, description="current / baseline mean; None when the baseline is 0")
    status: TrendStatus
    alert: bool
    series: list[int] = Field(default_factory=list, description="baseline windows, oldest first, then the current")
    ticket_ids: list[str] = Field(default_factory=list, description="tickets of the current window")
    window_start: datetime
    window_end: datetime


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class TrendDetector:
    """Clusters tickets into topics and flags topics that grow faster than their baseline."""

    def __init__(self, config: TrendConfig | None = None) -> None:
        self.config = config or TrendConfig()
        self.clusters: list[TopicCluster] = []
        self._seen: dict[str, str] = {}
        self._latest: datetime | None = None

    def __len__(self) -> int:
        return len(self._seen)

    def window_of(self, moment: datetime) -> int:
        hours = (_utc(moment) - _EPOCH).total_seconds() / 3600
        return math.floor(hours / self.config.window_hours)

    def window_bounds(self, index: int) -> tuple[datetime, datetime]:
        start = _EPOCH + timedelta(hours=index * self.config.window_hours)
        return start, start + timedelta(hours=self.config.window_hours)

    def tokens_for(self, ticket: Ticket) -> set[str]:
        tokens = set(content_tokens(topic_text(ticket), self.config.min_token_chars))
        if self.config.use_tags:
            tokens |= {f"tag:{fold(t)}" for t in ticket.tags if t}
            if ticket.product:
                tokens.add(f"product:{fold(ticket.product)}")
        return tokens

    def observe(self, ticket: Ticket) -> TopicCluster | None:
        """Add ``ticket`` to its topic. A ticket id seen before is ignored. Returns the cluster."""
        if ticket.id in self._seen:
            return self._cluster(self._seen[ticket.id])
        tokens = self.tokens_for(ticket)
        if not tokens:
            return None
        cluster = self._closest(tokens)
        if cluster is None:
            cluster = TopicCluster(id=f"topic-{len(self.clusters) + 1}", tokens=sorted(tokens))
            self.clusters.append(cluster)
        window = self.window_of(ticket.created_at)
        cluster.windows.setdefault(window, []).append(ticket.id)
        for token in tokens:
            cluster.term_counts[token] = cluster.term_counts.get(token, 0) + 1
        self._seen[ticket.id] = cluster.id
        created = _utc(ticket.created_at)
        if self._latest is None or created > self._latest:
            self._latest = created
        return cluster

    def observe_many(self, tickets: Iterable[Ticket]) -> int:
        """Observe tickets in creation order. Returns how many joined a cluster."""
        ordered = sorted(tickets, key=lambda t: (_utc(t.created_at), t.id))
        return sum(self.observe(t) is not None for t in ordered)

    def cluster_of(self, ticket_id: str) -> str | None:
        return self._seen.get(ticket_id)

    def _cluster(self, cluster_id: str) -> TopicCluster | None:
        return next((c for c in self.clusters if c.id == cluster_id), None)

    def _closest(self, tokens: set[str]) -> TopicCluster | None:
        best, best_sim = None, 0.0
        for cluster in self.clusters:
            sim = jaccard(tokens, set(cluster.tokens))
            if sim >= self.config.similarity and sim > best_sim:
                best, best_sim = cluster, sim
        return best

    def keywords(self, cluster: TopicCluster) -> list[str]:
        ranked = sorted(Counter(cluster.term_counts).items(), key=lambda kv: (-kv[1], kv[0]))
        return [term for term, _ in ranked][: self.config.max_keywords]

    def report(self, now: datetime | None = None) -> list[Trend]:
        """Every topic with tickets in the current or baseline windows; alerts first, then by z-score."""
        moment = now or self._latest
        if moment is None:
            return []
        cfg = self.config
        current = self.window_of(moment)
        start, end = self.window_bounds(current)
        out = []
        for cluster in self.clusters:
            series = [len(cluster.windows.get(w, [])) for w in range(current - cfg.baseline_windows, current + 1)]
            baseline, count = series[:-1], series[-1]
            if not any(series):
                continue
            mean = sum(baseline) / len(baseline)
            std = math.sqrt(sum((x - mean) ** 2 for x in baseline) / len(baseline))
            z = (count - mean) / max(std, cfg.min_std)
            ratio = count / mean if mean > 0 else None
            status = self._status(count, mean, z, ratio)
            out.append(
                Trend(
                    cluster_id=cluster.id,
                    keywords=self.keywords(cluster),
                    current=count,
                    baseline_mean=round(mean, 4),
                    baseline_std=round(std, 4),
                    z_score=round(z, 4),
                    ratio=round(ratio, 4) if ratio is not None else None,
                    status=status,
                    alert=status in ("new", "spike"),
                    series=series,
                    ticket_ids=list(cluster.windows.get(current, [])),
                    window_start=start,
                    window_end=end,
                )
            )
        return sorted(out, key=lambda t: (not t.alert, -t.z_score, t.cluster_id))

    def alerts(self, now: datetime | None = None) -> list[Trend]:
        return [t for t in self.report(now) if t.alert]

    def _status(self, count: int, mean: float, z: float, ratio: float | None) -> TrendStatus:
        cfg = self.config
        if count >= cfg.min_count:
            if mean == 0:
                return "new"
            if z >= cfg.z_threshold or (ratio is not None and ratio >= cfg.ratio_threshold):
                return "spike"
        if count > mean:
            return "rising"
        if count < mean:
            return "falling"
        return "stable"


def detect_trends(
    tickets: Iterable[Ticket], config: TrendConfig | None = None, now: datetime | None = None
) -> list[Trend]:
    """One shot helper: observe every ticket and report."""
    detector = TrendDetector(config)
    detector.observe_many(tickets)
    return detector.report(now)
