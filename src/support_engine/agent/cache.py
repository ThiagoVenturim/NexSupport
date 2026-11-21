"""Semantic cache of approved answers.

A question close enough to one already answered at level 1 (cosine similarity at or above
``CacheConfig.threshold``) gets the stored answer without retrieval or model calls. Keys are
masked questions, so the cache holds no raw personal data. Only level 1 answers are stored,
which keeps drafts, escalations and refusals out of it.

The embedder is any object with ``embed_query(text) -> list[float]``, for example an
embedder built by ``nexcoder_core``. ``HashingEmbedder`` is the offline default: a signed
hashed bag of words and word bigrams.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from support_engine.agent.citations import Citation
from support_engine.agent.config import CacheConfig
from support_engine.agent.retrieval import tokens


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic offline embedder: hashed unigrams and bigrams, L2 normalised."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        words = tokens(text)
        features = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        vec = [0.0] * self.dim
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vec[value % self.dim] += 1.0 if (value >> 32) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class CacheEntry(BaseModel):
    question: str = Field(description="masked question")
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    score: float | None = None
    namespace: str = "default"
    created_at: float = 0.0
    vector: list[float] = Field(default_factory=list, repr=False)


class CacheHit(BaseModel):
    entry: CacheEntry
    similarity: float


class SemanticCache:
    """In-memory semantic cache with a time to live and a size bound (oldest entries go first)."""

    def __init__(
        self,
        config: CacheConfig | None = None,
        *,
        embedder: Embedder | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or CacheConfig()
        self.embedder = embedder or HashingEmbedder()
        self.clock = clock
        self.entries: list[CacheEntry] = []
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self.entries)

    def _expired(self, entry: CacheEntry, now: float) -> bool:
        return now - entry.created_at > self.config.ttl_seconds

    def get(self, question: str, namespace: str = "default") -> CacheHit | None:
        if not self.config.enabled or not question.strip():
            return None
        now = self.clock()
        self.entries = [e for e in self.entries if not self._expired(e, now)]
        vector = self.embedder.embed_query(question)
        best: CacheHit | None = None
        for entry in self.entries:
            if entry.namespace != namespace:
                continue
            sim = cosine(vector, entry.vector)
            if sim >= self.config.threshold and (best is None or sim > best.similarity):
                best = CacheHit(entry=entry, similarity=round(sim, 6))
        if best is None:
            self.misses += 1
        else:
            self.hits += 1
        return best

    def put(self, question: str, answer: str, namespace: str = "default", **extra: Any) -> bool:
        """Store an approved answer. Returns False when the cache is off or the input is empty."""
        if not self.config.enabled or not question.strip() or not answer.strip():
            return False
        vector = self.embedder.embed_query(question)
        if not any(vector):
            return False
        self.entries = [e for e in self.entries if not (e.namespace == namespace and e.question == question)]
        self.entries.append(
            CacheEntry(
                question=question, answer=answer, namespace=namespace, created_at=self.clock(), vector=vector, **extra
            )
        )
        overflow = len(self.entries) - self.config.max_entries
        if overflow > 0:
            self.entries = self.entries[overflow:]
        return True

    def clear(self) -> None:
        self.entries.clear()
