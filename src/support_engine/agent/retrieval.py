"""Retrieval for the support agent over curated chunks.

A retriever is any callable ``(query, k) -> list[(Chunk, relevance)]`` with relevance in 0..1,
so a vector or hybrid store (for example ``nexcoder_core`` retrieval) can be plugged in.
``LexicalRetriever`` is the offline default: BM25 over the chunks produced by ingestion and
curation, normalised to 0..1 and blended with the curation priority through
``CurationPipeline.rerank`` so proven answers and human corrections rank first.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field

from support_engine.agent.guards import fold
from support_engine.curation import CurationPipeline, chunk_priority
from support_engine.ingestion import Chunk

Retriever = Callable[[str, int], list[tuple[Chunk, float]]]

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "be", "it",
        "my", "your", "our", "we", "you", "i", "me", "this", "that", "how", "what", "why", "when", "can", "do",
        "does", "not", "no", "please", "hi", "hello", "thanks", "at", "by", "from", "as", "if", "so", "but",
        "o", "os", "um", "uma", "de", "da", "das", "dos", "e", "em", "na", "nos", "nas", "para",
        "por", "com", "que", "se", "meu", "minha", "nao", "como", "ola", "oi", "obrigado", "obrigada",
    }
)


def tokens(text: str) -> list[str]:
    """Lowercase content words without masked placeholders and stopwords."""
    text = _PLACEHOLDER.sub(" ", text)
    return [t for t in (m.group(0) for m in _TOKEN.finditer(fold(text))) if len(t) > 1 and t not in STOPWORDS]


class Source(BaseModel):
    """A retrieved document as shown to the model, with the label used to cite it."""

    label: str = Field(description="citation label, for example S1")
    chunk_id: str
    ticket_id: str
    kind: str
    text: str
    relevance: float = 0.0
    priority: float = 0.5

    @classmethod
    def from_chunk(cls, label: str, chunk: Chunk, relevance: float, max_chars: int) -> Source:
        text = chunk.text if len(chunk.text) <= max_chars else chunk.text[: max_chars - 3].rstrip() + "..."
        return cls(
            label=label,
            chunk_id=chunk.id,
            ticket_id=chunk.ticket_id,
            kind=chunk.kind,
            text=text,
            relevance=round(relevance, 6),
            priority=chunk_priority(chunk),
        )


class LexicalRetriever:
    """BM25 retriever over chunks, reranked by curation priority."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        curation: CurationPipeline | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.curation = curation or CurationPipeline()
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        df: Counter[str] = Counter()
        for chunk in chunks:
            terms = Counter(tokens(chunk.embed_text))
            if not terms:
                continue
            self.chunks.append(chunk)
            self._tf.append(terms)
            self._len.append(sum(terms.values()))
            df.update(terms.keys())
        n = len(self.chunks)
        self._avg = (sum(self._len) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self.chunks)

    def _bm25(self, idx: int, query: list[str]) -> float:
        tf, length, score = self._tf[idx], self._len[idx], 0.0
        for term in query:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * length / (self._avg or 1.0))
            score += self._idf.get(term, 0.0) * f * (self.k1 + 1) / denom
        return score

    def relevance(self, query: str) -> list[tuple[Chunk, float]]:
        """Every matching chunk with a 0..1 relevance (BM25 over the sum of the query term weights, capped)."""
        q = list(dict.fromkeys(tokens(query)))
        if not q or not self.chunks:
            return []
        # A chunk of average length holding every query term once scores 1.0.
        ceiling = sum(self._idf.get(t, 0.0) for t in q)
        if ceiling <= 0:
            return []
        out = []
        for idx, chunk in enumerate(self.chunks):
            raw = self._bm25(idx, q)
            if raw > 0:
                out.append((chunk, min(1.0, raw / ceiling)))
        return out

    def __call__(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Top ``k`` chunks; the returned score is the plain relevance, ordering uses curation priority too."""
        hits = self.relevance(query)
        relevance = {c.id: rel for c, rel in hits}
        ranked = self.curation.rerank(hits)
        return [(chunk, relevance[chunk.id]) for chunk, _ in ranked[:k]]
