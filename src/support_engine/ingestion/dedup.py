"""Exact and near-duplicate detection with MinHash and LSH banding.

Pure Python and deterministic: the same texts and seed always give the same
signatures, so results are reproducible across runs and machines.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Iterable
from typing import TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1
_PLACEHOLDER = re.compile(r"\[([A-Z]+)_\d+\]")
_WORD = re.compile(r"\w+", re.UNICODE)


class DedupConfig(BaseModel):
    """Parameters for near-duplicate detection."""

    enabled: bool = True
    threshold: float = Field(default=0.85, gt=0.0, le=1.0, description="estimated Jaccard similarity")
    num_perm: int = Field(default=128, ge=8)
    bands: int = Field(default=32, ge=1)
    shingle_size: int = Field(default=3, ge=1, description="words per shingle")
    seed: int = 1

    @model_validator(mode="after")
    def _bands_divide_perm(self) -> DedupConfig:
        if self.num_perm % self.bands:
            raise ValueError("num_perm must be a multiple of bands")
        return self


def normalize_for_dedup(text: str) -> str:
    """Lowercase, collapse whitespace and turn ``[EMAIL_3]`` style placeholders into ``[email]``."""
    text = _PLACEHOLDER.sub(lambda m: f" {m.group(1).lower()} ", text)
    return " ".join(_WORD.findall(text.lower()))


def shingles(text: str, size: int = 5) -> set[str]:
    """Word shingles of a normalised text; short texts give one shingle with all words."""
    words = normalize_for_dedup(text).split()
    if not words:
        return set()
    if len(words) <= size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


class MinHasher:
    """Builds MinHash signatures with ``num_perm`` universal hash permutations."""

    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = random.Random(seed)  # noqa: S311 - deterministic hashing, not cryptography
        self.num_perm = num_perm
        self._params = [
            (rng.randint(1, _MERSENNE_PRIME - 1), rng.randint(0, _MERSENNE_PRIME - 1)) for _ in range(num_perm)
        ]

    def signature(self, items: Iterable[str]) -> tuple[int, ...]:
        hashes = [_hash64(i) for i in items]
        if not hashes:
            return tuple([_MAX_HASH] * self.num_perm)
        return tuple(min(((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH for h in hashes) for a, b in self._params)


def estimate_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Share of equal positions between two MinHash signatures."""
    if len(sig_a) != len(sig_b) or not sig_a:
        raise ValueError("signatures must have the same non-zero length")
    return sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b) / len(sig_a)


def jaccard(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class NearDuplicateIndex:
    """Incremental index: ``add`` returns the key of an earlier near-duplicate, or ``None``."""

    def __init__(self, config: DedupConfig | None = None) -> None:
        self.config = config or DedupConfig()
        self._hasher = MinHasher(self.config.num_perm, self.config.seed)
        self._rows = self.config.num_perm // self.config.bands
        self._exact: dict[str, str] = {}
        self._signatures: dict[str, tuple[int, ...]] = {}
        self._buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}

    def __len__(self) -> int:
        return len(self._signatures)

    def find(self, text: str) -> str | None:
        """Key of an indexed text that is an exact or near duplicate of ``text``."""
        key, _sig, _digest = self._lookup(text)
        return key

    def _lookup(self, text: str) -> tuple[str | None, tuple[int, ...], str]:
        normalized = normalize_for_dedup(text)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        sig = self._hasher.signature(shingles(text, self.config.shingle_size))
        if digest in self._exact:
            return self._exact[digest], sig, digest
        best: tuple[float, str] | None = None
        seen: set[str] = set()
        for band in range(self.config.bands):
            chunk = sig[band * self._rows : (band + 1) * self._rows]
            for candidate in self._buckets.get((band, chunk), []):
                if candidate in seen:
                    continue
                seen.add(candidate)
                score = estimate_jaccard(sig, self._signatures[candidate])
                if score >= self.config.threshold and (best is None or score > best[0]):
                    best = (score, candidate)
        return (best[1] if best else None), sig, digest

    def add(self, key: str, text: str) -> str | None:
        """Index ``text`` under ``key`` unless it duplicates an earlier text; return that text's key."""
        duplicate_of, sig, digest = self._lookup(text)
        if duplicate_of is not None:
            return duplicate_of
        self._exact[digest] = key
        self._signatures[key] = sig
        for band in range(self.config.bands):
            chunk = sig[band * self._rows : (band + 1) * self._rows]
            self._buckets.setdefault((band, chunk), []).append(key)
        return None


def deduplicate(
    items: Iterable[T],
    *,
    text: Callable[[T], str],
    key: Callable[[T], str],
    config: DedupConfig | None = None,
) -> tuple[list[T], dict[str, str]]:
    """Keep the first of each group of near-duplicates.

    Returns the kept items in order and a map ``removed key -> kept key``.
    """
    index = NearDuplicateIndex(config)
    kept: list[T] = []
    removed: dict[str, str] = {}
    for item in items:
        original = index.add(key(item), text(item))
        if original is None:
            kept.append(item)
        else:
            removed[key(item)] = original
    return kept, removed
