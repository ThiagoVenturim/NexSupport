"""Tests for MinHash near-duplicate detection."""

import pytest

from support_engine.ingestion.dedup import (
    DedupConfig,
    MinHasher,
    NearDuplicateIndex,
    deduplicate,
    estimate_jaccard,
    jaccard,
    normalize_for_dedup,
    shingles,
)

BASE = (
    "To reset your password open the account settings page, choose security, click reset password "
    "and follow the link we send to your registered address within fifteen minutes."
)


def test_normalize_placeholders_and_case() -> None:
    assert normalize_for_dedup("Hello [EMAIL_3], CALL [PHONE_12]!") == "hello email call phone"


def test_shingles_short_and_long() -> None:
    assert shingles("one two", 5) == {"one two"}
    assert len(shingles("a b c d e f", 5)) == 2
    assert shingles("", 5) == set()


def test_minhash_is_deterministic_and_estimates_jaccard() -> None:
    a = shingles(BASE, 3)
    b = shingles(BASE.replace("fifteen", "ten"), 3)
    h1, h2 = MinHasher(256, seed=7), MinHasher(256, seed=7)
    assert h1.signature(a) == h2.signature(a)
    est = estimate_jaccard(h1.signature(a), h1.signature(b))
    assert abs(est - jaccard(a, b)) < 0.15


def test_estimate_jaccard_rejects_mismatched_signatures() -> None:
    with pytest.raises(ValueError):
        estimate_jaccard((1, 2), (1,))


def test_index_finds_exact_and_near_duplicates() -> None:
    index = NearDuplicateIndex(DedupConfig(threshold=0.7))
    assert index.add("a", BASE) is None
    assert index.add("b", BASE.upper()) == "a"
    assert index.add("c", BASE.replace("fifteen minutes", "fifteen minutes.")) == "a"
    near = BASE.replace("registered", "primary")
    assert index.add("d", near) == "a"
    assert index.add("e", "The invoice PDF is generated nightly from the billing export job.") is None
    assert len(index) == 2


def test_placeholders_with_different_numbers_are_duplicates() -> None:
    index = NearDuplicateIndex()
    assert index.add("x", f"Hi [NAME_1], {BASE}") is None
    assert index.add("y", f"Hi [NAME_7], {BASE}") == "x"


def test_deduplicate_keeps_first_and_reports_removed() -> None:
    items = [("1", BASE), ("2", "Totally different content about invoices and billing cycles."), ("3", BASE)]
    kept, removed = deduplicate(items, text=lambda i: i[1], key=lambda i: i[0])
    assert [k for k, _ in kept] == ["1", "2"]
    assert removed == {"3": "1"}


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        DedupConfig(num_perm=100, bands=32)
