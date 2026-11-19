"""Tests for personal name masking and the NER hook."""

from types import SimpleNamespace

from support_engine.pii import Anonymizer, build_ner_detector, detect_names


def test_detect_names_from_context() -> None:
    text = (
        "Hi Maria da Silva, thanks for writing.\n"
        "My name is Carlos Pereira and I manage the account.\n"
        "Sra. Oliveira approved it.\n"
        "Atenciosamente,\nJo\u00e3o Souza"
    )
    names = detect_names(text)
    assert "Maria da Silva" in names
    assert "Carlos Pereira" in names
    assert "Oliveira" in names
    assert "Jo\u00e3o Souza" in names


def test_detect_names_ignores_generic_greetings() -> None:
    assert detect_names("Hello Team, hi there. Bom dia equipe. Ola Pessoal") == []


def test_anonymizer_masks_names_everywhere_in_text() -> None:
    anon = Anonymizer(mask_names=True)
    masked = anon.anonymize_text("Hello Ana Lima, your ticket is open. Ana Lima can reply here; Ana is fine.")
    assert "Ana" not in masked
    assert "Lima" not in masked
    assert "[NAME_1]" in masked
    assert anon.counts["NAME"] >= 1


def test_names_off_by_default_keeps_previous_behaviour() -> None:
    anon = Anonymizer()
    assert anon.anonymize_text("Hi Maria") == "Hi Maria"


def test_known_names_are_masked_without_context() -> None:
    anon = Anonymizer(mask_names=True, known_names=["Beatriz Costa"])
    masked = anon.anonymize_text("The request from Beatriz Costa was escalated. Costa called again.")
    assert "Beatriz" not in masked
    assert "Costa" not in masked


def test_ner_detector_hook_with_fake_model() -> None:
    def fake_nlp(text: str) -> SimpleNamespace:
        ents = [SimpleNamespace(text="Rafael", label_="PER"), SimpleNamespace(text="Lisboa", label_="LOC")]
        return SimpleNamespace(ents=[e for e in ents if e.text in text])

    anon = Anonymizer(mask_names=True, name_detectors=[build_ner_detector(fake_nlp)])
    masked = anon.anonymize_text("Rafael opened the case from Lisboa.")
    assert "Rafael" not in masked
    assert "Lisboa" in masked


def test_names_and_regex_rules_combine_and_restore() -> None:
    anon = Anonymizer(mask_names=True)
    text = "Dear Paulo Mendes, we sent the invoice to paulo.mendes@example.com."
    masked = anon.anonymize_text(text)
    assert "Paulo" not in masked
    assert "paulo.mendes@example.com" not in masked
    assert anon.deanonymize_text(masked) == text
