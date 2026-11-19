"""PII masking and anonymization engine for support tickets.

Provides regex-based detection and deterministic masking of Personally Identifiable
Information (PII) including email addresses, credit card numbers, Brazilian CNPJ/CPF,
phone numbers, IPv4 addresses, and secret API keys.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from support_engine.models import Ticket


class AnonymizationRule(BaseModel):
    """Specification of a regex rule for detecting and replacing sensitive entities."""

    name: str
    pattern: str
    placeholder_prefix: str


DEFAULT_PII_RULES: list[AnonymizationRule] = [
    AnonymizationRule(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        placeholder_prefix="EMAIL",
    ),
    AnonymizationRule(
        name="credit_card",
        pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        placeholder_prefix="CARD",
    ),
    AnonymizationRule(
        name="cnpj",
        pattern=r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
        placeholder_prefix="CNPJ",
    ),
    AnonymizationRule(
        name="cpf",
        pattern=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
        placeholder_prefix="CPF",
    ),
    AnonymizationRule(
        name="phone",
        pattern=r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\d{4}|\d{4})[-.\s]?\d{4}\b",
        placeholder_prefix="PHONE",
    ),
    AnonymizationRule(
        name="ipv4",
        pattern=r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        placeholder_prefix="IP",
    ),
    AnonymizationRule(
        name="secret_key",
        pattern=r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|bearer\s+[A-Za-z0-9._-]{20,})\b",
        placeholder_prefix="SECRET",
    ),
]


NAME_PLACEHOLDER_PREFIX = "NAME"

# One capitalised word, accented Latin letters included (A-Z plus U+00C0..U+00DE, minus U+00D7).
_UPPER = r"A-Z\u00c0-\u00d6\u00d8-\u00de"
_LOWER = r"a-z\u00df-\u00f6\u00f8-\u00ff"
_CAP_WORD = rf"[{_UPPER}][{_LOWER}]+(?:[-'][{_UPPER}][{_LOWER}]+)?"
# Up to four capitalised words, allowing Portuguese particles such as "da" or "dos" in between.
_FULL_NAME = rf"{_CAP_WORD}(?:[ \t]+(?:(?:da|de|do|das|dos|e)[ \t]+)?{_CAP_WORD}){{0,3}}"

_NAME_CONTEXT_PATTERNS: list[re.Pattern[str]] = [
    # Greetings: "Hi Maria", "Dear John Smith", "Ola Joao", "Prezada Ana Souza".
    re.compile(
        r"(?:^|(?<=[\s,.;:!?]))(?i:hi|hello|hey|dear|ola|ol\u00e1|oi|prezad[oa]|car[oa]|"
        r"bom dia|boa tarde|boa noite)[ \t]*,?[ \t]+(?P<name>" + _FULL_NAME + ")",
        re.MULTILINE,
    ),
    # Self introductions: "my name is Ana", "this is Carlos", "meu nome e Pedro", "me chamo Rita".
    re.compile(
        r"(?i:my name is|this is|meu nome \u00e9|meu nome e|me chamo|aqui \u00e9 o|aqui \u00e9 a)[ \t]+"
        r"(?P<name>" + _FULL_NAME + ")"
    ),
    # Honorifics: "Mr. Smith", "Sra. Oliveira", "Dr Silva".
    re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Dra|Sr|Sra|Srta)\.?[ \t]+(?P<name>" + _FULL_NAME + ")"),
    # Sign-off followed by a line holding only a name: "Regards,\nJohn Smith".
    re.compile(
        r"^[ \t]*(?i:best regards|kind regards|regards|thanks|thank you|cheers|sincerely|"
        r"atenciosamente|att|abs|abra\u00e7os|abracos|obrigad[oa]|grat[oa]|cordialmente)[ \t]*[,.!]?[ \t]*\n"
        r"[ \t]*(?P<name>" + _FULL_NAME + r")[ \t]*$",
        re.MULTILINE,
    ),
]

# Capitalised words that follow greetings but are not personal names.
NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        "all", "everyone", "team", "there", "support", "folks", "guys", "sir", "madam", "customer",
        "todos", "todas", "equipe", "time", "pessoal", "suporte", "senhor", "senhora", "cliente",
        "good", "morning", "afternoon", "evening", "bom", "boa", "dia", "tarde", "noite",
        "thanks", "thank", "obrigado", "obrigada", "regards", "hello", "hi",
        "please", "we", "our", "my", "the", "this", "por", "favor", "nos", "estou", "gostaria",
    }
)

NameDetector = Callable[[str], Iterable[str]]
"""A callable that returns the personal names found in a text (for example an NER model)."""


def detect_names(text: str, stopwords: frozenset[str] = NAME_STOPWORDS) -> list[str]:
    """Find personal names using context rules (greetings, introductions, honorifics, sign-offs)."""
    found: list[str] = []
    for pattern in _NAME_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            words = [w for w in match.group("name").split() if w.lower() not in stopwords]
            # Drop leading or trailing particles left after removing stopwords.
            while words and words[0] in {"da", "de", "do", "das", "dos", "e"}:
                words.pop(0)
            while words and words[-1] in {"da", "de", "do", "das", "dos", "e"}:
                words.pop()
            if words:
                found.append(" ".join(words))
    return found


def build_ner_detector(nlp: Any, labels: Iterable[str] = ("PER", "PERSON")) -> NameDetector:
    """Wrap an already loaded NER pipeline (spaCy style: ``nlp(text).ents``) as a ``NameDetector``.

    The model is supplied by the caller, so no model is downloaded and no network is used here.
    """
    wanted = set(labels)

    def _detect(text: str) -> list[str]:
        doc = nlp(text)
        return [ent.text for ent in getattr(doc, "ents", []) if getattr(ent, "label_", "") in wanted]

    return _detect


class Anonymizer:
    """Anonymizes sensitive PII deterministically across text and support records.

    Regex rules always run. Personal names are masked when ``mask_names`` is true, using the
    context rules of ``detect_names``, any ``known_names`` (for example requester names taken
    from structured fields) and optional extra ``name_detectors`` such as an NER model.
    """

    def __init__(
        self,
        custom_rules: list[AnonymizationRule] | None = None,
        *,
        mask_names: bool = False,
        known_names: Iterable[str] = (),
        name_detectors: Iterable[NameDetector] = (),
        name_stopwords: frozenset[str] = NAME_STOPWORDS,
    ) -> None:
        self.rules = custom_rules or DEFAULT_PII_RULES
        self._compiled = [(r, re.compile(r.pattern, re.IGNORECASE)) for r in self.rules]
        self.mapping: dict[str, str] = {}
        self._reverse_mapping: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self.mask_names = mask_names
        self.name_detectors = list(name_detectors)
        self.name_stopwords = name_stopwords
        self._known_names: set[str] = set()
        for name in known_names:
            self.add_known_name(name)

    @property
    def counts(self) -> dict[str, int]:
        """Number of distinct values masked per placeholder prefix (never the values themselves)."""
        return dict(self._counters)

    def add_known_name(self, name: str) -> None:
        """Register a personal name (and its longer parts) to be masked wherever it appears."""
        name = " ".join(name.split())
        if len(name) < 2:
            return
        self._known_names.add(name)
        for part in name.split():
            if len(part) >= 3 and part.lower() not in self.name_stopwords and part[0].isupper():
                self._known_names.add(part)

    def _mask_names(self, text: str) -> str:
        for name in detect_names(text, self.name_stopwords):
            self.add_known_name(name)
        for detector in self.name_detectors:
            for name in detector(text):
                self.add_known_name(name)
        if not self._known_names:
            return text
        # Longest names first so "Maria Silva" wins over "Maria".
        ordered = sorted(self._known_names, key=len, reverse=True)
        pattern = re.compile(r"(?<![\w\[])(?:" + "|".join(re.escape(n) for n in ordered) + r")(?![\w\]])")
        return pattern.sub(lambda m: self._get_placeholder(m.group(0), NAME_PLACEHOLDER_PREFIX), text)

    def _get_placeholder(self, value: str, prefix: str) -> str:
        """Get or assign a consistent placeholder for a sensitive value."""
        if value in self.mapping:
            return self.mapping[value]

        count = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = count
        placeholder = f"[{prefix}_{count}]"
        self.mapping[value] = placeholder
        self._reverse_mapping[placeholder] = value
        return placeholder

    def anonymize_text(self, text: str) -> str:
        """Scan text and replace all matched PII entities with placeholders."""
        if not text:
            return text

        result = text
        for rule, regex in self._compiled:
            prefix = rule.placeholder_prefix

            def _repl(match: re.Match[str], p: str = prefix) -> str:
                return self._get_placeholder(match.group(0), p)

            result = regex.sub(_repl, result)

        if self.mask_names:
            result = self._mask_names(result)
        return result

    def deanonymize_text(self, text: str) -> str:
        """Restore original values from placeholders using the internal mapping."""
        if not text or not self._reverse_mapping:
            return text

        result = text
        for placeholder, original in self._reverse_mapping.items():
            result = result.replace(placeholder, original)
        return result

    def anonymize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Recursively anonymize all string values inside a dictionary or record."""
        out: dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, str):
                out[k] = self.anonymize_text(v)
            elif isinstance(v, dict):
                out[k] = self.anonymize_record(v)
            elif isinstance(v, list):
                out[k] = [self.anonymize_text(i) if isinstance(i, str) else i for i in v]
            else:
                out[k] = v
        return out


def mask_pii(text: str) -> str:
    """Mask sensitive PII in text using the default rules."""
    anonymizer = Anonymizer()
    return anonymizer.anonymize_text(text)


def mask_ticket(ticket: Ticket, anonymizer: Anonymizer | None = None) -> Ticket:
    """Return a copy of the ticket with subject, description and comment bodies masked."""
    anon = anonymizer or Anonymizer()
    masked_comments = [
        comment.model_copy(update={"body": anon.anonymize_text(comment.body)})
        for comment in ticket.comments
    ]
    return ticket.model_copy(
        update={
            "subject": anon.anonymize_text(ticket.subject),
            "description": anon.anonymize_text(ticket.description),
            "comments": masked_comments,
        }
    )
