"""Guards around the support agent: PII masking, prompt injection, scope, sensitive topics and limits.

Every question is masked before it reaches retrieval, the cache, a model or a log line.
The injection guard is a set of English and Portuguese patterns run on accent-folded text;
extra detectors (for example a classifier or ``nexcoder_core.PromptFirewall``) can be added
as callables that return True when a text is an attack. Retrieved ticket text is scanned
too, because customers can write instructions into a ticket that later becomes context.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field

from support_engine.agent.config import GuardConfig
from support_engine.pii import Anonymizer

InjectionDetector = Callable[[str], bool]

INJECTION_PATTERNS: list[str] = [
    # English
    r"\b(?:ignore|disregard|skip|override|bypass)\b[\w\s,]{0,30}?\b(?:previous|prior|above|earlier|system|all|"
    r"any|your)\b[\w\s]{0,20}?\b(?:instructions?|rules|prompts?|directions|guidelines|messages?)\b",
    r"\bforget\b[\w\s]{0,20}?\b(?:instructions?|rules|everything|prompts?)\b",
    r"\b(?:reveal|show|print|repeat|display|leak|tell me)\b[\w\s]{0,20}?\b(?:system prompt|hidden prompt|"
    r"initial prompt|your prompt|your instructions|your rules)\b",
    r"\byou are now\b",
    r"\b(?:act|behave) as\b[\w\s]{0,15}?\b(?:unrestricted|unfiltered|jailbroken|dan|developer mode|admin)\b",
    r"\b(?:developer|god|jailbreak|dan) mode\b",
    r"\bjailbreak\b",
    r"\bnew instructions?\s*:",
    # Portuguese (accent-folded)
    r"\b(?:ignore|ignora|desconsidere|desconsidera|esqueca|esquece|pule)\b[\w\s,]{0,30}?\b(?:instrucoes|"
    r"instrucao|regras|prompt|comandos|orientacoes)\b",
    r"\b(?:revele|mostre|mostra|imprima|repita|exiba)\b[\w\s]{0,20}?\b(?:prompt|instrucoes|regras)\b",
    r"\bvoce agora e\b",
    r"\b(?:aja|atue|finja ser|comporte-se) como\b[\w\s]{0,15}?\b(?:sem restricoes|sem filtro|admin|"
    r"administrador|desenvolvedor)\b",
    r"\bmodo (?:desenvolvedor|deus|sem restricoes)\b",
    r"\bnovas instrucoes\s*:",
    # Chat role markers smuggled into user text
    r"^\s*(?:system|assistant|sistema)\s*:",
    r"<\|?(?:im_start|im_end|system|endoftext)\|?>",
    r"</?\s*(?:system|instructions?)\s*>",
    r"\[/?(?:inst|sys)\]",
]

HUMAN_REQUEST_PATTERNS: list[str] = [
    r"\b(?:talk|speak|chat)\b[\w\s]{0,12}?\b(?:to|with)\b[\w\s]{0,10}?\b(?:human|person|agent|someone|"
    r"representative|manager|operator)\b",
    r"\b(?:real|live) (?:person|human|agent)\b",
    r"\bhuman (?:agent|support)\b",
    r"\b(?:falar|conversar)\b[\w\s]{0,12}?\b(?:com)\b[\w\s]{0,10}?\b(?:humano|pessoa|atendente|gerente|"
    r"operador|alguem)\b",
    r"\batendimento humano\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]
_HUMAN_RE = [re.compile(p, re.IGNORECASE) for p in HUMAN_REQUEST_PATTERNS]
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def fold(text: str) -> str:
    """Lowercase text without accents, so one pattern covers both spellings."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _mentions(folded: str, terms: Iterable[str]) -> list[str]:
    found = []
    for term in terms:
        t = fold(term).strip()
        if t and re.search(rf"(?<!\w){re.escape(t)}(?!\w)", folded):
            found.append(term)
    return found


class InputVerdict(BaseModel):
    """Outcome of the input guard. ``question`` is the masked text; the raw text is not kept."""

    question: str
    blocked: str | None = Field(default=None, description="too_long, injection or out_of_scope when refused")
    escalate: list[str] = Field(default_factory=list, description="requested_human or sensitive_topic")
    flags: list[str] = Field(default_factory=list)
    pii_masked: dict[str, int] = Field(default_factory=dict, description="counts per placeholder prefix")


class OutputVerdict(BaseModel):
    answer: str
    flags: list[str] = Field(default_factory=list)
    pii_masked: dict[str, int] = Field(default_factory=dict)


class Guard:
    """Input and output checks for the support agent."""

    def __init__(self, config: GuardConfig | None = None, *, detectors: Iterable[InjectionDetector] = ()) -> None:
        self.config = config or GuardConfig()
        self.detectors = list(detectors)

    # Injection ------------------------------------------------------------------------------

    def is_injection(self, text: str) -> bool:
        """True when ``text`` tries to override the agent instructions."""
        folded = fold(text)
        if any(rx.search(folded) for rx in _INJECTION_RE):
            return True
        for detector in self.detectors:
            try:
                if detector(text):
                    return True
            except Exception:
                # A broken detector must not open the gate silently.
                return True
        return False

    def wants_human(self, text: str) -> bool:
        folded = fold(text)
        return any(rx.search(folded) for rx in _HUMAN_RE)

    # Input ----------------------------------------------------------------------------------

    def check_input(self, question: str) -> InputVerdict:
        """Mask PII first, then apply limits, injection, scope and escalation rules."""
        cfg = self.config
        question = question.strip()
        if len(question) > cfg.max_question_chars:
            # Nothing of an oversized message is kept.
            return InputVerdict(question="", blocked="too_long", flags=["too_long"])

        anonymizer = Anonymizer(mask_names=cfg.mask_names)
        masked = anonymizer.anonymize_text(question)
        verdict = InputVerdict(question=masked, pii_masked=anonymizer.counts)
        if anonymizer.counts:
            verdict.flags.append("pii_in_question")

        if cfg.block_injection and self.is_injection(question):
            verdict.blocked = "injection"
            verdict.flags.append("injection")
            return verdict

        folded = fold(masked)
        if cfg.allowed_topics and not _mentions(folded, cfg.allowed_topics):
            verdict.blocked = "out_of_scope"
            verdict.flags.append("out_of_scope")
            return verdict

        if cfg.detect_human_request and self.wants_human(masked):
            verdict.escalate.append("requested_human")
        if _mentions(folded, cfg.sensitive_topics):
            verdict.escalate.append("sensitive_topic")
        verdict.flags.extend(verdict.escalate)
        return verdict

    def is_safe_source(self, text: str) -> bool:
        """False for retrieved text that carries injection instructions."""
        return not (self.config.scan_sources and self.is_injection(text))

    # Output ---------------------------------------------------------------------------------

    def check_output(self, answer: str) -> OutputVerdict:
        """Mask any PII the model wrote, refuse prompt echoes and cut overlong answers."""
        anonymizer = Anonymizer()
        masked = anonymizer.anonymize_text(answer.strip())
        verdict = OutputVerdict(answer=masked, pii_masked=anonymizer.counts)
        if anonymizer.counts:
            verdict.flags.append("pii_in_answer")
        if self.is_injection(masked):
            verdict.flags.append("injection_in_answer")
        limit = self.config.max_answer_chars
        if len(masked) > limit:
            verdict.answer = truncate_sentences(masked, limit)
            verdict.flags.append("answer_truncated")
        return verdict


def truncate_sentences(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, at a sentence end when possible."""
    if len(text) <= limit:
        return text
    out = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{out} {sentence}" if out else sentence
        if len(candidate) > limit:
            break
        out = candidate
    return out or text[: limit - 3].rstrip() + "..."
