"""Text cleanup for support interactions.

Normalises whitespace and Unicode, turns HTML into plain text and removes the parts
of an e-mail thread that add noise to retrieval: quoted earlier replies, signatures,
mobile footers and legal disclaimers. Every step can be switched off in
``CleanupConfig``.
"""

from __future__ import annotations

import html
import re
import unicodedata

from pydantic import BaseModel, Field

DEFAULT_REPLY_HEADER_PATTERNS: list[str] = [
    r"^\s*On .{3,200}wrote:\s*$",
    r"^\s*Em .{3,200}escreveu:\s*$",
    r"^\s*-{2,}\s*(?:Original Message|Forwarded message|Mensagem original|Mensagem encaminhada)\s*-{2,}\s*$",
    r"^\s*_{10,}\s*$",
    r"^\s*-{5,}\s*Reply above this line\s*-{5,}\s*$",
    r"^\s*##-\s*(?:Please type your reply above this line|Digite sua resposta acima desta linha)\s*-##\s*$",
]
"""Lines that start a quoted earlier message; everything from them on is dropped."""

DEFAULT_SIGNOFF_PATTERNS: list[str] = [
    r"best regards", r"kind regards", r"warm regards", r"regards", r"thanks", r"thank you", r"many thanks",
    r"cheers", r"sincerely", r"atenciosamente", r"att", r"abs", r"abra\u00e7os", r"abracos",
    r"obrigad[oa]", r"grat[oa]", r"cordialmente", r"sauda\u00e7\u00f5es", r"saudacoes",
]
"""Closing words that begin a signature block when they sit alone on a line."""

DEFAULT_FOOTER_PATTERNS: list[str] = [
    r"^\s*Sent from my .+$",
    r"^\s*Enviado do meu .+$",
    r"^\s*Enviado de meu .+$",
    r"^\s*Get Outlook for .+$",
    r"^\s*Obter o Outlook para .+$",
]
"""Single-line mobile or client footers."""

DEFAULT_DISCLAIMER_PATTERNS: list[str] = [
    r"^\s*(?:CONFIDENTIALITY NOTICE|DISCLAIMER|AVISO DE CONFIDENCIALIDADE|AVISO LEGAL)\b.*$",
    r"^\s*This (?:e-?mail|message) and any attachments? .*$",
    r"^\s*Esta (?:mensagem|e-?mail) (?:e|pode conter|cont\u00e9m) .*$",
]
"""Lines that start a legal disclaimer; everything from them on is dropped."""

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_SPACE_LIKE = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
_HTML_BREAK = re.compile(r"<\s*(?:br|/p|/div|/li|/tr|/h[1-6])\b[^>]*>", re.IGNORECASE)
_HTML_DROP = re.compile(r"<\s*(script|style|head)\b[^>]*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)


class CleanupConfig(BaseModel):
    """Switches and patterns for ``clean_text``."""

    normalize_unicode: bool = True
    strip_html: bool = True
    strip_quoted_replies: bool = True
    strip_signatures: bool = True
    strip_footers: bool = True
    strip_disclaimers: bool = True
    max_signature_lines: int = Field(default=6, ge=0, description="lines allowed after a sign-off to treat it as one")
    reply_header_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_REPLY_HEADER_PATTERNS))
    signoff_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_SIGNOFF_PATTERNS))
    footer_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_FOOTER_PATTERNS))
    disclaimer_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_DISCLAIMER_PATTERNS))


def normalize_text(text: str, *, normalize_unicode: bool = True) -> str:
    """Normalise Unicode, line endings and whitespace without changing the words."""
    if normalize_unicode:
        text = unicodedata.normalize("NFC", text)
        text = text.translate(_ZERO_WIDTH)
        text = _SPACE_LIKE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _CONTROL.sub("", text)
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(text: str) -> str:
    """Convert an HTML fragment to plain text, keeping paragraph breaks."""
    if not _HTML_TAG.search(text) and "&" not in text:
        return text
    text = _HTML_DROP.sub("", text)
    text = _HTML_BREAK.sub("\n", text)
    text = _HTML_TAG.sub("", text)
    return html.unescape(text)


def _cut_at_first(lines: list[str], patterns: list[re.Pattern[str]]) -> list[str]:
    for idx, line in enumerate(lines):
        if any(p.match(line) for p in patterns):
            return lines[:idx]
    return lines


def strip_quoted_replies(text: str, patterns: list[str] | None = None) -> str:
    """Drop quoted lines (``> ...``) and everything after a reply header such as "On ... wrote:"."""
    compiled = [re.compile(p, re.IGNORECASE) for p in (patterns or DEFAULT_REPLY_HEADER_PATTERNS)]
    lines = _cut_at_first(text.split("\n"), compiled)
    lines = _cut_outlook_header(lines)
    return "\n".join(line for line in lines if not line.lstrip().startswith(">"))


def _cut_outlook_header(lines: list[str]) -> list[str]:
    """Cut at an Outlook style block: a "From:" line followed shortly by "Sent:" or "Date:"."""
    start = re.compile(r"^\s*(?:From|De):\s+\S", re.IGNORECASE)
    follow = re.compile(r"^\s*(?:Sent|Date|Enviado|Enviada|Data|To|Para):\s", re.IGNORECASE)
    for idx, line in enumerate(lines):
        if start.match(line) and any(follow.match(nxt) for nxt in lines[idx + 1 : idx + 4]):
            return lines[:idx]
    return lines


def strip_signature(text: str, signoff_patterns: list[str] | None = None, max_signature_lines: int = 6) -> str:
    """Remove a trailing signature: the ``-- `` delimiter, or a sign-off followed by a few short lines."""
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if line.rstrip() == "--":
            return "\n".join(lines[:idx])
    words = "|".join(signoff_patterns or DEFAULT_SIGNOFF_PATTERNS)
    signoff = re.compile(rf"^\s*(?:{words})\s*[,.!]*\s*$", re.IGNORECASE)
    for idx in range(len(lines) - 1, -1, -1):
        if signoff.match(lines[idx]):
            tail = [t for t in lines[idx + 1 :] if t.strip()]
            if len(tail) <= max_signature_lines and all(len(t) <= 80 for t in tail):
                return "\n".join(lines[:idx])
            break
    return text


def strip_lines(text: str, patterns: list[str]) -> str:
    """Remove every line matching one of the patterns."""
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    return "\n".join(line for line in text.split("\n") if not any(p.match(line) for p in compiled))


def strip_disclaimer(text: str, patterns: list[str] | None = None) -> str:
    """Drop a legal disclaimer and everything after it."""
    compiled = [re.compile(p, re.IGNORECASE) for p in (patterns or DEFAULT_DISCLAIMER_PATTERNS)]
    return "\n".join(_cut_at_first(text.split("\n"), compiled))


def clean_text(text: str, config: CleanupConfig | None = None) -> str:
    """Run the configured cleanup steps and return normalised plain text (possibly empty)."""
    cfg = config or CleanupConfig()
    if not text:
        return ""
    if cfg.strip_html:
        text = html_to_text(text)
    text = normalize_text(text, normalize_unicode=cfg.normalize_unicode)
    if cfg.strip_quoted_replies:
        text = strip_quoted_replies(text, cfg.reply_header_patterns)
    if cfg.strip_disclaimers:
        text = strip_disclaimer(text, cfg.disclaimer_patterns)
    if cfg.strip_footers:
        text = strip_lines(text, cfg.footer_patterns)
    if cfg.strip_signatures:
        text = strip_signature(text.rstrip(), cfg.signoff_patterns, cfg.max_signature_lines)
    return normalize_text(text, normalize_unicode=False)
