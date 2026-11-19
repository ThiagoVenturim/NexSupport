"""Text helpers shared by the analytics modules: accent folding, word lists and ticket text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from support_engine.models import Ticket

_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS: frozenset[str] = frozenset(
    {
        # English
        "the", "and", "for", "with", "how", "what", "why", "when", "can", "does", "are", "was", "you",
        "your", "our", "this", "that", "have", "has", "from", "into", "after", "before", "get", "its", "there",
        "they", "them", "then", "than", "but", "out", "all", "any", "will", "would", "could", "should", "been",
        "were", "which", "who", "please", "thanks", "thank", "hello", "regards", "just", "also", "again",
        # Portuguese
        "como", "para", "com", "que", "uma", "por", "meu", "minha", "esta", "isso", "ser", "depois", "antes",
        "quando", "porque", "qual", "onde", "fazer", "consigo", "nos", "das", "dos", "pelo", "pela", "mais",
        "mas", "ele", "ela", "voce", "voces", "obrigado", "obrigada", "ola", "favor", "tem", "estou",
        # Spanish
        "con", "los", "las", "del", "pero", "este", "eso", "hola", "gracias", "cuando",
        "donde", "usted", "tengo", "puedo",
    }
)


def fold(text: str) -> str:
    """Lowercase text without accents, so one word list covers both spellings."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def words(text: str) -> list[str]:
    """Folded words in order, masked placeholders such as [EMAIL_1] removed. Keeps stopwords."""
    return _WORD.findall(fold(_PLACEHOLDER.sub(" ", text)).replace("\u2019", "'"))


def content_tokens(text: str, min_chars: int = 3) -> list[str]:
    """Folded content words in order, without stopwords, short words and numbers only."""
    return [w for w in words(text) if len(w) >= min_chars and w not in STOPWORDS and not w.isdigit()]


def customer_messages(ticket: Ticket, roles: Iterable[str] = ("customer",)) -> list[str]:
    """Public messages of the given roles in time order; the description when there are none."""
    wanted = set(roles)
    ordered = sorted(ticket.comments, key=lambda c: c.created_at)
    messages = [c.body for c in ordered if c.public and c.author_role in wanted]
    messages = [m for m in messages if m.strip()]
    if messages:
        return messages
    opening = "\n".join(p for p in (ticket.subject, ticket.description) if p)
    return [opening] if opening.strip() else []


def topic_text(ticket: Ticket) -> str:
    """Subject and description: what the ticket is about."""
    return "\n".join(p for p in (ticket.subject, ticket.description) if p)


def ticket_field(ticket: Ticket, name: str) -> list[str]:
    """Values of a ticket field, folded. ``tag`` gives every tag; unknown names read ``Ticket.extra``."""
    if name in ("tag", "tags"):
        return [fold(t) for t in ticket.tags if t]
    value = getattr(ticket, name, None) if name in Ticket.model_fields else ticket.extra.get(name)
    if value is None or value == "":
        return []
    if hasattr(value, "value"):
        value = value.value
    return [fold(str(value))]
