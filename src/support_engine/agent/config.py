"""Parameters of the support agent: guards, retrieval, citations, judge, levels and cache."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

DEFAULT_SENSITIVE_TOPICS: list[str] = [
    "lawsuit",
    "legal action",
    "lawyer",
    "chargeback",
    "fraud",
    "data breach",
    "processo judicial",
    "advogado",
    "procon",
    "fraude",
    "vazamento de dados",
]


class GuardConfig(BaseModel):
    """Input and output guards."""

    max_question_chars: int = Field(default=4000, ge=1, description="longer questions are refused")
    max_answer_chars: int = Field(default=3000, ge=50, description="longer answers are cut at a sentence")
    mask_names: bool = Field(default=True, description="mask personal names in the question, not only regex PII")
    block_injection: bool = True
    scan_sources: bool = Field(default=True, description="drop retrieved documents that carry injection text")
    allowed_topics: list[str] = Field(
        default_factory=list,
        description="when set, a question must mention one of these terms or it is refused as out of scope",
    )
    sensitive_topics: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_TOPICS),
        description="questions mentioning these terms go straight to a person (level 3)",
    )
    detect_human_request: bool = Field(default=True, description="escalate when the customer asks for a person")


class RetrievalConfig(BaseModel):
    """How many documents reach the model and how relevant they must be."""

    k: int = Field(default=4, ge=1, le=20)
    min_relevance: float = Field(default=0.15, ge=0.0, le=1.0, description="hits below this are ignored")
    max_source_chars: int = Field(default=1200, ge=50, description="each document is cut to this size in the prompt")


class CitationConfig(BaseModel):
    """Citation check against the retrieved documents."""

    required: bool = Field(default=True, description="an answer without any valid citation fails the check")
    min_support: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="share of a cited sentence's content words that must appear in the cited documents",
    )
    max_uncited_share: float = Field(
        default=0.5, ge=0.0, le=1.0, description="share of content sentences allowed without a citation"
    )
    min_sentence_tokens: int = Field(default=3, ge=1, description="shorter sentences are not checked")


class LevelConfig(BaseModel):
    """Score thresholds of the three service levels.

    Level 1: the answer goes to the customer automatically.
    Level 2: the answer is a draft that a support agent reviews before sending.
    Level 3: the case is escalated to a person; any answer is kept only as a draft.
    """

    level1_min_score: float = Field(default=0.8, ge=0.0, le=1.0)
    level2_min_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> LevelConfig:
        if self.level2_min_score > self.level1_min_score:
            raise ValueError("level2_min_score must not exceed level1_min_score")
        return self


class CacheConfig(BaseModel):
    """Semantic cache of level 1 answers."""

    enabled: bool = True
    threshold: float = Field(default=0.92, gt=0.0, le=1.0, description="cosine similarity needed for a hit")
    ttl_seconds: int = Field(default=7 * 24 * 3600, ge=1)
    max_entries: int = Field(default=1000, ge=1)


class AgentMessages(BaseModel):
    """Fixed replies sent when the agent does not answer by itself."""

    refuse_out_of_scope: str = (
        "I can only help with questions about our products and services. Please rephrase your question."
    )
    refuse_injection: str = "I cannot follow that request. Please describe the problem you need help with."
    refuse_too_long: str = "Your message is too long for me to process. Please send a shorter description."
    escalate: str = "I have passed your request to our support team. A person will reply to you shortly."


class AgentConfig(BaseModel):
    """All support agent parameters."""

    guards: GuardConfig = Field(default_factory=GuardConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    citations: CitationConfig = Field(default_factory=CitationConfig)
    levels: LevelConfig = Field(default_factory=LevelConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    messages: AgentMessages = Field(default_factory=AgentMessages)
    max_retries: int = Field(default=2, ge=0, le=5, description="extra generations after a failed check")
    generator_tier: str = "medium"
    judge_tier: str = "small"
