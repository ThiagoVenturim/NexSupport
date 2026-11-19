"""Support agent: a LangGraph flow with guards, semantic cache, cited answers, a judge and levels 1/2/3.

Typical use, after ingestion and curation::

    from support_engine.agent import SupportAgent
    from support_engine.curation import CurationPipeline
    from support_engine.ingestion import IngestionPipeline

    ingested = IngestionPipeline().process(tickets)
    curation = CurationPipeline()
    curated = curation.curate(ingested.tickets, ingested.chunks)
    agent = SupportAgent.from_chunks(curated.chunks, curation=curation)
    result = agent.answer("The invoice export fails with error 500")
    result.action     # answer, review, escalate or refuse
    result.level      # 1, 2, 3 (None when refused)
    result.citations  # retrieved documents the answer cites
"""

from support_engine.agent.cache import CacheEntry, CacheHit, Embedder, HashingEmbedder, SemanticCache, cosine
from support_engine.agent.citations import (
    Citation,
    CitationReport,
    check_citations,
    citations_for,
    labels_in,
    split_sentences,
)
from support_engine.agent.config import (
    DEFAULT_SENSITIVE_TOPICS,
    AgentConfig,
    AgentMessages,
    CacheConfig,
    CitationConfig,
    GuardConfig,
    LevelConfig,
    RetrievalConfig,
)
from support_engine.agent.graph import Action, AgentResult, AgentState, SupportAgent
from support_engine.agent.guards import (
    HUMAN_REQUEST_PATTERNS,
    INJECTION_PATTERNS,
    Guard,
    InjectionDetector,
    InputVerdict,
    OutputVerdict,
    fold,
    truncate_sentences,
)
from support_engine.agent.llm import (
    AnswerGenerator,
    AnswerJudge,
    JudgeResult,
    LLMAnswerGenerator,
    LLMAnswerJudge,
    format_sources,
)
from support_engine.agent.retrieval import LexicalRetriever, Retriever, Source, tokens

__all__ = [
    "DEFAULT_SENSITIVE_TOPICS",
    "HUMAN_REQUEST_PATTERNS",
    "INJECTION_PATTERNS",
    "Action",
    "AgentConfig",
    "AgentMessages",
    "AgentResult",
    "AgentState",
    "AnswerGenerator",
    "AnswerJudge",
    "CacheConfig",
    "CacheEntry",
    "CacheHit",
    "Citation",
    "CitationConfig",
    "CitationReport",
    "Embedder",
    "Guard",
    "GuardConfig",
    "HashingEmbedder",
    "InjectionDetector",
    "InputVerdict",
    "JudgeResult",
    "LLMAnswerGenerator",
    "LLMAnswerJudge",
    "LevelConfig",
    "LexicalRetriever",
    "OutputVerdict",
    "RetrievalConfig",
    "Retriever",
    "SemanticCache",
    "SupportAgent",
    "Source",
    "check_citations",
    "citations_for",
    "cosine",
    "fold",
    "format_sources",
    "labels_in",
    "split_sentences",
    "tokens",
    "truncate_sentences",
]
