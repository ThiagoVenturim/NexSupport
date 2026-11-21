"""Support copilot: suggested replies, summaries, next steps and tone, served over REST and MCP.

Typical use::

    from support_engine.copilot import Copilot

    copilot = Copilot.from_tickets(history, feedback)
    suggestion = copilot.suggest(open_ticket)
    suggestion.reply       # grounded reply for the human agent (None when the case must escalate)
    suggestion.citations   # tickets the reply relies on
    suggestion.summary     # problem, latest messages and status
    suggestion.next_step   # send_reply, review_reply, escalate or handle_manually, with a priority
    suggestion.tone        # empathetic, neutral or friendly, from the customer sentiment

REST: ``support_engine.copilot.api.create_app(copilot)`` (needs the ``api`` extra).
MCP: ``CopilotMCPServer(CopilotService(copilot)).run_stdio()``.
"""

from support_engine.copilot.assist import (
    STEP_TEXT,
    Copilot,
    CopilotSuggestion,
    ExtractiveSummarizer,
    LLMSummarizer,
    NextStep,
    Summarizer,
    SummaryVerdict,
    TicketSummary,
    ToneAdvice,
    advise_tone,
    first_sentence,
    lead,
    plan_next_step,
)
from support_engine.copilot.auth import TokenAuth, TokenError, load_token, read_env_file
from support_engine.copilot.config import CopilotConfig, InterfaceConfig, SummaryConfig, ToneConfig
from support_engine.copilot.mcp import SUPPORTED_PROTOCOLS, CopilotMCPServer, Tool
from support_engine.copilot.service import (
    AnswerRequest,
    CopilotService,
    FeedbackRequest,
    FeedbackResponse,
    SuggestRequest,
    SummaryRequest,
    validation_summary,
)

__all__ = [
    "STEP_TEXT",
    "SUPPORTED_PROTOCOLS",
    "AnswerRequest",
    "Copilot",
    "CopilotConfig",
    "CopilotMCPServer",
    "CopilotService",
    "CopilotSuggestion",
    "ExtractiveSummarizer",
    "FeedbackRequest",
    "FeedbackResponse",
    "InterfaceConfig",
    "LLMSummarizer",
    "NextStep",
    "SuggestRequest",
    "Summarizer",
    "SummaryConfig",
    "SummaryRequest",
    "SummaryVerdict",
    "TicketSummary",
    "TokenAuth",
    "TokenError",
    "Tool",
    "ToneAdvice",
    "ToneConfig",
    "advise_tone",
    "first_sentence",
    "lead",
    "load_token",
    "plan_next_step",
    "read_env_file",
    "validation_summary",
]
