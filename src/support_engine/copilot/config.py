"""Settings of the support copilot and of its REST and MCP interfaces."""

from __future__ import annotations

from pydantic import BaseModel, Field

from support_engine.agent import AgentConfig
from support_engine.analytics import RiskConfig, SentimentConfig


class ToneConfig(BaseModel):
    """How the copilot turns customer sentiment into tone advice.

    Openings are fixed sentences chosen by the support team. They carry no facts, so adding
    one in front of a grounded answer does not change what the answer claims.
    """

    empathetic_below: float = Field(default=-0.2, ge=-1.0, le=1.0, description="sentiment score that calls for empathy")
    friendly_above: float = Field(
        default=0.3, ge=-1.0, le=1.0, description="sentiment score that allows a lighter tone"
    )
    openings: dict[str, str] = Field(
        default_factory=lambda: {
            "empathetic": "I am sorry for the trouble this has caused.",
            "neutral": "",
            "friendly": "Thanks for reaching out.",
        }
    )
    guidance: dict[str, str] = Field(
        default_factory=lambda: {
            "empathetic": "Acknowledge the problem once, give clear steps and say when you will follow up.",
            "neutral": "Be direct: give the steps and ask the customer to confirm the result.",
            "friendly": "Keep the reply short and warm; the customer is satisfied so far.",
        }
    )


class SummaryConfig(BaseModel):
    max_chars: int = Field(default=600, ge=80, description="length limit of the summary text")
    sentence_chars: int = Field(default=220, ge=40, description="length limit of each quoted message")


class CopilotConfig(BaseModel):
    """All copilot parameters."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tone: ToneConfig = Field(default_factory=ToneConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    max_question_chars: int = Field(default=4000, ge=100, description="last customer message sent to the agent")
    add_opening: bool = Field(default=True, description="prefix the suggested reply with the tone opening")


class InterfaceConfig(BaseModel):
    """Limits of the REST API and the MCP server."""

    token_env: str = Field(default="SUPPORT_ENGINE_API_TOKEN", description="variable that holds the access token")
    env_file: str | None = Field(default=".env", description="file read when the variable is not set; None skips it")
    min_token_chars: int = Field(default=16, ge=8)
    max_body_bytes: int = Field(default=1_000_000, ge=1_000, description="largest request body accepted")
