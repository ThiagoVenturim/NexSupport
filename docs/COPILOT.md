# Copilot, REST API and MCP

`support_engine.copilot` helps the human agent who works an open ticket. For one ticket it
returns:

| Field | What it holds |
| --- | --- |
| `reply` | Suggested reply: the support agent's grounded answer, with the tone opening in front. `None` when the case escalates or is refused. |
| `answer`, `citations`, `judge_score`, `level`, `action` | The support agent result the reply comes from (see [AGENT.md](AGENT.md)). |
| `summary` | Problem, latest customer message, latest agent reply, status and message count. |
| `next_step` | `send_reply`, `review_reply`, `escalate` or `handle_manually`, a priority (`low`, `normal`, `high`) and the reasons. |
| `tone` | `empathetic`, `neutral` or `friendly`, the sentiment behind it, an opening sentence and writing guidance. |
| `risk` | The risk radar assessment of the ticket (see [ANALYTICS.md](ANALYTICS.md)). |

## How it works

1. The ticket is masked with the default PII rules (`mask_ticket`). Nothing after this step sees
   raw e-mails, phones, documents or names.
2. The latest public customer message (or the problem text when there is none) goes to the
   `SupportAgent`, which retrieves curated answers, writes a cited answer, checks the citations,
   runs the judge and picks a level.
3. Only levels 1 and 2 produce a suggested reply. Ungrounded answers are never suggested.
4. Sentiment (`ticket_sentiment`) sets the tone: a score at or below `empathetic_below`, or a
   customer whose last message turned negative, gets `empathetic`; a score at or above
   `friendly_above` without a drop gets `friendly`.
5. The risk radar sets the priority. A high-risk ticket turns `send_reply` into `review_reply`,
   so a person reads the answer before it reaches a customer who may churn.
6. The summary is extractive by default. `LLMSummarizer` writes it with a model through
   `call_structured`, masks the result again and falls back to the extractive summary on error.

```python
from support_engine.copilot import Copilot, CopilotConfig, LLMSummarizer

copilot = Copilot.from_tickets(history, feedback, summarizer=LLMSummarizer(tier="small"))
suggestion = copilot.suggest(open_ticket)

# The agent sent, edited or discarded the reply: feed curation.
copilot.record_feedback(
    [c.ticket_id for c in suggestion.citations],
    accepted=True,
    correction="Text the agent actually sent",
    query=suggestion.question,
)
```

Feedback ratings: accepted as is 1.0, accepted after an edit 0.5 (the edit becomes a correction
for curation), rejected 0.0. Pass `feedback_store=FeedbackStore(...)` to store it.

## Access token

The REST API and MCP over HTTP need one access token:

1. `SUPPORT_ENGINE_API_TOKEN` in the environment, or
2. the same key in the `.env` file of the working directory (`--env-file` changes the path).

The app refuses to start without a token or with one shorter than 16 characters. The token is
never a default, never in the code and never logged. Clients send it as
`Authorization: Bearer <token>` or `X-API-Key: <token>`; the check runs in constant time.
Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## REST API

Install the extra and start the server (binds to 127.0.0.1 by default):

```bash
pip install -e ".[api]"
support-engine serve --input tickets.json --feedback feedback.jsonl --port 8000
```

| Method and path | Body | Answer |
| --- | --- | --- |
| `GET /health` | | `{"status": "ok"}`, no token needed |
| `POST /v1/suggest` | `{"ticket": Ticket, "namespace": "default", "now": null}` | `CopilotSuggestion` |
| `POST /v1/answer` | `{"question": "...", "namespace": "default"}` | `AgentResult` |
| `POST /v1/summary` | `{"ticket": Ticket}` | `TicketSummary` |
| `POST /v1/feedback` | `{"ticket_ids": ["T-1"], "accepted": true, "correction": null, "query": null}` | ids, rating, stored |
| `POST /mcp` | JSON-RPC 2.0 message or batch | MCP answer (202 for notifications) |

`Ticket` is the common model of `support_engine.models` as JSON. Example:

```bash
curl -s http://127.0.0.1:8000/v1/answer \
  -H "Authorization: Bearer $SUPPORT_ENGINE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "The invoice export fails with error 500"}'
```

Safety rules of the API:

- Every path except `/health` needs the token, including the OpenAPI schema. The interactive docs
  pages are off.
- Validation errors list field locations and error types only; the input is not echoed back.
- Unexpected errors answer `{"detail": "internal_error"}` and log the error type only.
- Request bodies need a `Content-Length` and are limited to `max_body_bytes` (1 MB by default).

## MCP server

Tools: `suggest_reply`, `answer_question`, `summarize_ticket` and `record_feedback`. Their input
schemas are the request models above.

- stdio: `support-engine mcp --input tickets.json`. The MCP client starts the server as a child
  process under its own user. Protocol messages go to stdout, logs to stderr.
- HTTP: `POST /mcp` on the REST server, behind the same token.

Example client configuration for stdio:

```json
{
  "mcpServers": {
    "support-engine": {
      "command": "support-engine",
      "args": ["mcp", "--input", "tickets.json"]
    }
  }
}
```

Supported protocol versions: `2025-06-18`, `2025-03-26` and `2024-11-05`. A failed tool call
answers with `isError: true` and the error type, never with the arguments.

## Models and tests

The support agent asks `nexcoder_core.get_chat_model` for its tiers, and so does
`LLMSummarizer`. The tests pass scripted generators and `FakeChatModel`, use synthetic tickets
and run offline.

## Options

`CopilotConfig` holds `agent` (`AgentConfig`), `sentiment`, `risk`, `tone` (`ToneConfig`:
thresholds, openings and guidance per tone), `summary` (`SummaryConfig`: length limits),
`max_question_chars` and `add_opening`. `InterfaceConfig` holds the token variable name, the
`.env` path, the minimum token length and the body limit.
