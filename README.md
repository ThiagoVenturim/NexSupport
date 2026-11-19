# Support Intelligence Engine

Support intelligence on top of help desk tickets. It ingests tickets from any
platform into one common model, builds a curated hybrid RAG knowledge base with
a copilot for human agents, and produces operational analytics: sentiment,
trends with alerts, risk radar, quality, root cause, deflection and ROI.

Models (open source or paid), Langfuse tracing and retrieval primitives come
from `nexcoder-core` (the `packages/core` package of the nexcoder monorepo).

Status: under active development. All sample data in this repository is synthetic.

## Connectors

Every ticket source normalizes records into `support_engine.Ticket`:

- `ZendeskConnector`: Zendesk Support API v2 (`/api/v2/tickets.json` and comments).
- `FreshdeskConnector`: Freshdesk API v2 (`/api/v2/tickets` and conversations).
- `JiraSMConnector`: Jira Service Management / Jira Cloud REST API v3 (`/rest/api/3/search` with Atlassian Document Format extraction).
- `IntercomConnector`: Intercom Conversations REST API (`/conversations` and `conversation_parts`).
- `CSVConnector` and `JSONConnector` (`load_tickets_csv`, `load_tickets_json`, `load_tickets_file`): local CSV, JSON and JSONL files guarded by `nexcoder_core.resolve_inside`.

```python
from support_engine import ZendeskConnector, load_tickets_csv

connector = ZendeskConnector(base_url="https://acme.zendesk.com")
ticket = connector.to_ticket(
    {
        "id": 1042,
        "subject": "Cannot export monthly invoice",
        "description": "Clicking Export PDF returns HTTP 500.",
        "status": "solved",
        "priority": "high",
        "created_at": "2026-02-10T14:00:00Z",
        "satisfaction_rating": {"score": "good"},
    }
)

csv_tickets = load_tickets_csv("tests/fixtures/tickets_sample.csv")
```

## Storage and Pluggable Databases

Support Intelligence Engine includes a database-agnostic storage layer built on
SQLAlchemy Core with parameterized queries:

- **Default:** SQLite (`sqlite:///support_engine.db`) for zero configuration.
- **Custom database:** point `DB_URL` in `.env` or the environment to PostgreSQL (`.[postgres]`), MySQL (`.[mysql]`), or SQL Server ODBC (`.[mssql]`).
- **PII masking:** on by default before writing (`mask_pii=False` to opt out); redacts emails, CPF, CNPJ, phone numbers, credit cards, and secret keys.

```python
from support_engine import TicketStore, mask_ticket

with TicketStore() as store:
    store.save_ticket(ticket)  # masked by default
    saved = store.get_ticket("1042")
```

See [docs/DATABASE.md](docs/DATABASE.md) for full configuration details.

## Ingestion Quality

`support_engine.ingestion` cleans threads (HTML, quoted replies, signatures,
footers, disclaimers), masks PII including personal names, drops repeated
interactions, and chunks each ticket per interaction plus a case summary, with
MinHash near-duplicate removal across the corpus. All steps are configurable and
run offline.

```python
from support_engine.ingestion import ChunkingConfig, IngestionConfig, IngestionPipeline

config = IngestionConfig(chunking=ChunkingConfig(max_chars=800, overlap_chars=100))
result = IngestionPipeline(config).process(tickets)
print(result.stats.pii_masked, len(result.chunks))
```

See [docs/INGESTION.md](docs/INGESTION.md) for every option and the privacy defaults.

## Answer Curation

`support_engine.curation` gives every ticket answer a resolution score (solved
status, reopens, CSAT, an optional LLM judge and human ratings), writes it into
the chunk metadata so retrieval can prefer proven answers, turns human
corrections into new index entries that replace the old answer, and suggests a
knowledge base article when the same unanswered question keeps coming back.
Quality metrics summarise each run. Weights and thresholds are configurable.

```python
from support_engine.curation import CurationPipeline, Feedback

feedback = [Feedback.vote("T-101", helpful=False, correction="Reset the token, then sign in again.")]
result = CurationPipeline().curate(ingested.tickets, ingested.chunks, feedback)
print(result.metrics.mean_score, len(result.suggestions))
```

See [docs/CURATION.md](docs/CURATION.md) for the score formula and every option.

## Support Agent

`support_engine.agent` answers customer questions from the curated knowledge
base with a LangGraph flow: input guards (PII masking, prompt injection, out of
scope and size limits), a semantic cache, answers that cite the retrieved
documents, a citation check, a judge with explanatory retries and output guards.
Each result lands on one of three levels: 1 answers the customer, 2 is a draft
for a support agent to review, 3 escalates to a person.

```python
from support_engine.agent import SupportAgent

agent = SupportAgent.from_chunks(curated.chunks, curation=curation)
result = agent.answer("The invoice export fails with error 500")
print(result.level, result.action, [c.ticket_id for c in result.citations])
```

See [docs/AGENT.md](docs/AGENT.md) for the flow, the levels and every option.

## Support Analytics

`support_engine.analytics` turns masked tickets and agent outcomes into one
report: multilingual sentiment (English, Portuguese, Spanish, replacing TextBlob,
with an optional LLM backend), live trends that alert when a topic grows past its
baseline, a risk radar for churn, escalation and SLA breaches, service quality per
product, root cause candidates for negative and trending tickets, and deflection
with ROI for the support agent. Every threshold, weight and cost is configurable.

```python
from support_engine.analytics import AnalyticsPipeline

report = AnalyticsPipeline().run(ingested.tickets, agent_results)
print(report.sentiment.negative_share, [a.keywords for a in report.alerts])
print([r.ticket_id for r in report.high_risk()], report.roi.deflection_rate)
```

See [docs/ANALYTICS.md](docs/ANALYTICS.md) for the formulas and every option.

## Copilot, REST API and MCP

`support_engine.copilot` assists the human agent on an open ticket: a suggested
reply grounded in the support agent (cited curated answers, judge and levels), a
short summary, the next step (send, review, escalate or handle manually, with a
priority from the risk radar) and tone advice from the customer sentiment. The
same operations are served over a REST API and an MCP server (stdio, or HTTP at
`/mcp`). The API needs an access token from `SUPPORT_ENGINE_API_TOKEN` or the
`.env` file and refuses to start without one.

```python
from support_engine.copilot import Copilot

copilot = Copilot.from_tickets(history)
s = copilot.suggest(open_ticket)
print(s.reply, s.next_step.action, s.tone.tone, s.summary.text)
```

See [docs/COPILOT.md](docs/COPILOT.md) for the endpoints, the MCP tools and the security rules.

## Integration with other projects (NexLink)

`support_engine.nexlink` lets the other projects of the portfolio call the Support
Engine with one message envelope (sender, target, action, parameters, correlation
id): NexHub can trigger an ingestion (`support.ingest`), NexCoder can read support
metrics (`support.metrics`), and any client with the scope can ask for a suggested
reply (`support.suggest_reply`). The Support Engine in turn asks the nexdash
(askdash) report engine for the `support_metrics` dashboard. Each calling project has
its own token and scope list, read from the environment or `.env`.

```bash
python examples/nexlink/nexcoder_reads_metrics.py   # offline: NexCoder -> Support Engine -> nexdash
```

See [docs/NEXLINK.md](docs/NEXLINK.md) for the envelope, the actions, the scopes and the examples.

## Command-Line Interface

```bash
# Import tickets into the database (PII masked by default; --no-mask-pii to opt out)
support-engine import --input tests/fixtures/tickets_sample.csv

# Clean, mask, deduplicate and chunk tickets; write chunks as JSON Lines
support-engine ingest --input tests/fixtures/tickets_sample.json --chunks-out chunks.jsonl

# Score answers, apply feedback and corrections, report gaps and quality metrics
support-engine curate --input tests/fixtures/tickets_sample.json --feedback feedback.jsonl --report-out report.json

# Sentiment, trends, risk radar, quality, root cause and ROI as one JSON report
support-engine analyze --input tests/fixtures/tickets_sample.json --agent-results results.jsonl --report-out analytics.json

# Copilot REST API and MCP over HTTP (needs the api extra and SUPPORT_ENGINE_API_TOKEN in the env or .env)
support-engine serve --input tests/fixtures/tickets_sample.json

# Copilot MCP server over stdio
support-engine mcp --input tests/fixtures/tickets_sample.json

# List stored tickets
support-engine list --status open

# Show ticket details and comments
support-engine show --id T-101

# Show statistics breakdown
support-engine stats
```

## Containers and Multiplatform Scripts

- **Docker Compose:** `docker compose up` runs the app with local storage; `docker compose --profile postgres up` spins up a PostgreSQL container alongside it.
- **Kubernetes:** example manifests in `k8s/` (`deployment.yaml`, `service.yaml`, `configmap.yaml`, `secret.yaml`).
- **Scripts:** `.\scripts\run.ps1` (Windows PowerShell) and `./scripts/run.sh` (Linux / macOS Bash).

## Layout

- `src/support_engine/`: the Python package.
- `tests/`: tests and synthetic fixtures. They run offline.
- `k8s/`: Kubernetes manifests.
- `scripts/`: multiplatform execution scripts.
- `docs/`, `examples/`: guides and runnable examples.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e path/to/nexcoder/packages/core
.venv/bin/pip install -e ".[dev]"
git config core.hooksPath .githooks
.venv/bin/python -m pytest
.venv/bin/ruff check .
python scripts/check_no_emoji.py
```

## Origin

Evolved from the support intelligence architecture initially built inside the
ApoIA platform and lochome, rewritten as a clean, platform-agnostic open-source
package with synthetic fixtures.

## License

MIT. See `LICENSE`.
