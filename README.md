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
- **PII masking:** automatic redaction of emails, CPF, CNPJ, phone numbers, credit cards, and secret keys.

```python
from support_engine import TicketStore, mask_ticket

with TicketStore() as store:
    store.save_ticket(ticket, mask_pii=True)
    saved = store.get_ticket("1042")
```

See [docs/DATABASE.md](docs/DATABASE.md) for full configuration details.

## Command-Line Interface

```bash
# Import tickets into the database with PII masking
support-engine import --input tests/fixtures/tickets_sample.csv --mask-pii

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
