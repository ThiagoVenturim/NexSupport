# Support metrics on askdash (NexDash)

`support_engine.metrics` connects the Support Engine analytics to the askdash
(NexDash) report engine. NexSupport does not build dashboards of its own: it writes
two tables into the database NexDash reads, and any client (NexSupport itself,
NexLink, an MCP client, a browser through a proxy) asks NexDash for the report over
REST or MCP. The request names a report model and passes period, team and project;
it never carries SQL. See the NexDash guide `docs/REPORT_ENGINE.md` for the engine
contract (operations, scopes, errors).

```
help desk -> connectors -> ingestion (clean, mask) -> analytics (sentiment, trends, risk, quality, root cause, ROI)
          -> support_engine.metrics -> support_conversations + support_insights tables
          -> NexDash report engine -> REST /api/reports/... or MCP nexdash_run_report -> HTML dashboard and panel data
```

## Report models

| Model | Owner | Panels |
| --- | --- | --- |
| `support_metrics` | NexDash | Volume, first response time, resolution rate, deflection, CSAT; monthly volume, first response and CSAT; tickets by sentiment; resolution hours per channel |
| `support_insights` | NexSupport (`support_engine.metrics.askdash`) | High risk tickets, negative sentiment share, SLA breach rate, tickets in alerting topics, hours saved; monthly sentiment and risk; tickets by risk level, strongest risk signal, topics of the current trend window, root cause candidates; sentiment and SLA breaches per queue; net savings of the agent per team |

Both accept the NexDash parameters `start`, `end` (inclusive dates), `team` and
`project`. KPIs are compared with the previous period of the same length.

### Where the numbers come from

| Column | Source |
| --- | --- |
| `sentiment`, `sentiment_score` | `AnalyticsReport.ticket_sentiment` (lexicon or LLM backend) |
| `risk_level`, `risk_score`, `risk_reason` | Risk radar; resolved tickets are `not_rated` unless `RiskConfig.include_resolved` |
| `sla_breached` | `sla_breached` with `RiskConfig.sla` at the report time |
| `trend_topic`, `trend_status`, `trend_alert` | Current window of `TrendDetector`; topic = first three keywords |
| `root_cause` | Strongest root cause candidate (by score) whose tickets include this one |
| `deflected_flag`, `hours_saved`, `net_savings`, `llm_cost` | `compute_roi` on the agent outcome of the ticket (fixed cost excluded) |
| `team` | `extra["team"]`, then `extra["queue"]`, then `team_of(product)`, else `unassigned` |
| `queue` | `extra["queue"]`, then `extra["team"]` |
| `channel` | `extra["channel"]`, then the connector name |

`support_conversations` (the NexDash `support_metrics` table) is built by NexDash's
own `support_rows_from_tickets` from text-free ticket records: the analytics
sentiment and the agent deflection are passed in `extra`. Change the key names with
`build_support_metrics(..., team_key=, queue_key=, channel_key=)`.

Agent outcomes tie to tickets with a `ticket_id` field (or a mapping
`ticket_id -> outcome`). Outcomes without one still count in `AnalyticsReport.roi`
but not in the per-ticket rows.

## Use

```python
from nexdash.executor import get_executor
from support_engine.metrics import build_support_metrics
from support_engine.metrics.askdash import build_engine, load_into_nexdash

metrics = build_support_metrics(tickets, agent_results)   # ingestion + analytics
executor = get_executor("postgresql+psycopg://user@db/analytics")   # the database NexDash reads
load_into_nexdash(executor, metrics)                       # rebuilds both tables
report = build_engine(executor).run(
    {"model": "support_insights", "parameters": {"start": "2026-04-01", "end": "2026-06-30", "team": "tier1"}}
)
html = report["dashboard"]["html"]
```

The NexDash server must know the `support_insights` model. Start it through the
Support Engine, which registers the model on the NexDash engine and keeps the NexDash
authentication (`NEXDASH_API_TOKEN`, `NEXDASH_CLIENT_<NAME>_TOKEN` and `_SCOPES`):

```bash
# load tickets and serve REST and MCP over HTTP on 127.0.0.1:8000
support-engine metrics-server --input tickets.json --agent-results results.jsonl --nexdash-db sqlite:///nexdash.db

# MCP over stdio
support-engine metrics-server --mcp --nexdash-db sqlite:///nexdash.db

# load the tables only, and render the dashboards locally
support-engine metrics --input tickets.json --agent-results results.jsonl \
  --nexdash-db sqlite:///nexdash.db --dashboard-out dashboards/ --start 2026-04-01 --end 2026-06-30 --team tier1
```

A stock `nexdash --report-models` server pointed at the same database serves
`support_metrics` only.

## Client

```python
from support_engine.metrics import NexDashClient

with NexDashClient.from_env(transport="rest") as nexdash:   # or transport="mcp"
    nexdash.list_models()
    report = nexdash.run_report("support_metrics", start="2026-04-01", end="2026-06-30", team="tier1")
    html = nexdash.dashboard("support_insights", locale="pt-BR")
```

`NEXDASH_URL` (default `http://127.0.0.1:8000`) and `NEXDASH_ENGINE_TOKEN` come from
the environment. Give NexSupport its own NexDash client token with only
`reports:list,reports:run`. The client checks the request before sending it (model
name, dates, `end` not before `start`, at most 200 characters for team and project),
refuses URLs with credentials or a query string, and raises `NexDashError` with the
HTTP status and the server's client-safe message.

## Reproducible example

```bash
python examples/metrics_askdash/nexsupport_to_nexdash.py --out dashboards/
```

It builds 126 synthetic tickets (April to June 2026, two teams, three queues, a spike
of export errors on the last day) with agent outcomes, starts NexDash on 127.0.0.1
with a SQLite file in a temporary directory and a fresh client token, and asks for
both dashboards over REST and over MCP. It checks the pages (doctype, one `<main>`,
charts, no `<script>`), that both transports return the same KPIs, that bad
parameters and unknown models are refused, that a request without a token gets 401,
and that the made-up e-mail and phone number of the tickets appear in no answer.
`tests/test_metrics_askdash.py` runs the same round trip.

## Privacy and secrets

- Tickets are ingested (cleaned and masked) before the analytics; the rows hold ids,
  labels, numbers, folded topic tokens and root cause feature names, never subjects,
  descriptions or messages.
- Logs carry counts, status codes and error types only; tokens are never logged and
  `repr(NexDashClient)` shows only whether a token is set.
- `team`, `queue` and `channel` become dashboard labels. Do not map them to `extra`
  keys that hold personal data.
