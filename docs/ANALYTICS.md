# Support analytics

`support_engine.analytics` runs after ingestion (and, for ROI, next to the support
agent). It takes the masked tickets from `IngestionPipeline` and optional agent
outcomes, and returns one `AnalyticsReport`:

- `sentiment`: customer sentiment per ticket and in aggregate, per language;
- `trends` and `alerts`: topics that grow faster than their own baseline;
- `risk`: open tickets ranked by churn, escalation and SLA risk;
- `quality`: resolution, reopen, response time, SLA, CSAT and sentiment per group;
- `root_causes`: features over-represented in negative, high risk and trending tickets;
- `roi`: deflection, hours saved, model cost and ROI of the agent.

```python
from support_engine.analytics import AnalyticsConfig, AnalyticsPipeline, TrendConfig
from support_engine.ingestion import IngestionPipeline

ingested = IngestionPipeline().process(tickets)
config = AnalyticsConfig(trends=TrendConfig(window_hours=24, baseline_windows=7))
report = AnalyticsPipeline(config).run(ingested.tickets, agent_results)
for alert in report.alerts:
    print(alert.keywords, alert.series, alert.status)
for risk in report.high_risk():
    print(risk.ticket_id, risk.score, risk.reasons)
```

Everything runs offline and is deterministic. The only model call is the optional
`LLMSentiment` backend. `now` defaults to the latest timestamp in the tickets, so the
same input gives the same report; pass `now=datetime.now(UTC)` for live use.

## Sentiment

`LexiconSentiment` replaces TextBlob, which only understood English. It scores
English, Portuguese and Spanish with one folded word list (accents and case are
ignored) and handles:

- negation within `negation_window` words: "not good", "nao e bom", "no es bueno"
  (flipped and halved);
- phrases: "not working", "nao funciona", "no funciona", "waste of time", "muchas gracias";
- intensifiers ("very", "muito", "muy") add `intensifier` (0.5); diminishers ("slightly",
  "pouco", "poco") halve the weight;
- contrast ("but", "mas", "pero"): words before weigh 0.5, words after 1.5;
- up to three exclamation marks and three words in capitals add 10% each.

The sum `s` becomes a score with `s / sqrt(s^2 + normalisation)` (15 by default).
Scores at or below `negative_threshold` (-0.2) are negative, at or above
`positive_threshold` (0.2) positive. Add domain words with
`LexiconSentiment(extra_lexicon={"gambiarra": -2}, extra_phrases={("deu", "certo"): 2})`.

A ticket's sentiment reads the public messages of `SentimentConfig.roles` (default
`customer`, the subject and description when there are none). Later messages weigh
more (the last weighs `recency_weight`, 1.5, against 1 for the first), and `delta`
(last minus first) shows whether the customer is getting worse.

`LLMSentiment` asks a model tier through `nexcoder_core.call_structured` (never a
hardcoded vendor model) and falls back to the lexicon when the call fails. Select it
with `AnalyticsConfig(sentiment_backend="llm", llm_tier="small")` or pass any object
with `analyze(text, language)` as `AnalyticsPipeline(sentiment=...)`.

## Trends

`TrendDetector` groups tickets into topics by token Jaccard (`similarity`, 0.3) of
subject, description, tags and product, and counts each topic per window of
`window_hours`. The window that contains `now` is compared with the
`baseline_windows` before it:

| Status | Rule | Alert |
| --- | --- | --- |
| `new` | baseline empty, at least `min_count` now | yes |
| `spike` | at least `min_count` now and `z >= z_threshold` (2) or `ratio >= ratio_threshold` (2) | yes |
| `rising` / `falling` | above / below the baseline mean | no |
| `stable` | equal to the baseline mean | no |

`z = (current - mean) / max(std, min_std)` with `min_std` 1, so a flat baseline does
not turn one extra ticket into an alert. The detector is incremental: keep one
instance, call `observe(ticket)` as tickets arrive, and `report(now)` or
`alerts(now)` whenever needed, or pass it to `AnalyticsPipeline.run(..., trends=detector)`.

## Risk radar

Each open ticket (solved ones too with `include_resolved`) gets signals in 0..1:

| Signal | Value | Weight |
| --- | --- | --- |
| `negative_sentiment` | `-score / negative_score` (0.6), clipped | 0.5 |
| `sentiment_drop` | `-delta / sentiment_drop` (0.5), clipped | 0.3 |
| `churn_terms` | 1 when the customer writes a term of `churn_terms` (cancel, refund, procon, reembolso, ...) | 0.6 |
| `reopens` | `reopen_count / max_reopens` (2), clipped | 0.4 |
| `sla_breach` | 1 past the first response or resolution target, rising from half the target | 0.5 |
| `waiting` | hours since the last unanswered customer message / `waiting_hours` (24) | 0.3 |
| `priority` | `priority_levels` (urgent 1, high 0.6) | 0.2 |
| `low_csat` | 1 when CSAT is at or below `low_csat` (0.4) | 0.5 |

`risk = 1 - prod(1 - weight * signal)`: one strong signal raises the risk and weak ones
add up. `high` is 0.6 and above, `medium` 0.35. `reasons` lists the signals at 0.5 or
more, strongest first. SLA targets per priority live in `SLAConfig` (first response:
urgent 1 h, high 4 h, normal 8 h, low 24 h; resolution: 8 h, 24 h, 72 h, 168 h).

## Service quality

`service_quality` reports, overall and per `QualityConfig.group_by` (product by
default; also source, priority, language, status, tag or any `Ticket.extra` key):
resolved rate, reopen rate, mean first response hours, median resolution hours, SLA
breach rate, mean CSAT, mean sentiment, negative share and high risk tickets.

## Root cause

`RootCauseAnalyzer` compares a target group of tickets with a baseline. Features are
content words, adjacent word pairs inside one sentence, ticket fields (`tag:`,
`product:`, `priority:`, `source:`, `language:`), chosen `Ticket.extra` keys
(`environment:`, `version:`, `channel:`, `plan:`) and change mentions (`change:update`
for "after the update", "depois da atualizacao", "desde la actualizacion"; `version:2.3`
for "since version 2.3" or "v2.3"). For each feature in at least `min_support` target
tickets:

```
share = (count + s) / (n + 2s)          s = smoothing (1)
lift  = target share / baseline share   kept when lift >= min_lift (1.5)
score = lift * log(1 + support)
```

Features found in exactly the same tickets with the same lift are merged: the most
specific one (change, then field, then word pair, then word) stays and the others go to
`related`. The pipeline runs it for negative or high risk tickets and for the tickets of
each trend alert. Call `analyze(target, baseline)` or `analyze_where(tickets, predicate)`
for any other group.

## Deflection and ROI

`compute_roi` takes agent outcomes: `AgentResult` objects, `Outcome` models or dicts
with `action`, `level`, `cached` and `attempts`.

| Outcome | Counts as | Human time saved |
| --- | --- | --- |
| level 1 answer (cache hits too with `deflect_cached`) | deflected, times `confirmed_rate` | `handle_minutes` (12) |
| level 2 draft | assisted | `handle_minutes - review_minutes` (12 - 4) |
| escalation, refusal | no saving | 0 |

```
gross savings = hours saved * agent_hourly_cost            (30)
llm cost      = attempts * llm_calls_per_attempt * llm_cost_per_call   (2 calls, 0.002; cache hits are free)
net savings   = gross savings - llm cost - fixed_cost
roi           = net savings / (llm cost + fixed_cost)      (None when nothing was spent)
```

`baseline_cost` is what people would cost for every question that was not refused.
Money values share `ROIConfig.currency`.

## Command line

```bash
support-engine analyze --input tickets.json --agent-results results.jsonl --report-out analytics.json \
  --window-hours 24 --baseline-windows 7 --group-by product --currency EUR --hourly-cost 28
```

`--agent-results` reads JSON Lines or a JSON list of outcomes. The command ingests
(and masks) the tickets first and prints one summary line.

## Privacy

The pipeline masks tickets again before analysis (`mask_pii`, on by default, a no-op
after ingestion), including personal names and `Ticket.extra` values. Reports hold
numbers, ticket ids, folded tokens and the configured churn terms, never message text:
masked placeholders such as `[EMAIL_1]` are dropped before tokenising. Logs carry counts
and error types only. Grouping by an `extra` key that holds personal data would expose
it as a group name; choose grouping keys accordingly.
