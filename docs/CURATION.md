# Answer curation

`support_engine.curation` runs after ingestion. It takes the masked tickets and
chunks from `IngestionPipeline` plus optional human feedback, and returns:

- the chunks with a resolution score in their metadata, ready to index;
- new `correction` chunks written by people, which replace the answers they fix;
- article suggestions for questions that keep coming back without a good answer;
- quality metrics (numbers only).

```python
from support_engine.curation import CurationConfig, CurationPipeline, Feedback
from support_engine.ingestion import IngestionPipeline

ingested = IngestionPipeline().process(tickets)
feedback = [
    Feedback.vote("T-101", helpful=True),
    Feedback.stars("T-102", 2, query="export stops at night"),
    Feedback.vote("T-103", helpful=False, correction="Restart the export worker, then rerun the job."),
]
result = CurationPipeline(CurationConfig(min_score=0.3)).curate(ingested.tickets, ingested.chunks, feedback)
```

## Resolution score

The answer of a ticket is its last public reply by a role in
`ScoringConfig.answer_roles` (default `agent`). Each signal is a number in 0..1:

| Signal | Value | Default weight |
| --- | --- | --- |
| `solved` | 1 for solved or closed, `pending_value` (0.3) for pending, `unresolved_value` (0) otherwise | 0.25 |
| `no_reopen` | `1 - reopen_penalty * reopen_count`, floored at 0 (`reopen_penalty` 0.5) | 0.15 |
| `csat` | `Ticket.csat`, when present | 0.20 |
| `judge` | verdict of the LLM judge, when one is configured and answers | 0.15 |
| `feedback` | `(prior_mean * prior_weight + sum(ratings)) / (prior_weight + n)` | 0.25 |

The score is the weighted mean of the signals that are present. Missing signals
are left out and the remaining weights renormalised. `confidence` is the share of
the total weight carried by the present signals, so a score built from the status
alone is marked as weak. A ticket with no agent answer scores 0 unless
`require_answer=False`.

The feedback prior (`feedback_prior_mean` 0.5, `feedback_prior_weight` 2) keeps
one vote from swinging the score; raise the weight to require more votes.

Every chunk of a ticket gets `resolution_score`, `score_confidence` and
`curation_priority` in its metadata. A chunk rated on its own (`Feedback.chunk_id`)
blends that rating in, with the feedback weight's share of the total.

### LLM judge

A judge is any callable `(problem, answer) -> float | None`. `LLMJudge` uses
`nexcoder_core.call_structured` with a model tier (`small` by default) and returns
None when the model fails, which only removes that signal. The judge sees the
masked ticket text from ingestion.

```python
from support_engine.curation import CurationPipeline, LLMJudge

pipeline = CurationPipeline(judge=LLMJudge(tier="small"))
```

Tests pass a fake model with an `ainvoke` method and stay offline.

## Feedback loop

`Feedback` holds a rating in 0..1 (`Feedback.vote` for thumbs, `Feedback.stars`
for star scales), an optional `correction`, the `query` the answer was shown for,
and the reviewer role. `correction` and `query` are masked with the default PII
rules when the object is built.

With `FeedbackConfig.apply_corrections` (default on), each correction becomes a
`correction` chunk with `resolution_score = correction_score` (1.0) and a
`corrects` list. The chunks it replaces are the rated chunk (`chunk_id`) or, when
the feedback rates the whole ticket, its summary and the chunks of the answer
reply. `superseded_policy` decides what happens to them:

- `demote` (default): keep them, add `superseded_by`, multiply `curation_priority` by `demote_factor` (0.5);
- `drop`: remove them from the output;
- `keep`: only add `superseded_by`.

`CurationPipeline.rerank(hits)` blends retrieval relevance with curation
priority (`relevance_weight` 0.7) so curated answers rise. `min_score` drops
chunks whose priority is below it; corrections stay unless `correction_score` is
set lower.

`FeedbackStore` keeps feedback in the same database as the tickets (`DB_URL`,
SQLite by default), table `answer_feedback`, with parameterized SQL.

## Knowledge gaps and article suggestions

`GapDetector` records a question as a gap when:

- `CurationPipeline.observe_query(query, hits)` finds no hit with priority at or
  above `gap_threshold` (0.4), or no hits at all;
- a person rated an answer below `helpful_threshold` and wrote no correction.

Questions are masked, reduced to content words (placeholders and stopwords
removed), and joined to the closest cluster when the token Jaccard is at least
`similarity` (0.5). A cluster with `min_occurrences` (3) questions becomes an
`ArticleSuggestion` with a title, keywords, up to `max_samples` masked sample
questions and the related ticket ids.

## Quality metrics

`QualityMetrics`: answers and answered tickets, mean and median score, mean
confidence, share at or above `high_score` (0.7) and below `low_score` (0.4), a
five-bucket score distribution, coverage of each signal, feedback count, helpful
rate, corrections, gaps, gap clusters and suggested articles.

## Privacy

- Curation runs on ingestion output, which is masked by default.
- Feedback text is masked on construction; gap questions are masked again before they are kept.
- Log lines and metrics carry counts only. The judge logs only the error type when it fails.
- The CLI reports invalid feedback by error type, without echoing the input.

## Command line

```bash
support-engine curate --input tickets.json --feedback feedback.jsonl \
  --chunks-out curated.jsonl --report-out report.json --min-score 0.3 --superseded demote
support-engine curate --input tickets.json --feedback feedback.jsonl --save-feedback
support-engine curate --input tickets.json --with-stored-feedback --gap-min-occurrences 2
```

The feedback file is JSON Lines or a JSON list of `Feedback` objects, for example
`{"ticket_id": "T-101", "rating": 0, "correction": "Restart the export worker."}`.
`--report-out` writes metrics, run counters, per-ticket scores and suggestions.
