# Support agent

`support_engine.agent` answers customer questions from the curated knowledge base
built by ingestion and curation. It is a LangGraph state graph with input and
output guards, a semantic cache, cited answers, a judge with explanatory retries
and three service levels.

```python
from support_engine.agent import SupportAgent
from support_engine.curation import CurationPipeline
from support_engine.ingestion import IngestionPipeline

ingested = IngestionPipeline().process(tickets)
curation = CurationPipeline()
curated = curation.curate(ingested.tickets, ingested.chunks, feedback)

agent = SupportAgent.from_chunks(curated.chunks, curation=curation)
result = agent.answer("The invoice export fails with error 500")
result.action      # "answer", "review", "escalate" or "refuse"
result.level       # 1, 2 or 3; None when refused
result.reply       # message that can go to the customer now (None at level 2)
result.answer      # generated answer, a draft for the team at levels 2 and 3
result.citations   # chunk and ticket ids of the documents the answer cites
```

Use `await agent.aanswer(...)` inside an event loop. Models come from
`nexcoder_core.get_chat_model` (tiers `medium` for the answer and `small` for the
judge, see `AgentConfig.generator_tier` and `judge_tier`); pass `llm=` and
`judge_llm=` to use specific chat models, or `generator=` and `judge=` for custom
objects with `agenerate` and `ajudge`.

## Flow

```
guard_input -> cache_lookup -> retrieve -> generate -> check_citations -> judge -> guard_output -> decide
     |              |             |           ^              |               |
   refuse /       hit: END    no documents:   +--- retry ----+---- retry ----+
   escalate                    escalate
```

1. `guard_input` masks PII (regex rules and personal names) before anything else,
   refuses oversized questions, prompt injection and, when `allowed_topics` is set,
   out of scope questions. A request for a person or a sensitive topic (legal action,
   chargeback, fraud and similar, configurable) escalates at once.
2. `cache_lookup` returns a stored level 1 answer for a question with cosine
   similarity at or above `CacheConfig.threshold` (0.92) in the same namespace.
3. `retrieve` takes the top `k` chunks (4), ignores hits below `min_relevance`
   (0.15) and drops documents that carry injection text. No documents: escalate.
4. `generate` writes an answer that cites documents by label (`[S1]`, `[S1, S2]`).
5. `check_citations` verifies the answer against the retrieved documents (below).
   On failure the problems go back to the generator as instructions.
6. `judge` scores the answer from 0 to 1 against the question and the documents.
   Below the level 1 threshold, its reason goes back to the generator.
7. `guard_output` masks any PII the model wrote, flags injection text in the answer
   and cuts answers longer than `max_answer_chars` at a sentence end.
8. `decide` picks the level. Level 1 answers are stored in the cache.

Retries: at most `max_retries` (2) extra generations per question, shared by the
citation check and the judge.

## Levels

| Level | Action | When |
| --- | --- | --- |
| 1 | `answer`: reply sent to the customer | citations pass, judge score >= `level1_min_score` (0.8), judge says grounded, no PII in the answer |
| 2 | `review`: draft for a support agent | citations pass and score >= `level2_min_score` (0.5), or the judge was unavailable, or the model wrote PII |
| 3 | `escalate`: a person takes over | no documents, citations still failing, score below level 2, unsafe answer, generation or retrieval error, request for a person, sensitive topic |
| - | `refuse` | injection, out of scope or oversized question |

`result.reasons` lists why: `needs_review`, `judge_unavailable`,
`judge_not_grounded`, `pii_in_answer`, `ungrounded`, `low_score`, `no_documents`,
`unsafe_answer`, `generation_failed`, `retrieval_failed`, `requested_human`,
`sensitive_topic`, `injection`, `out_of_scope`, `too_long`.

Questions escalated for `no_documents` or `low_score` are added to the curation
gap detector when the agent was built with `curation=`, so repeated ones become
article suggestions.

## Citation check

The answer passes when:

- it cites at least one retrieved document (`CitationConfig.required`);
- every cited label belongs to a retrieved document;
- each cited sentence has at least `min_support` (0.3) of its content words in the
  documents it cites;
- at most `max_uncited_share` (0.5) of the content sentences have no citation.

Sentences with fewer than `min_sentence_tokens` (3) content words are not checked.
`check_citations(answer, sources)` can be used on its own.

## Retrieval

A retriever is any callable `(query, k) -> list[(Chunk, relevance)]` with
relevance in 0..1. `LexicalRetriever` is the offline default: BM25 over the
chunks, normalised so a chunk holding every query term once scores 1, then
ordered with `CurationPipeline.rerank`, which blends relevance with the curation
priority so human corrections and proven answers come first.

## Privacy and logs

- The raw question never enters the graph state past `guard_input`, the cache,
  the prompts or the logs.
- Log lines carry the action, level, counters and fixed flag names only.
- Failed model calls log the exception type, never the message.

## Configuration

`AgentConfig` groups `guards` (`GuardConfig`), `retrieval` (`RetrievalConfig`),
`citations` (`CitationConfig`), `levels` (`LevelConfig`), `cache` (`CacheConfig`)
and `messages` (`AgentMessages`, the fixed refusal and escalation replies).

Extra injection detectors, for example a classifier, can be added with
`Guard(config, detectors=[callable])`; a detector that raises counts as a hit.
The semantic cache takes any embedder with `embed_query(text)`; the default
`HashingEmbedder` runs offline.
