# Ingestion quality

`support_engine.ingestion` turns raw tickets into cleaned, masked tickets and
retrieval chunks. Everything runs in memory and offline; nothing is sent to a
model or written anywhere unless you do it with the result.

```python
from support_engine.ingestion import IngestionPipeline

result = IngestionPipeline().process(tickets)
result.tickets  # cleaned and masked copies, safe to store
result.chunks   # interaction chunks plus one summary chunk per ticket
result.stats    # counters only: tickets, dropped interactions, duplicates, masked values per kind
```

## Steps

For every ticket, in this order:

1. **Cleanup** (`CleanupConfig`): HTML to text, Unicode NFC, zero-width and
   control characters removed, whitespace collapsed. Quoted earlier replies
   ("On ... wrote:", "Em ... escreveu:", Outlook "From:/Sent:" blocks, `>` lines),
   signatures (`--` delimiter or a sign-off such as "Best regards" or
   "Atenciosamente" followed by at most `max_signature_lines` short lines),
   mobile footers and legal disclaimers are removed. Each step can be switched
   off, and every pattern list can be replaced.
2. **PII masking** (`PrivacyConfig`): e-mail, CPF, CNPJ, phone, card, IPv4 and
   API keys by regex, plus personal names. Names come from context rules
   (greetings, "my name is", honorifics, sign-offs), from structured fields in
   `Ticket.extra` (`name_fields`, for example `requester_name`) and from any
   `name_detectors` you pass, such as an NER model wrapped with
   `build_ner_detector(nlp)`. A fresh `Anonymizer` is used per ticket, so the
   mapping between real values and placeholders never spans tickets and is
   dropped after the ticket is processed.
3. **Interaction cleanup**: interactions left empty by cleanup are dropped, and
   interactions that repeat an earlier one in the same ticket (or the
   description) are removed with MinHash.
4. **Chunking** (`ChunkingConfig`): one chunk set per interaction (description
   first, then public comments in time order; internal notes only with
   `include_private=True`). Long interactions are split on paragraph and
   sentence boundaries into pieces of at most `max_chars`, with
   `overlap_chars` of word-aligned overlap. A case summary chunk holds subject,
   status, product, tags, the problem and the last agent answer. Pass
   `summarizer=` to replace it, for example with a model call; it receives the
   masked ticket only.
5. **Near-duplicate removal** (`DedupConfig`, `dedup_chunks`): interaction chunks
   are compared with MinHash signatures (`num_perm`, `bands`, `shingle_size`
   word shingles) and an estimated Jaccard `threshold` (default 0.85). Scope is
   `"corpus"` (default), `"ticket"` or `"none"`. Placeholders such as
   `[NAME_1]` and `[NAME_7]` count as the same token, so the same macro answer
   sent to different customers is detected. Summary chunks are always kept.

## Privacy defaults (LGPD)

- Masking is on by default in the pipeline, in `TicketStore.save_ticket` and in
  `support-engine import`.
- Turning it off in the pipeline needs both `mask_pii=False` and
  `allow_unmasked=True`; in the store it needs `mask_pii=False`; in the CLI it
  needs `--no-mask-pii`.
- Log lines from the pipeline carry counts only, never ticket text or detected
  values.
- Regex and context rules can miss names written without context. Add an NER
  detector or `known_names` when the data needs it, and review a sample of the
  output before sharing it.

## Command line

```bash
support-engine ingest --input tests/fixtures/tickets_sample.json --chunks-out chunks.jsonl
support-engine ingest --input tickets.csv --max-chars 800 --overlap-chars 100 --dedup-threshold 0.9 --no-store
```

`ingest` stores the masked tickets (unless `--no-store`), writes chunks as JSON
Lines when `--chunks-out` is given and prints a short report with counts.
