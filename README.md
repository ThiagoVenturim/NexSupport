# Support Intelligence Engine

Support intelligence on top of help desk tickets. It ingests tickets from any
platform into one common model, builds a curated hybrid RAG knowledge base with
a copilot for human agents, and produces operational analytics: sentiment,
trends with alerts, risk radar, quality, root cause, deflection and ROI.

Models (open source or paid), Langfuse tracing and retrieval primitives come
from `faccoder-core` (the `packages/core` package of the faccoder monorepo).

Status: under active development. All sample data in this repository is synthetic.

## Layout

- `src/support_engine/`: the Python package.
- `tests/`: tests and synthetic fixtures. They run offline.
- `docs/`, `examples/`: guides and runnable examples.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e path/to/faccoder/packages/core
.venv/bin/pip install -e ".[dev]"
git config core.hooksPath .githooks
.venv/bin/python -m pytest
.venv/bin/ruff check .
python scripts/check_no_emoji.py
```

## License

MIT. See `LICENSE`.
