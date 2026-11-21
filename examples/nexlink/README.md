# NexLink examples

Reproducible, offline examples of the Support Engine talking to the other projects of the
portfolio over NexLink. No network, no real tokens: each run generates its tokens in
process, and the tickets are synthetic.

| Example | Who calls whom |
| --- | --- |
| `support_asks_dashboard.py` | The Support Engine asks nexdash (askdash) for the `support_metrics` dashboard |
| `nexhub_triggers_ingest.py` | NexHub asks the Support Engine to ingest tickets (`support.ingest`) |
| `nexcoder_reads_metrics.py` | NexCoder reads support metrics (`support.metrics`); the Support Engine asks nexdash for the dashboard on the way |

Run them from the repository root after `pip install -e ".[dev]"`:

```bash
python examples/nexlink/support_asks_dashboard.py --out dashboards/
python examples/nexlink/nexhub_triggers_ingest.py
python examples/nexlink/nexcoder_reads_metrics.py
```

When the `nexdash` package is installed the examples use its real report engine in process,
on its synthetic data. Without it, or with `--stub`, they use `FakeReportEngine` from
`demo_setup.py`, which answers in the same format. `tests/test_nexlink_examples.py` runs
every example in the test suite.

See [docs/NEXLINK.md](../../docs/NEXLINK.md) for the envelope, the actions and the scopes.
