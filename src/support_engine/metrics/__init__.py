"""Support metrics on the askdash (NexDash) report engine.

The analytics of the Support Engine (sentiment, trends with alerts, risk radar, service
quality, root cause, deflection and ROI) become two NexDash report models:
``support_metrics`` (shipped with NexDash) and ``support_insights`` (defined in
``support_engine.metrics.askdash``). Typical use::

    from nexdash.executor import get_executor
    from support_engine.metrics import build_support_metrics
    from support_engine.metrics.askdash import build_engine, load_into_nexdash

    metrics = build_support_metrics(tickets, agent_results)   # ingestion + analytics
    executor = get_executor("sqlite:///nexdash.db")            # the database NexDash reads
    load_into_nexdash(executor, metrics)
    report = build_engine(executor).run({"model": "support_insights", "parameters": {"team": "tier1"}})

Remote clients use ``NexDashClient`` over REST or MCP. ``support_engine.metrics.askdash``
needs the ``nexdash`` package; the rows and the client do not.
"""

from support_engine.metrics.client import NexDashClient, NexDashError, report_request
from support_engine.metrics.rows import RISK_LEVELS, SENTIMENTS, SupportMetrics, build_support_metrics, split_outcomes

__all__ = [
    "RISK_LEVELS",
    "SENTIMENTS",
    "NexDashClient",
    "NexDashError",
    "SupportMetrics",
    "build_support_metrics",
    "report_request",
    "split_outcomes",
]
