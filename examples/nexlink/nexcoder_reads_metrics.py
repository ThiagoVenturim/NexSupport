"""NexCoder reads support metrics over NexLink; the Support Engine asks nexdash for the dashboard.

The full chain runs offline in this process:

    nexcoder --NexLink support.metrics--> Support Engine --report engine--> nexdash

    python examples/nexlink/nexcoder_reads_metrics.py
    python examples/nexlink/nexcoder_reads_metrics.py --stub    # nexdash stub instead of the engine

NexCoder can use the numbers (answer quality, gaps, resolution) to choose what to improve next.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from demo_setup import Demo, build_demo


def run(demo: Demo) -> dict[str, Any]:
    nexcoder = demo.clients["nexcoder"]
    reply = nexcoder.request(
        "nexsupport",
        "support.metrics",
        {"dashboard": True, "start": "2026-04-01", "end": "2026-06-30", "locale": "en"},
        context={"pipeline": "improvement-review"},
    )
    if not reply.ok:
        return {"status": reply.status, "error": reply.error.model_dump(mode="json") if reply.error else None}
    result = reply.result or {}
    report = result.get("report", {})
    quality = result.get("quality", {})
    return {
        "status": reply.status,
        "correlation_id": reply.correlation_id,
        "tickets": result.get("tickets"),
        "by_status": result.get("by_status"),
        "mean_answer_score": quality.get("mean_score"),
        "gaps": quality.get("gaps"),
        "report_title": report.get("title"),
        "report_panels": sorted(report.get("data", {}).get("panels", {})),
        "dashboard_chars": len(report.get("dashboard", {}).get("html", "")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stub", action="store_true", help="use the nexdash stub even when nexdash is installed")
    args = parser.parse_args(argv)
    demo = build_demo(prefer_real_nexdash=not args.stub)
    print(json.dumps({"nexdash": demo.nexdash_label, **run(demo)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
