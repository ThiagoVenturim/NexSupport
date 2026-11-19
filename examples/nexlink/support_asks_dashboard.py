"""The Support Engine asks nexdash (askdash) for the support_metrics dashboard.

Offline by default: the nexdash report engine runs in this process on synthetic data (or a
stub with the same answer format when nexdash is not installed).

    python examples/nexlink/support_asks_dashboard.py
    python examples/nexlink/support_asks_dashboard.py --out dashboards/

Against a running nexdash server, pass its URL; the token comes from NEXDASH_ENGINE_TOKEN
(environment or .env), never from the command line:

    python examples/nexlink/support_asks_dashboard.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from demo_setup import report_transport

from support_engine.nexlink import NexDashClient, ReportError


def run(client: NexDashClient) -> dict[str, Any]:
    models = client.list_models()
    report = client.support_dashboard(start="2026-04-01", end="2026-06-30", locale="en")
    try:
        client.support_dashboard(start="2026-06-30", end="2026-01-01")
        refused = None
    except (ReportError, ValueError) as exc:
        refused = type(exc).__name__
    return {
        "models": models,
        "title": report["title"],
        "panels": sorted(report.get("data", {}).get("panels", {})),
        "errors": report.get("errors", {}),
        "dashboard_html": report.get("dashboard", {}).get("html", ""),
        "invalid_period_refused_by": refused,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="URL of a running nexdash server (default: in-process engine)")
    parser.add_argument("--stub", action="store_true", help="use the nexdash stub even when nexdash is installed")
    parser.add_argument("--out", type=Path, help="directory where the dashboard HTML is written")
    args = parser.parse_args(argv)

    if args.url:
        client, label = NexDashClient.from_env(args.url), f"nexdash at {args.url}"
    else:
        transport, label = report_transport(prefer_real=not args.stub)
        client = NexDashClient(transport)
    result = run(client)

    html = result.pop("dashboard_html")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "support_metrics.html").write_text(html, encoding="utf-8")
    print(json.dumps({"nexdash": label, **result, "dashboard_chars": len(html)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
