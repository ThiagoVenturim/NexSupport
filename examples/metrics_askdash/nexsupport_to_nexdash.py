"""NexSupport -> NexDash: support dashboards from synthetic tickets, offline.

Ingests and analyses synthetic tickets, loads the support tables into a local NexDash
server on 127.0.0.1, and asks it for the support_metrics and support_insights
dashboards over REST and MCP. Needs the support engine and the nexdash package:

    pip install -e <askdash checkout>
    pip install -e .
    python examples/metrics_askdash/nexsupport_to_nexdash.py --out dashboards/

Exit status 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import sys

from support_engine.metrics.demo import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="Directory for the HTML dashboards")
    parser.add_argument("--team", default=None, help="Only this team or queue (tier1, tier2)")
    args = parser.parse_args()
    summary = run_demo(args.out, team=args.team)
    print(json.dumps(summary, indent=2))
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
