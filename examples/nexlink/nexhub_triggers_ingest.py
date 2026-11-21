"""NexHub asks the Support Engine to ingest tickets over NexLink, then reads the manifest.

Offline: the Support Engine NexLink server runs in this process with generated tokens.

    python examples/nexlink/nexhub_triggers_ingest.py

The reply holds counters only (tickets, chunks, masked PII per kind), never ticket text. The
example also shows that a project without the ingest:run scope (nexcoder) is refused.
"""

from __future__ import annotations

import json
from typing import Any

from demo_setup import Demo, build_demo, synthetic_tickets


def run(demo: Demo) -> dict[str, Any]:
    nexhub = demo.clients["nexhub"]
    manifest = nexhub.manifest("nexsupport")
    batch = [t.model_dump(mode="json") for t in synthetic_tickets()]

    reply = nexhub.request(
        "nexsupport",
        "support.ingest",
        {"tickets": batch},
        context={"floor": "support-demo", "run": "ingest-1"},
    )
    refused = demo.clients["nexcoder"].request("nexsupport", "support.ingest", {"tickets": batch})
    return {
        "actions": [a["name"] for a in manifest.result["actions"]] if manifest.ok else [],
        "status": reply.status,
        "correlation_id": reply.correlation_id,
        "result": reply.result,
        "nexcoder_ingest": refused.error.code.value if refused.error else refused.status,
    }


def main() -> int:
    demo = build_demo(tickets=[])
    print(json.dumps(run(demo), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
