"""NexLink: the Support Engine talks to the other projects of the portfolio.

The Support Engine exposes its actions (ingest, suggest a reply, metrics) to NexHub, NexCoder
and the other projects, and consumes the nexdash (askdash) report engine for dashboards.

Typical use::

    from support_engine.nexlink import ClientRegistry, NexDashClient, build_support_server

    nexdash = NexDashClient.from_env("http://127.0.0.1:8000")      # token: NEXDASH_ENGINE_TOKEN
    server = build_support_server(ClientRegistry.from_env(), nexdash=nexdash)
    reply = server.handle_sync(envelope_dict, token)                # or create_nexlink_app(server)

See ``docs/NEXLINK.md`` for the envelope, the action catalog, the scopes and the examples.
"""

from support_engine.nexlink.client import HttpTransport, InProcessTransport, NexLinkClient, Transport
from support_engine.nexlink.nexdash import (
    SUPPORT_MODEL,
    EngineReportTransport,
    NexDashClient,
    ReportError,
    ReportParameters,
    ReportRequest,
    RestReportTransport,
)
from support_engine.nexlink.protocol import PROTOCOL, Envelope, ErrorCode, ErrorInfo, Reply
from support_engine.nexlink.security import (
    SCOPE_INGEST,
    SCOPE_METRICS,
    SCOPE_SUGGEST,
    SCOPES,
    Client,
    ClientRegistry,
    outgoing_token,
)
from support_engine.nexlink.server import MANIFEST_ACTION, ActionError, ActionSpec, Manifest, NexLinkServer
from support_engine.nexlink.support import (
    PROJECT,
    IngestParams,
    MetricsParams,
    SupportActions,
    SupportWorkspace,
    build_support_server,
)

__all__ = [
    "MANIFEST_ACTION",
    "PROJECT",
    "PROTOCOL",
    "SCOPES",
    "SCOPE_INGEST",
    "SCOPE_METRICS",
    "SCOPE_SUGGEST",
    "SUPPORT_MODEL",
    "ActionError",
    "ActionSpec",
    "Client",
    "ClientRegistry",
    "EngineReportTransport",
    "Envelope",
    "ErrorCode",
    "ErrorInfo",
    "HttpTransport",
    "InProcessTransport",
    "IngestParams",
    "Manifest",
    "MetricsParams",
    "NexDashClient",
    "NexLinkClient",
    "NexLinkServer",
    "Reply",
    "ReportError",
    "ReportParameters",
    "ReportRequest",
    "RestReportTransport",
    "SupportActions",
    "SupportWorkspace",
    "Transport",
    "build_support_server",
    "outgoing_token",
]
