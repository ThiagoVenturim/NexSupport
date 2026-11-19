"""Command-line interface for Support Intelligence Engine.

Provides management subcommands to import, list, show and summarize help desk
tickets using the pluggable SQLAlchemy storage layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from support_engine.analytics import (
    AnalyticsConfig,
    AnalyticsPipeline,
    Outcome,
    QualityConfig,
    RiskConfig,
    ROIConfig,
    TrendConfig,
)
from support_engine.connectors.files import load_tickets_file
from support_engine.curation import CurationConfig, CurationPipeline, Feedback, FeedbackConfig, FeedbackStore, GapConfig
from support_engine.ingestion import ChunkingConfig, DedupConfig, IngestionConfig, IngestionPipeline
from support_engine.models import TicketStatus
from support_engine.storage import TicketStore


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="support-engine",
        description="Support Intelligence Engine management CLI",
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help="SQLAlchemy database URL (overrides DB_URL env variable)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # import subcommand
    import_parser = subparsers.add_parser(
        "import",
        help="Import tickets from a CSV, JSON or JSONL file into the database",
    )
    import_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input file (CSV, JSON, or JSONL)",
    )
    import_parser.add_argument(
        "--mask-pii",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask sensitive PII entities before writing to the database (default: on; "
        "--no-mask-pii stores customer data in clear text)",
    )

    # ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Clean, mask PII, deduplicate and chunk tickets from a file; store tickets and write chunks",
    )
    ingest_parser.add_argument("--input", "-i", required=True, help="Path to the input file (CSV, JSON, or JSONL)")
    ingest_parser.add_argument("--chunks-out", default=None, help="Write chunks as JSON Lines to this path")
    ingest_parser.add_argument("--max-chars", type=int, default=1200, help="Maximum characters per chunk")
    ingest_parser.add_argument("--overlap-chars", type=int, default=150, help="Overlap between chunk pieces")
    ingest_parser.add_argument(
        "--dedup-threshold", type=float, default=0.85, help="Near-duplicate Jaccard threshold (0..1]"
    )
    ingest_parser.add_argument("--no-dedup", action="store_true", help="Keep near-duplicate chunks")
    ingest_parser.add_argument("--no-summary", action="store_true", help="Do not build case summary chunks")
    ingest_parser.add_argument("--no-store", action="store_true", help="Do not write tickets to the database")

    # curate subcommand
    curate_parser = subparsers.add_parser(
        "curate",
        help="Ingest tickets, score answers, apply feedback and corrections, report gaps and quality metrics",
    )
    curate_parser.add_argument("--input", "-i", required=True, help="Path to the input file (CSV, JSON, or JSONL)")
    curate_parser.add_argument("--feedback", default=None, help="Feedback as JSON Lines or a JSON list")
    curate_parser.add_argument(
        "--save-feedback", action="store_true", help="Store the feedback file in the database (text is masked)"
    )
    curate_parser.add_argument(
        "--with-stored-feedback", action="store_true", help="Also use feedback already stored in the database"
    )
    curate_parser.add_argument("--chunks-out", default=None, help="Write curated chunks as JSON Lines to this path")
    curate_parser.add_argument("--report-out", default=None, help="Write metrics and article suggestions as JSON")
    curate_parser.add_argument("--min-score", type=float, default=None, help="Drop chunks scored below this (0..1)")
    curate_parser.add_argument(
        "--superseded", choices=["drop", "demote", "keep"], default="demote", help="Answers replaced by a correction"
    )
    curate_parser.add_argument(
        "--gap-min-occurrences", type=int, default=3, help="Repeats of a gap before an article is suggested"
    )

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Ingest tickets and report sentiment, trends, risk radar, service quality, root cause and ROI",
    )
    analyze_parser.add_argument("--input", "-i", required=True, help="Path to the input file (CSV, JSON, or JSONL)")
    analyze_parser.add_argument(
        "--agent-results", default=None, help="Agent outcomes (action, level, cached, attempts) as JSON Lines or a list"
    )
    analyze_parser.add_argument("--report-out", default=None, help="Write the full report as JSON")
    analyze_parser.add_argument("--now", default=None, help="ISO time the report is computed for (default: latest)")
    analyze_parser.add_argument("--window-hours", type=int, default=24, help="Trend window size in hours")
    analyze_parser.add_argument("--baseline-windows", type=int, default=7, help="Trend windows in the baseline")
    analyze_parser.add_argument("--min-count", type=int, default=3, help="Tickets in a window needed to alert")
    analyze_parser.add_argument("--group-by", default="product", help="Service quality grouping field")
    analyze_parser.add_argument("--include-resolved", action="store_true", help="Rate solved tickets on the radar")
    analyze_parser.add_argument("--currency", default="USD", help="Currency of the ROI money values")
    analyze_parser.add_argument("--hourly-cost", type=float, default=30.0, help="Cost of one support agent hour")
    analyze_parser.add_argument("--handle-minutes", type=float, default=12.0, help="Human minutes per question")
    analyze_parser.add_argument("--fixed-cost", type=float, default=0.0, help="Platform cost of the period")

    # copilot interfaces
    serve_parser = subparsers.add_parser(
        "serve", help="Run the copilot REST API (and MCP over HTTP) over curated tickets; needs an access token"
    )
    mcp_parser = subparsers.add_parser("mcp", help="Run the copilot MCP server over stdio")
    for sub in (serve_parser, mcp_parser):
        sub.add_argument("--input", "-i", required=True, help="Ticket history to answer from (CSV, JSON, or JSONL)")
        sub.add_argument("--feedback", default=None, help="Feedback as JSON Lines or a JSON list")
        sub.add_argument(
            "--save-feedback", action="store_true", help="Store feedback recorded by the copilot in the database"
        )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--env-file", default=".env", help="File read when SUPPORT_ENGINE_API_TOKEN is unset")

    # support metrics on askdash (NexDash)
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Load support metrics and analytics into a NexDash database and render the support dashboards",
    )
    metrics_parser.add_argument("--input", "-i", required=True, help="Path to the input file (CSV, JSON, or JSONL)")
    metrics_parser.add_argument(
        "--agent-results", default=None, help="Agent outcomes with ticket_id, as JSON Lines or a JSON list"
    )
    metrics_parser.add_argument("--now", default=None, help="ISO time the analytics are computed for (default: latest)")
    metrics_parser.add_argument(
        "--dashboard-out", default=None, help="Directory for one HTML dashboard per report model"
    )
    metrics_parser.add_argument(
        "--model",
        action="append",
        choices=["support_metrics", "support_insights"],
        default=None,
        help="Report model to render (repeatable; default both)",
    )
    metrics_parser.add_argument("--start", default=None, help="First day of the period (YYYY-MM-DD)")
    metrics_parser.add_argument("--end", default=None, help="Last day of the period (YYYY-MM-DD)")
    metrics_parser.add_argument("--team", default=None, help="Team or queue")
    metrics_parser.add_argument("--project", default=None, help="Product or project")
    metrics_parser.add_argument("--locale", choices=["en", "pt-BR"], default="en", help="Dashboard locale")
    metrics_server_parser = subparsers.add_parser(
        "metrics-server",
        help="Serve the support report models of NexDash over REST and MCP (tokens from the NexDash environment)",
    )
    metrics_server_parser.add_argument("--input", "-i", default=None, help="Tickets to load before serving")
    metrics_server_parser.add_argument("--agent-results", default=None, help="Agent outcomes with ticket_id")
    metrics_server_parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    metrics_server_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    metrics_server_parser.add_argument("--mcp", action="store_true", help="Serve MCP over stdio instead of HTTP")
    for sub in (metrics_parser, metrics_server_parser):
        sub.add_argument(
            "--nexdash-db",
            default=None,
            help="SQLAlchemy URL of the database NexDash reads (default: in memory for metrics, DB_URL for the server)",
        )

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List stored tickets")
    list_parser.add_argument(
        "--status",
        choices=["new", "open", "pending", "solved", "closed"],
        default=None,
        help="Filter tickets by status",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of tickets to return (default: 50)",
    )
    list_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Pagination offset (default: 0)",
    )

    # show subcommand
    show_parser = subparsers.add_parser("show", help="Show details and comments for a ticket")
    show_parser.add_argument(
        "--id",
        required=True,
        help="Unique identifier of the ticket",
    )

    # stats subcommand
    subparsers.add_parser("stats", help="Show summary statistics of stored tickets")

    return parser


def cmd_import(args: argparse.Namespace, store: TicketStore) -> int:
    path = Path(args.input)
    if not path.is_file():
        sys.stderr.write(f"Error: input file '{args.input}' not found.\n")
        return 1

    tickets = load_tickets_file(path)
    count = store.save_tickets(tickets, mask_pii=args.mask_pii)
    pii_note = " (PII masked)" if args.mask_pii else " (PII NOT masked)"
    sys.stdout.write(f"Imported {count} ticket(s) from '{args.input}'{pii_note}.\n")
    return 0


def cmd_ingest(args: argparse.Namespace, store: TicketStore | None) -> int:
    path = Path(args.input)
    if not path.is_file():
        sys.stderr.write(f"Error: input file '{args.input}' not found.\n")
        return 1
    try:
        config = IngestionConfig(
            chunking=ChunkingConfig(
                max_chars=args.max_chars, overlap_chars=args.overlap_chars, include_summary=not args.no_summary
            ),
            dedup=DedupConfig(enabled=not args.no_dedup, threshold=args.dedup_threshold),
        )
    except ValueError as exc:
        sys.stderr.write(f"Error: invalid ingestion settings: {exc}\n")
        return 2

    result = IngestionPipeline(config).process(load_tickets_file(path))
    if store is not None:
        # Tickets are already masked by the pipeline; masking again is a no-op safety net.
        store.save_tickets(result.tickets, mask_pii=True)
    if args.chunks_out:
        out = Path(args.chunks_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for chunk in result.chunks:
                fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")

    s = result.stats
    masked = ", ".join(f"{k}={v}" for k, v in sorted(s.pii_masked.items())) or "none"
    sys.stdout.write(
        f"Ingested {s.tickets} ticket(s): {s.comments_kept} interaction(s) kept, "
        f"{s.comments_empty + s.comments_duplicate} dropped, {s.chunks} chunk(s), "
        f"{s.chunks_duplicate} duplicate chunk(s) removed. PII masked: {masked}.\n"
    )
    return 0


def _load_feedback(path: Path) -> list[Feedback]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        records = json.loads(raw)
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [Feedback.model_validate(r) for r in records]


def cmd_curate(args: argparse.Namespace, feedback_store: FeedbackStore | None) -> int:
    path = Path(args.input)
    if not path.is_file():
        sys.stderr.write(f"Error: input file '{args.input}' not found.\n")
        return 1
    try:
        config = CurationConfig(
            min_score=args.min_score,
            feedback=FeedbackConfig(superseded_policy=args.superseded),
            gaps=GapConfig(min_occurrences=args.gap_min_occurrences),
        )
        feedback: list[Feedback] = []
        if args.feedback:
            fb_path = Path(args.feedback)
            if not fb_path.is_file():
                sys.stderr.write(f"Error: feedback file '{args.feedback}' not found.\n")
                return 1
            feedback = _load_feedback(fb_path)
    except ValueError as exc:
        # Report only the error type; the message could echo feedback text.
        sys.stderr.write(f"Error: invalid curation settings or feedback ({type(exc).__name__}).\n")
        return 2

    if feedback_store is not None:
        if args.save_feedback:
            feedback_store.add_many(feedback)
        if args.with_stored_feedback:
            seen = {f.id for f in feedback}
            feedback += [f for f in feedback_store.list_feedback() if f.id not in seen]

    ingested = IngestionPipeline().process(load_tickets_file(path))
    result = CurationPipeline(config).curate(ingested.tickets, ingested.chunks, feedback)
    if args.chunks_out:
        out = Path(args.chunks_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for chunk in result.chunks:
                fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "metrics": result.metrics.model_dump(mode="json"),
            "stats": result.stats.model_dump(mode="json"),
            "scores": {k: v.model_dump(mode="json") for k, v in result.scores.items()},
            "suggestions": [s.model_dump(mode="json") for s in result.suggestions],
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    m, s = result.metrics, result.stats
    helpful = f"{m.helpful_rate:.2f}" if m.helpful_rate is not None else "-"
    sys.stdout.write(
        f"Curated {s.tickets} ticket(s): mean score {m.mean_score:.2f}, {m.share_high:.0%} high, "
        f"{m.share_low:.0%} low. {s.chunks_out} chunk(s) out, {s.chunks_below_min} below min score, "
        f"{s.chunks_superseded} superseded, {s.corrections_added} correction(s) added. "
        f"Feedback {m.feedback_count} (helpful rate {helpful}). "
        f"Gaps {m.gaps} in {m.gap_clusters} cluster(s), {m.suggested_articles} article(s) suggested.\n"
    )
    return 0


def _load_records(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.is_file():
        sys.stderr.write(f"Error: input file '{args.input}' not found.\n")
        return 1
    try:
        config = AnalyticsConfig(
            trends=TrendConfig(
                window_hours=args.window_hours, baseline_windows=args.baseline_windows, min_count=args.min_count
            ),
            risk=RiskConfig(include_resolved=args.include_resolved),
            quality=QualityConfig(group_by=args.group_by),
            roi=ROIConfig(
                currency=args.currency,
                agent_hourly_cost=args.hourly_cost,
                handle_minutes=args.handle_minutes,
                fixed_cost=args.fixed_cost,
            ),
        )
        now = datetime.fromisoformat(args.now) if args.now else None
        if now is not None and now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        results = None
        if args.agent_results:
            results_path = Path(args.agent_results)
            if not results_path.is_file():
                sys.stderr.write(f"Error: agent results file '{args.agent_results}' not found.\n")
                return 1
            results = [Outcome.of(r) for r in _load_records(results_path)]
    except ValueError as exc:
        # Report only the error type; the message could echo file content.
        sys.stderr.write(f"Error: invalid analytics settings or agent results ({type(exc).__name__}).\n")
        return 2

    ingested = IngestionPipeline().process(load_tickets_file(path))
    report = AnalyticsPipeline(config).run(ingested.tickets, results, now=now)
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    s = report.sentiment
    line = (
        f"Analyzed {report.tickets} ticket(s): mean sentiment {s.mean:+.2f}, {s.negative_share:.0%} negative, "
        f"{s.worsening} worsening. Trends {len(report.trends)}, {len(report.alerts)} alert(s). "
        f"Risk: {len(report.high_risk())} high of {len(report.risk)} rated. "
        f"Root cause groups {len(report.root_causes)}."
    )
    if report.roi is not None:
        r = report.roi
        roi = f"{r.roi:.2f}" if r.roi is not None else "-"
        line += (
            f" Deflection {r.deflection_rate:.0%} of {r.questions}, {r.hours_saved:.1f} h saved, "
            f"net {r.net_savings:.2f} {r.currency}, ROI {roi}."
        )
    sys.stdout.write(line + "\n")
    return 0


def _metrics_inputs(args: argparse.Namespace):
    """Tickets and agent outcomes of the metrics commands, or an exit code on error."""
    from support_engine.metrics import build_support_metrics

    path = Path(args.input)
    if not path.is_file():
        sys.stderr.write(f"Error: input file '{args.input}' not found.\n")
        return 1
    try:
        now = datetime.fromisoformat(args.now) if getattr(args, "now", None) else None
        if now is not None and now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        results = None
        if args.agent_results:
            results_path = Path(args.agent_results)
            if not results_path.is_file():
                sys.stderr.write(f"Error: agent results file '{args.agent_results}' not found.\n")
                return 1
            results = _load_records(results_path)
        return build_support_metrics(load_tickets_file(path), results, now=now)
    except ValueError as exc:
        # Report only the error type; the message could echo file content.
        sys.stderr.write(f"Error: invalid tickets, agent results or time ({type(exc).__name__}).\n")
        return 2


def _nexdash_missing() -> int:
    sys.stderr.write("Error: support metrics need the nexdash package (askdash): pip install -e <askdash checkout>\n")
    return 2


def cmd_metrics(args: argparse.Namespace) -> int:
    try:
        from nexdash.executor import get_executor
        from nexdash.service import ServiceError

        from support_engine.metrics.askdash import build_engine, load_into_nexdash
    except ImportError:
        return _nexdash_missing()
    from support_engine.metrics import report_request

    models = args.model or ["support_metrics", "support_insights"]
    try:
        requests = [
            report_request(
                name,
                start=args.start,
                end=args.end,
                team=args.team,
                project=args.project,
                output="dashboard",
                locale=args.locale,
            )
            for name in models
        ]
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}.\n")
        return 2
    metrics = _metrics_inputs(args)
    if isinstance(metrics, int):
        return metrics

    executor = get_executor(args.nexdash_db or "sqlite:///:memory:")
    try:
        counts = load_into_nexdash(executor, metrics)
        written = []
        if args.dashboard_out:
            out = Path(args.dashboard_out)
            out.mkdir(parents=True, exist_ok=True)
            engine = build_engine(executor)
            for request in requests:
                try:
                    html = engine.render(request)
                except ServiceError as exc:
                    sys.stderr.write(f"Error: {exc}\n")
                    return 2
                target = out / f"{request['model']}.html"
                target.write_text(html, encoding="utf-8")
                written.append(target.name)
    finally:
        executor.close()

    report = metrics.report
    line = (
        f"Loaded {counts['support_metrics']} ticket(s) into support_metrics and "
        f"{counts['support_insights']} into support_insights: {len(report.high_risk())} high risk, "
        f"{len(report.alerts)} trend alert(s), {len(metrics.outcomes)} agent outcome(s) tied to tickets."
    )
    if written:
        line += f" Dashboards: {', '.join(written)}."
    sys.stdout.write(line + "\n")
    return 0


def cmd_metrics_server(args: argparse.Namespace) -> int:
    import logging

    try:
        from support_engine.metrics.askdash import create_server, serve_mcp_stdio
    except ImportError:
        return _nexdash_missing()
    metrics = None
    if args.input:
        metrics = _metrics_inputs(args)
        if isinstance(metrics, int):
            return metrics
    if args.mcp:
        # stdout carries protocol messages only.
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        serve_mcp_stdio(args.nexdash_db, metrics=metrics)
        return 0
    server = create_server(args.host, args.port, args.nexdash_db, metrics=metrics)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.executor.close()
    return 0


def _build_copilot(args: argparse.Namespace, feedback_store: FeedbackStore | None):
    from support_engine.copilot import Copilot

    path = Path(args.input)
    if not path.is_file():
        sys.stderr.write(f"Error: input file '{args.input}' not found.\n")
        return None
    feedback: list[Feedback] = []
    if args.feedback:
        fb_path = Path(args.feedback)
        if not fb_path.is_file():
            sys.stderr.write(f"Error: feedback file '{args.feedback}' not found.\n")
            return None
        try:
            feedback = _load_feedback(fb_path)
        except ValueError as exc:
            sys.stderr.write(f"Error: invalid feedback ({type(exc).__name__}).\n")
            return None
    return Copilot.from_tickets(load_tickets_file(path), feedback, feedback_store=feedback_store)


def cmd_serve(args: argparse.Namespace, feedback_store: FeedbackStore | None) -> int:
    from support_engine.copilot import InterfaceConfig, TokenError, load_token

    config = InterfaceConfig(env_file=args.env_file or None)
    try:
        token = load_token(config.token_env, config.env_file, min_chars=config.min_token_chars)
    except TokenError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    try:
        import uvicorn

        from support_engine.copilot.api import create_app
    except ImportError:
        sys.stderr.write("Error: the REST API needs the 'api' extra: pip install 'support-intelligence-engine[api]'\n")
        return 2
    copilot = _build_copilot(args, feedback_store)
    if copilot is None:
        return 1
    app = create_app(copilot, token=token, config=config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_mcp(args: argparse.Namespace, feedback_store: FeedbackStore | None, stdin=None, stdout=None) -> int:
    import logging

    from support_engine.copilot import CopilotMCPServer, CopilotService

    # stdout carries protocol messages only.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    copilot = _build_copilot(args, feedback_store)
    if copilot is None:
        return 1
    CopilotMCPServer(CopilotService(copilot)).run_stdio(stdin, stdout)
    return 0


def cmd_list(args: argparse.Namespace, store: TicketStore) -> int:
    status_filter = TicketStatus(args.status) if args.status else None
    tickets = store.list_tickets(status=status_filter, limit=args.limit, offset=args.offset)
    if not tickets:
        sys.stdout.write("No tickets found.\n")
        return 0

    header = f"{'ID':<16} {'STATUS':<10} {'PRIORITY':<10} {'SOURCE':<10} {'SUBJECT'}"
    sys.stdout.write(header + "\n")
    sys.stdout.write("-" * len(header) + "\n")
    for t in tickets:
        prio = t.priority or "-"
        subj = (t.subject[:40] + "...") if len(t.subject) > 43 else t.subject
        sys.stdout.write(f"{t.id:<16} {t.status.value:<10} {prio:<10} {t.source:<10} {subj}\n")
    return 0


def cmd_show(args: argparse.Namespace, store: TicketStore) -> int:
    ticket = store.get_ticket(args.id)
    if ticket is None:
        sys.stderr.write(f"Ticket '{args.id}' not found.\n")
        return 1

    sys.stdout.write(f"Ticket ID:    {ticket.id}\n")
    sys.stdout.write(f"Source:       {ticket.source}\n")
    sys.stdout.write(f"Subject:      {ticket.subject}\n")
    sys.stdout.write(f"Status:       {ticket.status.value}\n")
    sys.stdout.write(f"Priority:     {ticket.priority or '-'}\n")
    sys.stdout.write(f"Product:      {ticket.product or '-'}\n")
    sys.stdout.write(f"Language:     {ticket.language or '-'}\n")
    sys.stdout.write(f"Created:      {ticket.created_at}\n")
    if ticket.tags:
        sys.stdout.write(f"Tags:         {', '.join(ticket.tags)}\n")
    if ticket.csat is not None:
        sys.stdout.write(f"CSAT:         {ticket.csat:.2f}\n")
    sys.stdout.write(f"\nDescription:\n{ticket.description}\n")

    if ticket.comments:
        sys.stdout.write(f"\nComments ({len(ticket.comments)}):\n")
        for idx, comment in enumerate(ticket.comments, start=1):
            pub = "public" if comment.public else "internal"
            sys.stdout.write(f"  [{idx}] {comment.author_role} ({pub}) at {comment.created_at}:\n")
            for line in comment.body.splitlines():
                sys.stdout.write(f"      {line}\n")
    return 0


def cmd_stats(_args: argparse.Namespace, store: TicketStore) -> int:
    total = store.count_tickets()
    sys.stdout.write(f"Total tickets: {total}\n")
    for status in TicketStatus:
        count = store.count_tickets(status=status)
        sys.stdout.write(f"  {status.value:<10}: {count}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "ingest" and args.no_store:
        return cmd_ingest(args, None)

    if args.command == "analyze":
        return cmd_analyze(args)

    if args.command == "metrics":
        return cmd_metrics(args)

    if args.command == "metrics-server":
        return cmd_metrics_server(args)

    if args.command in ("serve", "mcp"):
        handler = cmd_serve if args.command == "serve" else cmd_mcp
        if not args.save_feedback:
            return handler(args, None)
        with FeedbackStore(target=args.db_url) as feedback_store:
            return handler(args, feedback_store)

    if args.command == "curate":
        if not (args.save_feedback or args.with_stored_feedback):
            return cmd_curate(args, None)
        with FeedbackStore(target=args.db_url) as feedback_store:
            return cmd_curate(args, feedback_store)

    with TicketStore(target=args.db_url) as store:
        if args.command == "ingest":
            return cmd_ingest(args, store)
        if args.command == "import":
            return cmd_import(args, store)
        if args.command == "list":
            return cmd_list(args, store)
        if args.command == "show":
            return cmd_show(args, store)
        if args.command == "stats":
            return cmd_stats(args, store)

    return 0


if __name__ == "__main__":
    sys.exit(main())
