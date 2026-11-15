"""Command-line interface for Support Intelligence Engine.

Provides management subcommands to import, list, show and summarize help desk
tickets using the pluggable SQLAlchemy storage layer.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from support_engine.connectors.files import load_tickets_file
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
        action="store_true",
        default=False,
        help="Mask sensitive PII entities before writing to the database",
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
    pii_note = " (PII masked)" if args.mask_pii else ""
    sys.stdout.write(f"Imported {count} ticket(s) from '{args.input}'{pii_note}.\n")
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

    with TicketStore(target=args.db_url) as store:
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
