"""Offline unit tests for help desk ticket connectors (mock HTTP and synthetic fixtures)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from nexcoder_core import PathOutsideSandboxError

from support_engine import (
    CSVConnector,
    FreshdeskConnector,
    IntercomConnector,
    JiraSMConnector,
    JSONConnector,
    TicketStatus,
    ZendeskConnector,
    get_connector,
    load_tickets_csv,
    load_tickets_file,
    load_tickets_json,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_zendesk_to_ticket_and_mock_http_pagination() -> None:
    pages = {
        "/api/v2/tickets.json": {
            "tickets": [
                {
                    "id": 9001,
                    "subject": "Export report fails on large date range",
                    "description": "Timeout when exporting >90 days.",
                    "status": "hold",
                    "priority": "High",
                    "requester_id": 111,
                    "submitter_id": 111,
                    "assignee_id": 222,
                    "tags": ["reporting", "timeout"],
                    "locale": "en-US",
                    "created_at": "2026-02-01T10:00:00Z",
                    "updated_at": "2026-02-01T12:00:00Z",
                    "metric_set": {"reopens": 2, "solved_at": None},
                    "satisfaction_rating": {"score": "bad"},
                }
            ],
            "meta": {"has_more": True},
            "links": {"next": "/api/v2/tickets.json?page=2"},
        },
        "/api/v2/tickets.json?page=2": {
            "tickets": [
                {
                    "id": 9002,
                    "subject": "Password reset link expired",
                    "description": "User requested a fresh token.",
                    "status": "solved",
                    "priority": "normal",
                    "requester_id": 333,
                    "submitter_id": 222,
                    "tags": ["auth"],
                    "created_at": "2026-02-02T08:00:00Z",
                    "updated_at": "2026-02-02T09:00:00Z",
                    "satisfaction_rating": {"score": "good"},
                }
            ],
            "meta": {"has_more": False},
        },
        "/api/v2/tickets/9001/comments.json": {
            "comments": [
                {
                    "id": 501,
                    "author_id": 111,
                    "plain_body": "Happens every Monday morning.",
                    "public": True,
                    "created_at": "2026-02-01T10:05:00Z",
                },
                {
                    "id": 502,
                    "author_id": 222,
                    "html_body": "<p>Internal note: query plan scan on <b>invoices</b>.</p>",
                    "public": False,
                    "created_at": "2026-02-01T10:30:00Z",
                },
                {
                    "id": 503,
                    "author_id": -1,
                    "plain_body": "Automated status check.",
                    "public": True,
                    "created_at": "2026-02-01T11:00:00Z",
                },
            ],
            "next_page": None,
        },
        "/api/v2/tickets/9002/comments.json": {
            "comments": [
                {
                    "id": 504,
                    "author_id": 222,
                    "plain_body": "Sent a new reset link valid for 24 hours.",
                    "public": True,
                    "created_at": "2026-02-02T08:30:00Z",
                }
            ],
            "next_page": None,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.raw_path.decode("ascii")
        assert key in pages, f"Unexpected URL requested: {key}"
        return httpx.Response(200, json=pages[key])

    client = httpx.Client(base_url="https://synthetic.zendesk.com", transport=httpx.MockTransport(handler))
    connector = ZendeskConnector(base_url="https://synthetic.zendesk.com", client=client)
    tickets = connector.fetch_tickets(include_comments=True)

    assert len(tickets) == 2
    t1, t2 = tickets
    assert t1.id == "9001"
    assert t1.source == "zendesk"
    assert t1.status == TicketStatus.PENDING
    assert t1.priority == "high"
    assert t1.reopen_count == 2
    assert t1.csat == 0.0
    assert t1.language == "en-US"
    assert [c.author_role for c in t1.comments] == ["customer", "agent", "system"]
    assert t1.comments[1].public is False
    assert t1.comments[1].body == "Internal note: query plan scan on invoices."
    assert "Internal note" not in t1.thread_text()

    assert t2.id == "9002"
    assert t2.status == TicketStatus.SOLVED
    assert t2.is_resolved is True
    assert t2.csat == 1.0
    assert t2.solved_at == t2.updated_at


def test_freshdesk_to_ticket_and_mock_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/tickets":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 401,
                        "subject": "Tax calculation rounding difference",
                        "description": "<div>Invoice total differs by <b>0.01</b>.</div>",
                        "status": 4,
                        "priority": 3,
                        "requester_id": 700,
                        "responder_id": 800,
                        "tags": ["tax", "rounding"],
                        "created_at": "2026-02-03T11:00:00Z",
                        "updated_at": "2026-02-03T15:00:00Z",
                        "stats": {"resolved_at": "2026-02-03T15:00:00Z", "reopened_count": 1},
                        "survey_result": {"ratings": {"default_question": 103}},
                    }
                ],
            )
        if request.url.path == "/api/v2/tickets/401/conversations":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 4001,
                        "body_text": "Attached the sample order payload.",
                        "incoming": True,
                        "private": False,
                        "user_id": 700,
                        "created_at": "2026-02-03T11:30:00Z",
                    },
                    {
                        "id": 4002,
                        "body": "<p>Switched tax rule to half-even rounding.</p>",
                        "incoming": False,
                        "private": False,
                        "user_id": 800,
                        "created_at": "2026-02-03T14:45:00Z",
                    },
                ],
            )
        return httpx.Response(404)

    client = httpx.Client(base_url="https://synthetic.freshdesk.com", transport=httpx.MockTransport(handler))
    connector = FreshdeskConnector(base_url="https://synthetic.freshdesk.com", client=client)
    tickets = connector.fetch_tickets(include_conversations=True)

    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.id == "401"
    assert ticket.source == "freshdesk"
    assert ticket.description == "Invoice total differs by 0.01."
    assert ticket.status == TicketStatus.SOLVED
    assert ticket.priority == "high"
    assert ticket.reopen_count == 1
    assert ticket.csat == 1.0
    assert len(ticket.comments) == 2
    assert ticket.comments[0].author_role == "customer"
    assert ticket.comments[1].author_role == "agent"
    assert ticket.comments[1].body == "Switched tax rule to half-even rounding."


def test_jira_sm_adf_and_mock_http() -> None:
    adf_description = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Production webhook queue is backed up."},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "Latency exceeded 30s."},
                ],
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/api/3/search"
        return httpx.Response(
            200,
            json={
                "startAt": 0,
                "maxResults": 50,
                "total": 1,
                "issues": [
                    {
                        "id": "10099",
                        "key": "SUP-42",
                        "fields": {
                            "summary": "Webhook delivery delay",
                            "description": adf_description,
                            "status": {"name": "Waiting for customer"},
                            "priority": {"name": "Highest"},
                            "labels": ["webhooks", "queue"],
                            "components": [{"name": "messaging"}],
                            "reporter": {"accountId": "cust-1", "accountType": "customer"},
                            "created": "2026-02-04T09:00:00.000+0000",
                            "updated": "2026-02-04T10:30:00.000+0000",
                            "satisfaction": {"rating": 4},
                            "comment": {
                                "comments": [
                                    {
                                        "id": "c-10",
                                        "author": {"accountId": "cust-1", "accountType": "customer"},
                                        "body": {
                                            "type": "doc",
                                            "content": [
                                                {
                                                    "type": "paragraph",
                                                    "content": [{"type": "text", "text": "Still seeing delays."}],
                                                }
                                            ],
                                        },
                                        "created": "2026-02-04T09:15:00.000+0000",
                                        "jsdPublic": True,
                                    },
                                    {
                                        "id": "c-11",
                                        "author": {"accountId": "agent-9", "accountType": "atlassian"},
                                        "body": "Scaled worker pool from 4 to 12 replicas.",
                                        "created": "2026-02-04T09:45:00.000+0000",
                                        "properties": [
                                            {"key": "sd.public.comment", "value": {"internal": True}}
                                        ],
                                    },
                                ]
                            },
                        },
                    }
                ],
            },
        )

    client = httpx.Client(base_url="https://synthetic.atlassian.net", transport=httpx.MockTransport(handler))
    connector = JiraSMConnector(base_url="https://synthetic.atlassian.net", client=client)
    tickets = connector.fetch_tickets()

    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.id == "SUP-42"
    assert ticket.source == "jira_sm"
    assert ticket.status == TicketStatus.PENDING
    assert ticket.priority == "urgent"
    assert ticket.product == "messaging"
    assert ticket.csat == 0.75
    assert "Production webhook queue is backed up.\nLatency exceeded 30s." in ticket.description
    assert len(ticket.comments) == 2
    assert ticket.comments[0].author_role == "customer"
    assert ticket.comments[0].public is True
    assert ticket.comments[0].body == "Still seeing delays."
    assert ticket.comments[1].author_role == "agent"
    assert ticket.comments[1].public is False


def test_intercom_to_ticket_and_mock_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/conversations":
            return httpx.Response(
                200,
                json={
                    "conversations": [
                        {
                            "id": "ic-77",
                            "title": "Billing seat count question",
                            "state": "closed",
                            "priority": "priority",
                            "created_at": 1770200000,
                            "updated_at": 1770203600,
                        }
                    ],
                    "pages": {"next": None},
                },
            )
        if request.url.path == "/conversations/ic-77":
            return httpx.Response(
                200,
                json={
                    "id": "ic-77",
                    "title": "Billing seat count question",
                    "state": "closed",
                    "priority": "priority",
                    "created_at": 1770200000,
                    "updated_at": 1770203600,
                    "source": {
                        "subject": "Billing seat count question",
                        "body": "<p>Why was my team billed for 12 seats instead of 10?</p>",
                    },
                    "tags": {"tags": [{"name": "billing"}, {"name": "seats"}]},
                    "statistics": {"last_close_at": 1770203600, "count_reopens": 0},
                    "conversation_rating": {"rating": 5},
                    "custom_attributes": {"product": "subscriptions", "language": "en"},
                    "conversation_parts": {
                        "conversation_parts": [
                            {
                                "id": "p-1",
                                "part_type": "comment",
                                "body": "<p>Two guests were promoted to full members on Jan 28.</p>",
                                "author": {"type": "admin", "id": "adm-1"},
                                "created_at": 1770201800,
                            },
                            {
                                "id": "p-2",
                                "part_type": "note",
                                "body": "<p>Issued courtesy credit for this cycle.</p>",
                                "author": {"type": "admin", "id": "adm-1"},
                                "created_at": 1770202000,
                            },
                        ]
                    },
                },
            )
        return httpx.Response(404)

    client = httpx.Client(base_url="https://api.intercom.io", transport=httpx.MockTransport(handler))
    connector = IntercomConnector(client=client)
    tickets = connector.fetch_tickets(include_parts=True)

    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.id == "ic-77"
    assert ticket.source == "intercom"
    assert ticket.status == TicketStatus.CLOSED
    assert ticket.is_resolved is True
    assert ticket.product == "subscriptions"
    assert ticket.language == "en"
    assert ticket.tags == ["billing", "seats"]
    assert ticket.csat == 1.0
    assert len(ticket.comments) == 2
    assert ticket.comments[0].public is True
    assert ticket.comments[1].public is False


def test_csv_and_json_connectors_and_sandbox() -> None:
    csv_tickets = load_tickets_csv(FIXTURES_DIR / "tickets_sample.csv", sandbox_root=FIXTURES_DIR)
    assert len(csv_tickets) == 2
    assert csv_tickets[0].id == "CSV-101"
    assert csv_tickets[0].status == TicketStatus.SOLVED
    assert csv_tickets[0].tags == ["invoice", "pdf", "export"]
    assert csv_tickets[0].csat == 1.0
    assert len(csv_tickets[0].comments) == 2
    assert csv_tickets[0].extra["customer_tier"] == "enterprise"

    assert csv_tickets[1].id == "CSV-102"
    assert csv_tickets[1].tags == ["webhook", "security"]
    assert csv_tickets[1].csat == 0.5
    assert csv_tickets[1].comments[0].public is False

    json_tickets = load_tickets_json(FIXTURES_DIR / "tickets_sample.json", sandbox_root=FIXTURES_DIR)
    assert len(json_tickets) == 1
    assert json_tickets[0].id == "JSON-201"
    assert json_tickets[0].description == "Users are redirected back to the login screen after SAML assertion."
    assert json_tickets[0].csat == 1.0
    assert json_tickets[0].extra["environment"] == "production"

    jsonl_tickets = load_tickets_file(FIXTURES_DIR / "tickets_sample.jsonl", sandbox_root=FIXTURES_DIR)
    assert len(jsonl_tickets) == 2
    assert jsonl_tickets[0].id == "JSONL-301"
    assert jsonl_tickets[0].status == TicketStatus.PENDING
    assert jsonl_tickets[1].id == "JSONL-302"
    assert jsonl_tickets[1].csat == 1.0

    with pytest.raises(PathOutsideSandboxError):
        load_tickets_csv("../pyproject.toml", sandbox_root=FIXTURES_DIR)


def test_connector_registry_lookup() -> None:
    assert isinstance(get_connector("zendesk"), ZendeskConnector)
    assert isinstance(get_connector("freshdesk"), FreshdeskConnector)
    assert isinstance(get_connector("jira-sm"), JiraSMConnector)
    assert isinstance(get_connector("intercom"), IntercomConnector)
    assert isinstance(get_connector("csv"), CSVConnector)
    assert isinstance(get_connector("json"), JSONConnector)

    with pytest.raises(ValueError, match="Unknown connector"):
        get_connector("unknown_platform")
