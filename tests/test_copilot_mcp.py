"""MCP server: protocol handshake, tool listing, tool calls, errors and stdio. Offline."""

import asyncio
import io
import json

from copilot_data import EMAIL, GOOD, make_copilot, open_ticket

from support_engine.copilot import SUPPORTED_PROTOCOLS, CopilotMCPServer, CopilotService


def server(**kwargs):
    copilot, gen, judge = make_copilot(**kwargs)
    return CopilotMCPServer(CopilotService(copilot)), gen


def call(srv, message):
    return asyncio.run(srv.handle_payload(message))


def tool_call(srv, name, arguments, msg_id=1):
    return call(
        srv, {"jsonrpc": "2.0", "id": msg_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    )


def test_initialize_negotiates_protocol() -> None:
    srv, _ = server()
    answer = call(srv, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert answer["result"]["protocolVersion"] == "2024-11-05"
    assert answer["result"]["capabilities"] == {"tools": {"listChanged": False}}
    other = call(srv, {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert other["result"]["protocolVersion"] == SUPPORTED_PROTOCOLS[0]


def test_tools_list() -> None:
    srv, _ = server()
    tools = call(srv, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"suggest_reply", "answer_question", "summarize_ticket", "record_feedback"}
    suggest = next(t for t in tools if t["name"] == "suggest_reply")
    assert "ticket" in suggest["inputSchema"]["properties"] and suggest["description"]


def test_suggest_reply_tool() -> None:
    srv, _ = server()
    ticket = open_ticket().model_dump(mode="json")
    answer = tool_call(srv, "suggest_reply", {"ticket": ticket})
    result = answer["result"]
    assert result["isError"] is False
    data = result["structuredContent"]
    assert data["answer"] == GOOD and data["next_step"]["action"] == "review_reply"
    assert json.loads(result["content"][0]["text"]) == data
    assert EMAIL not in result["content"][0]["text"]


def test_answer_and_summary_and_feedback_tools() -> None:
    srv, _ = server()
    answer = tool_call(srv, "answer_question", {"question": "The invoice export fails with error 500"})
    assert answer["result"]["structuredContent"]["level"] == 1
    summary = tool_call(srv, "summarize_ticket", {"ticket": open_ticket().model_dump(mode="json")})
    assert summary["result"]["structuredContent"]["status"] == "open"
    fb = tool_call(srv, "record_feedback", {"ticket_ids": ["T-1"], "accepted": True})
    assert fb["result"]["structuredContent"]["rating"] == 1.0
    assert fb["result"]["structuredContent"]["stored"] is False


def test_invalid_arguments_do_not_echo_input() -> None:
    srv, _ = server()
    answer = tool_call(srv, "answer_question", {"question": "", "namespace": f"bad {EMAIL}"})
    result = answer["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "invalid_arguments" in text and "namespace" in text and EMAIL not in text


def test_tool_failure_reports_type_only() -> None:
    srv, _ = server(answers=[RuntimeError(f"model down {EMAIL}")])

    async def boom(_request):
        raise ValueError(EMAIL)

    srv.tools["answer_question"] = srv.tools["answer_question"].__class__(
        "answer_question", "x", srv.tools["answer_question"].model, boom
    )
    result = tool_call(srv, "answer_question", {"question": "hello"})["result"]
    assert result["isError"] is True and "ValueError" in result["content"][0]["text"]
    assert EMAIL not in result["content"][0]["text"]


def test_protocol_errors() -> None:
    srv, _ = server()
    assert tool_call(srv, "nope", {})["error"]["code"] == -32602
    assert call(srv, {"jsonrpc": "2.0", "id": 3, "method": "resources/list"})["error"]["code"] == -32601
    assert call(srv, {"id": 4, "method": "ping"})["error"]["code"] == -32600
    assert call(srv, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert call(srv, {"jsonrpc": "2.0", "id": 5, "method": "ping"}) == {"jsonrpc": "2.0", "id": 5, "result": {}}
    assert call(srv, [])["error"]["code"] == -32600
    bad_args = call(
        srv,
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "summarize_ticket", "arguments": [1]}},
    )
    assert bad_args["error"]["code"] == -32602
    assert json.loads(asyncio.run(srv.handle_json("{not json")))["error"]["code"] == -32700


def test_batch() -> None:
    srv, _ = server()
    answers = call(
        srv,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ],
    )
    assert [a["id"] for a in answers] == [1, 2]


def test_stdio_loop() -> None:
    srv, _ = server()
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(m) for m in lines) + "\n\n")
    stdout = io.StringIO()
    srv.run_stdio(stdin, stdout)
    out = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [o["id"] for o in out] == [1, 2]
    assert out[0]["result"]["serverInfo"]["name"] == "support-engine"
