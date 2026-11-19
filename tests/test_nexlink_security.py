"""NexLink envelope and client tokens: secrets stay out of JSON, repr and errors. Offline."""

import json

import pytest
from pydantic import ValidationError

from support_engine.copilot import TokenError
from support_engine.nexlink import (
    SCOPE_INGEST,
    SCOPE_METRICS,
    ClientRegistry,
    Envelope,
    outgoing_token,
)

HUB = "hub-token-0123456789abcdef"
CODER = "coder-token-0123456789abcdef"


def test_envelope_token_never_serialized_or_shown() -> None:
    env = Envelope(sender="nexhub", target="nexsupport", action="support.metrics", token=HUB)
    assert env.token is not None and env.token.get_secret_value() == HUB
    assert HUB not in json.dumps(env.model_dump(mode="json"))
    assert HUB not in env.model_dump_json()
    assert HUB not in repr(env) and HUB not in str(env)


def test_envelope_validation() -> None:
    with pytest.raises(ValidationError):
        Envelope(sender="NexHub", target="nexsupport", action="support.metrics")
    with pytest.raises(ValidationError):
        Envelope(sender="nexhub", target="nexsupport", action="metrics")
    with pytest.raises(ValidationError):
        Envelope(sender="nexhub", target="nexsupport", action="support.metrics", unknown=1)
    with pytest.raises(ValidationError):
        Envelope(sender="nexhub", target="nexsupport", action="support.metrics", context={"k": "x" * 500})
    env = Envelope(sender="nexhub", target="nexsupport", action="support.metrics")
    assert env.conversation == env.id
    assert Envelope(sender="a1", target="b1", action="x.y", correlation_id="run-1").conversation == "run-1"


def test_registry_from_env_and_dotenv(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"SUPPORT_ENGINE_NEXLINK_CLIENT_NEXHUB_TOKEN={HUB}\n"
        f"SUPPORT_ENGINE_NEXLINK_CLIENT_NEXHUB_SCOPES={SCOPE_INGEST}, {SCOPE_METRICS}\n",
        encoding="utf-8",
    )
    environ = {"SUPPORT_ENGINE_NEXLINK_CLIENT_NEXCODER_TOKEN": CODER}
    registry = ClientRegistry.from_env(environ, env_file)
    assert registry.names == ["nexcoder", "nexhub"]
    hub = registry.authenticate("nexhub", HUB)
    assert hub is not None and hub.scopes == {SCOPE_INGEST, SCOPE_METRICS}
    coder = registry.authenticate("nexcoder", CODER)
    assert coder is not None and coder.scopes == frozenset()
    assert registry.authenticate("nexhub", CODER) is None
    assert registry.authenticate("nexhub", None) is None
    assert registry.authenticate("unknown", HUB) is None
    assert HUB not in repr(registry) and HUB not in repr(hub)


def test_registry_refuses_bad_configuration() -> None:
    with pytest.raises(TokenError) as short:
        ClientRegistry.build({"nexhub": ("short", SCOPE_INGEST)})
    assert "short" not in str(short.value)
    with pytest.raises(TokenError):
        ClientRegistry.build({"nexhub": (HUB, "admin")})
    with pytest.raises(TokenError) as reused:
        ClientRegistry.build({"nexhub": (HUB, SCOPE_INGEST), "nexcoder": (HUB, SCOPE_METRICS)})
    assert HUB not in str(reused.value)
    with pytest.raises(TokenError):
        ClientRegistry.build({"Bad Name": (HUB, SCOPE_INGEST)})


def test_outgoing_token() -> None:
    environ = {"SUPPORT_ENGINE_NEXLINK_TOKEN_NEXHUB": HUB}
    assert outgoing_token("nexhub", environ=environ, env_file=None) == HUB
    with pytest.raises(TokenError) as missing:
        outgoing_token("nexcoder", environ=environ, env_file=None)
    assert "SUPPORT_ENGINE_NEXLINK_TOKEN_NEXCODER" in str(missing.value)
    with pytest.raises(TokenError) as short:
        outgoing_token("nexhub", environ={"SUPPORT_ENGINE_NEXLINK_TOKEN_NEXHUB": "abc"}, env_file=None)
    assert "abc" not in str(short.value)
