"""Access token from env or .env, constant-time checks, no token in messages. Offline."""

import pytest

from support_engine.copilot import TokenAuth, TokenError, load_token, read_env_file

TOKEN = "env-token-0123456789abcdef"


def test_token_from_environment_wins(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SUPPORT_ENGINE_API_TOKEN=file-token-0123456789\n", encoding="utf-8")
    assert load_token(env_file=env_file, environ={"SUPPORT_ENGINE_API_TOKEN": TOKEN}) == TOKEN


def test_token_from_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nDB_URL=sqlite:///x.db\nexport SUPPORT_ENGINE_API_TOKEN=\"file-token-0123456789\"\n",
        encoding="utf-8",
    )
    assert load_token(env_file=env_file, environ={}) == "file-token-0123456789"


def test_env_file_parsing(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=1 # note\nB='two words'\nnot a pair\n=x\nC=\n", encoding="utf-8")
    assert read_env_file(env_file) == {"A": "1", "B": "two words", "C": ""}
    assert read_env_file(tmp_path / "missing.env") == {}


def test_missing_token_raises_without_value(tmp_path) -> None:
    with pytest.raises(TokenError, match="SUPPORT_ENGINE_API_TOKEN"):
        load_token(env_file=tmp_path / "none.env", environ={})
    with pytest.raises(TokenError):
        load_token(env_file=None, environ={"SUPPORT_ENGINE_API_TOKEN": "   "})


def test_short_token_is_refused_and_not_echoed() -> None:
    with pytest.raises(TokenError) as info:
        load_token(env_file=None, environ={"SUPPORT_ENGINE_API_TOKEN": "short-secret"})
    assert "short-secret" not in str(info.value)
    with pytest.raises(TokenError):
        TokenAuth("short")


def test_custom_variable_name() -> None:
    assert load_token("MY_TOKEN", None, environ={"MY_TOKEN": TOKEN}) == TOKEN


def test_token_auth_headers() -> None:
    auth = TokenAuth(TOKEN)
    assert auth.check({"Authorization": f"Bearer {TOKEN}"})
    assert auth.check({"authorization": f"bearer {TOKEN}"})
    assert auth.check({"X-API-Key": TOKEN})
    assert not auth.check({"Authorization": "Bearer wrong-token-0123456789"})
    assert not auth.check({"Authorization": f"Basic {TOKEN}"})
    assert not auth.check({})
    assert not auth.verify(None) and not auth.verify("")
    assert TOKEN not in repr(auth)
