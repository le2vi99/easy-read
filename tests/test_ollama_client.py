"""Unit tests for the Ollama client (no live Ollama required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from easy_read.ai.ollama_client import (
    OllamaClient,
    OllamaModelNotFoundError,
    OllamaNotRunningError,
    from_config,
)


def _client(**kwargs) -> OllamaClient:
    defaults = dict(
        base_url="http://localhost:11434",
        model="llama3.2",
        temperature=0.7,
        timeout=60,
    )
    return OllamaClient(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

def test_health_check_passes_when_model_available() -> None:
    tags_response = MagicMock()
    tags_response.status_code = 200
    tags_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
    tags_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=tags_response):
        _client().health_check()  # should not raise


def test_health_check_raises_when_server_down() -> None:
    import httpx as _httpx

    with patch("httpx.get", side_effect=_httpx.ConnectError("refused")):
        with pytest.raises(OllamaNotRunningError, match="Cannot reach Ollama"):
            _client().health_check()


def test_health_check_raises_when_model_missing() -> None:
    tags_response = MagicMock()
    tags_response.status_code = 200
    tags_response.json.return_value = {"models": [{"name": "mistral:latest"}]}
    tags_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=tags_response):
        with pytest.raises(OllamaModelNotFoundError, match="llama3.2"):
            _client().health_check()


def test_health_check_accepts_model_with_tag() -> None:
    """Model 'llama3.2' should match 'llama3.2:latest' in the available list."""
    tags_response = MagicMock()
    tags_response.status_code = 200
    tags_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
    tags_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=tags_response):
        _client(model="llama3.2").health_check()  # should not raise


def test_health_check_raises_on_http_error() -> None:
    import httpx as _httpx

    bad_resp = MagicMock()
    bad_resp.status_code = 500
    bad_resp.raise_for_status.side_effect = _httpx.HTTPStatusError(
        "server error", request=MagicMock(), response=bad_resp
    )

    with patch("httpx.get", return_value=bad_resp):
        with pytest.raises(OllamaNotRunningError, match="unexpected status"):
            _client().health_check()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_returns_response_text() -> None:
    gen_response = MagicMock()
    gen_response.status_code = 200
    gen_response.json.return_value = {"response": "  Hello world  "}
    gen_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=gen_response):
        result = _client().generate("Say hello")

    assert result == "Hello world"


def test_generate_raises_on_connection_error() -> None:
    import httpx as _httpx

    with patch("httpx.post", side_effect=_httpx.ConnectError("lost")):
        with pytest.raises(OllamaNotRunningError, match="Lost connection"):
            _client().generate("test")


def test_generate_raises_on_timeout() -> None:
    import httpx as _httpx

    with patch("httpx.post", side_effect=_httpx.TimeoutException("timed out")):
        with pytest.raises(OllamaNotRunningError, match="Lost connection"):
            _client().generate("test")


def test_generate_sends_correct_payload() -> None:
    gen_response = MagicMock()
    gen_response.json.return_value = {"response": "ok"}
    gen_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=gen_response) as mock_post:
        _client(model="mistral", temperature=0.3).generate("prompt text")

    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "mistral"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.3
    assert payload["prompt"] == "prompt text"


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

def test_from_config_builds_client() -> None:
    ai_cfg = MagicMock()
    ai_cfg.base_url = "http://localhost:11434/"
    ai_cfg.model = "llama3.2"
    ai_cfg.temperature = 0.5
    ai_cfg.timeout_seconds = 30

    client = from_config(ai_cfg)

    assert client.base_url == "http://localhost:11434"  # trailing slash stripped
    assert client.model == "llama3.2"
    assert client.temperature == 0.5
    assert client.timeout == 30
