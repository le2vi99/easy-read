"""Ollama HTTP API client.

Talks to a locally running Ollama instance (default: http://localhost:11434).
Model and base URL are configurable via ~/.easy-read/config.toml.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class OllamaNotRunningError(Exception):
    """Raised when Ollama is not reachable."""


class OllamaModelNotFoundError(Exception):
    """Raised when the configured model is not pulled."""


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    temperature: float
    timeout: int

    def health_check(self) -> None:
        """Verify Ollama is running and the model is available.

        Raises:
            OllamaNotRunningError: if the Ollama server is unreachable.
            OllamaModelNotFoundError: if the model is not pulled locally.
        """
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise OllamaNotRunningError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                "Make sure Ollama is running: https://ollama.com\n"
                "  brew install ollama && ollama serve"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaNotRunningError(
                f"Ollama returned an unexpected status: {exc.response.status_code}"
            ) from exc

        data = resp.json()
        available = [m["name"] for m in data.get("models", [])]
        # Ollama model names may include a tag (e.g. "llama3.2:latest")
        model_base = self.model.split(":")[0]
        matches = [
            m for m in available
            if m.split(":")[0] == model_base or m == self.model
        ]
        if not matches:
            pulled = ", ".join(available) if available else "(none)"
            raise OllamaModelNotFoundError(
                f"Model '{self.model}' is not available locally.\n"
                f"Available models: {pulled}\n"
                f"Pull it with: ollama pull {self.model}"
            )

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the response text.

        Raises:
            OllamaNotRunningError: if the request fails.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise OllamaNotRunningError(
                f"Lost connection to Ollama during generation: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaNotRunningError(
                f"Ollama generation failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc

        return resp.json().get("response", "").strip()


def from_config(ai_config) -> OllamaClient:
    """Build an OllamaClient from an AiConfig dataclass."""
    return OllamaClient(
        base_url=ai_config.base_url.rstrip("/"),
        model=ai_config.model,
        temperature=ai_config.temperature,
        timeout=ai_config.timeout_seconds,
    )