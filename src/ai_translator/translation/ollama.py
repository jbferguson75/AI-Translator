from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import OllamaError


@dataclass(frozen=True, slots=True)
class Generation:
    model: str
    response: str
    created_at: str | None
    total_duration: int | None
    prompt_eval_count: int | None
    eval_count: int | None


class OllamaClient:
    """Small standard-library client for Ollama's local HTTP API."""

    def __init__(self, base_url: str = "http://localhost:11434", *, timeout_seconds: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, *, model: str, prompt: str, options: dict[str, Any] | None = None,
                 format: str | dict[str, Any] | None = None) -> Generation:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        if format is not None:
            payload["format"] = format
        data = self._request("POST", "/api/generate", payload)
        response = data.get("response")
        if not isinstance(response, str) or not response.strip():
            raise OllamaError(f"Ollama returned an empty response for model {model!r}")
        return Generation(
            model=str(data.get("model", model)),
            response=response,
            created_at=data.get("created_at"),
            total_duration=data.get("total_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/tags")
        models = data.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama /api/tags returned an invalid models list")
        return models

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama returned invalid JSON from {path}") from exc
        if not isinstance(result, dict):
            raise OllamaError(f"Ollama returned a non-object response from {path}")
        if "error" in result:
            raise OllamaError(str(result["error"]))
        return result
