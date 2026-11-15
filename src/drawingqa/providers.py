"""Small, auditable OpenAI-compatible transport; no model SDK dependency."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

import httpx
from pydantic import ValidationError

from .schemas import DrawingInventory, Extraction, ModelConfig, ModelResponse, Usage


class ProviderError(RuntimeError):
    """A redacted operational failure, with usage known before the failure."""

    def __init__(self, message: str, usage: Usage | None = None):
        super().__init__(message)
        self.usage = usage or Usage(usage_reported=False)


class InvalidExtraction(ValueError):
    """Model output did not satisfy the evidence contract."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidExtraction("Model JSON contains duplicate keys")
        result[key] = value
    return result


def _rescale_boxes(payload: dict, bbox_scale: int, key: str = "candidates") -> None:
    candidates = payload.get(key)
    if bbox_scale == 1 or not isinstance(candidates, list):
        return
    for candidate in candidates:
        box = candidate.get("bbox") if isinstance(candidate, dict) else None
        if isinstance(box, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
            candidate["bbox"] = [v / bbox_scale for v in box]


def parse_extraction(content: str, bbox_scale: int = 1) -> Extraction:
    """Accept one JSON object, optionally fenced/preceded by a thinking block.

    Never extract the first object from arbitrary prose: that can silently accept
    a partial answer or a second conflicting answer. No model repair call is made.
    """
    return _parse_object(content, bbox_scale, Extraction, "candidates")


def parse_inventory(content: str, bbox_scale: int = 1) -> DrawingInventory:
    return _parse_object(content, bbox_scale, DrawingInventory, "items")


def _parse_object(content: str, bbox_scale: int, schema, box_key: str):
    if len(content) > 2_000_000:
        raise InvalidExtraction("Model output exceeds the JSON parsing limit")
    text = content.strip()
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise InvalidExtraction("Model returned an unfinished thinking block")
        text = text[end + len("</think>"):].strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, flags=re.IGNORECASE)
        if not match:
            raise InvalidExtraction("Model returned an incomplete JSON fence")
        text = match.group(1).strip()
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON number")),
        )
        if not isinstance(payload, dict):
            raise InvalidExtraction("Model output must be one JSON object")
        _rescale_boxes(payload, bbox_scale, box_key)
        return schema.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        # Validation errors include input values. Keep provider text out of logs.
        if isinstance(exc, InvalidExtraction):
            raise
        raise InvalidExtraction("Model output is not valid evidence JSON; check the model/prompt and token limit") from None


def _retry_delay(response: httpx.Response | None, retry_index: int) -> float:
    if response is not None:
        value = response.headers.get("retry-after")
        if value:
            try:
                return min(30.0, max(0.0, float(value)))
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    return min(30.0, max(0.0, (date - datetime.now(timezone.utc)).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    pass
    return min(30.0, 0.5 * 2 ** retry_index + random.uniform(0, 0.25))


def _token_count(value) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: ModelConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.sleep = sleep
        self.client = httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds, connect=min(30.0, config.timeout_seconds)),
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def generate(self, system: str, user: str, image_paths: list[Path]) -> ModelResponse:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env, "").strip()
            if not key:
                raise ProviderError(f"Missing API key environment variable: {self.config.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        content = [{"type": "text", "text": user}]
        for value in image_paths:
            path = Path(value)
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ProviderError("Provider accepts prepared PNG, JPEG or WebP views only")
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError:
                raise ProviderError("Cannot read a prepared image view") from None
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
            **self.config.extra_body,
        }
        if self.config.json_mode:
            body["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        usage = Usage(usage_reported=False)
        unknown_retry_usage = False
        for attempt in range(self.config.max_retries + 1):
            usage.requests += 1
            usage.retries = attempt
            response = None
            try:
                response = self.client.post(f"{self.config.base_url}/chat/completions", headers=headers, json=body)
            except httpx.RequestError:
                unknown_retry_usage = True
                if attempt < self.config.max_retries:
                    self.sleep(_retry_delay(None, attempt))
                    continue
                usage.latency_seconds = time.monotonic() - started
                raise ProviderError("Model endpoint could not be reached or timed out", usage) from None
            if response.status_code == 429 or 500 <= response.status_code < 600:
                unknown_retry_usage = True
                if attempt < self.config.max_retries:
                    self.sleep(_retry_delay(response, attempt))
                    continue
            usage.latency_seconds = time.monotonic() - started
            if response.status_code == 202:
                raise ProviderError("Endpoint returned an asynchronous job; use a synchronous chat-completions deployment", usage)
            if not response.is_success:
                hints = {
                    401: "check the API key",
                    403: "check model access permissions",
                    404: "check base_url and model ID",
                    413: "reduce image size or number of views",
                    429: "rate limit reached; lower concurrency or retry later",
                }
                hint = hints.get(response.status_code, "check endpoint compatibility and server logs")
                raise ProviderError(f"Model endpoint returned HTTP {response.status_code}; {hint}", usage)
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError
                raw_usage = payload.get("usage") or {}
                input_tokens = _token_count(raw_usage.get("prompt_tokens"))
                output_tokens = _token_count(raw_usage.get("completion_tokens"))
                usage.input_tokens = input_tokens or 0
                usage.output_tokens = output_tokens or 0
                usage.usage_reported = input_tokens is not None and output_tokens is not None and not unknown_retry_usage
                if (usage.usage_reported and self.config.input_cost_per_million is not None
                        and self.config.output_cost_per_million is not None):
                    usage.estimated_cost_usd = (
                        usage.input_tokens * self.config.input_cost_per_million
                        + usage.output_tokens * self.config.output_cost_per_million
                    ) / 1_000_000
                choice = payload["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason in {"length", "max_tokens"}:
                    raise ProviderError("Model output was truncated; raise max_tokens or use smaller views", usage)
                if finish_reason == "content_filter":
                    raise ProviderError("Model endpoint filtered this response", usage)
                message = choice["message"]
                if message.get("refusal"):
                    raise ProviderError("Model endpoint declined this request", usage)
                result = message.get("content")
                if isinstance(result, list):
                    result = "".join(item.get("text", "") for item in result
                                     if isinstance(item, dict) and item.get("type") in {"text", "output_text"})
                if not isinstance(result, str) or not result.strip():
                    raise ProviderError("Model returned no visible answer content", usage)
                # Separate reasoning_content is intentionally not returned/stored.
                return ModelResponse(content=result, usage=usage, finish_reason=finish_reason)
            except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                raise ProviderError("Model endpoint returned an invalid chat-completions response", usage) from None
        raise ProviderError("Model request failed", usage)  # Defensive; loop always returns/raises.
