"""LLM provider interface, the Gemini implementation and provider selection.

Gemini is called through its OpenAI-compatible endpoint with the `openai` SDK.
The provider throttles to `LLM_MAX_RPM`, backs off on HTTP 429 and caps total calls
per run with `LLM_MAX_CALLS_PER_RUN`. Keys are never logged.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from claimguard import config


class LLMError(Exception):
    """The LLM call failed (network, server error, timeout)."""


class QuotaExhausted(LLMError):
    """Rate or daily quota is exhausted after retries, or the per-run call cap was hit."""


@dataclass
class LLMResponse:
    """Raw text returned by a provider and the call latency in seconds."""

    text: str
    latency_s: float


class LLMProvider(Protocol):
    """Anything that can complete a system + user prompt at temperature 0."""

    name: str
    model: str

    def complete(self, system: str, user: str) -> LLMResponse: ...


class RateLimiter:
    """Minimum interval between calls (requests per minute) plus a total call cap."""

    def __init__(self, max_rpm: int, max_calls: int) -> None:
        self.interval = 60.0 / max(1, max_rpm)
        self.max_calls = max_calls
        self.calls = 0
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise QuotaExhausted(f"per-run call cap of {self.max_calls} reached")
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


class GeminiProvider:
    """Gemini via the OpenAI-compatible endpoint."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_rpm: int = 10,
        max_calls: int = 600,
        timeout_s: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key, base_url=config.GEMINI_BASE_URL, timeout=timeout_s, max_retries=0
        )
        self.limiter = RateLimiter(max_rpm, max_calls)
        self.max_retries = max_retries

    def complete(self, system: str, user: str) -> LLMResponse:
        """One chat completion with throttling and exponential backoff on 429."""
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            start = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                text = resp.choices[0].message.content or ""
                return LLMResponse(text=text, latency_s=time.monotonic() - start)
            except Exception as exc:  # noqa: BLE001 - classified below
                code = _status_code(exc)
                if code == 429:
                    if attempt == self.max_retries:
                        raise QuotaExhausted("HTTP 429 after retries") from exc
                    backoff = min(60.0, 2.0 ** (attempt + 1)) + random.uniform(0, 1)  # nosec B311
                    time.sleep(backoff)
                    continue
                raise LLMError(f"{type(exc).__name__} (status {code})") from exc
        raise LLMError("unreachable")


def get_provider(settings: config.Settings | None = None) -> tuple[LLMProvider, str]:
    """Pick the provider: Gemini when a real key is set, else the mock.

    Returns the provider and the LLM status to report (`ok` or `not_configured`).
    """
    from claimguard.llm.mock import MockProvider

    settings = settings or config.get_settings()
    if not settings.llm_configured:
        return MockProvider(), "not_configured"
    provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.llm_model,
        max_rpm=settings.llm_max_rpm,
        max_calls=settings.llm_max_calls_per_run,
    )
    return provider, "ok"
