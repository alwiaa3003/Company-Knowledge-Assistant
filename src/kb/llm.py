"""Thin wrapper around the google-genai client: one place for retries, rate limiting and usage.

Every Gemini call in the project goes through `call_with_retry`, so 429 (quota), 5xx (server) and
network errors (connection refused, DNS lookup failed, timeouts) are retried with exponential
backoff, and calls are spaced to stay under free-tier limits.
"""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, TypeVar

import httpx
from google import genai
from google.genai import errors, types

from kb.config import Settings, estimate_cost

T = TypeVar("T")
RETRYABLE_CODES = {429, 500, 502, 503, 504}
# No HTTP response at all (DNS failure, connection reset, timeout).
NETWORK_ERRORS: tuple[type[BaseException], ...] = (httpx.TransportError, ConnectionError, TimeoutError)
MAX_NETWORK_ATTEMPTS = 5


MAX_HINTED_WAIT_S = 120.0  # longer hinted waits mean the daily quota is gone


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini can't be reached after all retries (network down, DNS, timeouts)."""


class QuotaExhaustedError(GeminiUnavailableError):
    """Raised when a 429 asks for a very long wait (daily free-tier quota used up). Waiting hours is never useful."""


@dataclass
class Usage:
    """Token usage and latency for one or more LLM calls (add them with `+`)."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    calls: int = 0
    est_cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.latency_s + other.latency_s, self.calls + other.calls,
                     self.est_cost_usd + other.est_cost_usd)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (rounded)."""
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "latency_s": round(self.latency_s, 3), "calls": self.calls,
                "est_cost_usd": round(self.est_cost_usd, 8)}


@dataclass
class GenResult:
    """Text returned by the model plus its usage."""

    text: str
    usage: Usage = field(default_factory=Usage)


class RateLimiter:
    """Spaces calls so we never exceed `rpm` requests per minute (simple minimum interval)."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / max(rpm, 1)
        self._next = 0.0
        self._lock = threading.Lock()

    def reserve(self) -> float:
        """Book the next call slot and return how many seconds to wait before using it.

        Non-blocking, so async code (ADK callbacks) can `await asyncio.sleep(delay)` instead.
        """
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        return max(delay, 0.0)

    def wait(self) -> None:
        """Block until the next call is allowed."""
        delay = self.reserve()
        if delay > 0:
            time.sleep(delay)


@lru_cache(maxsize=None)
def get_client(api_key: str) -> genai.Client:
    """Create (once) and return a Gemini client. Cached so we reuse the HTTP connection."""
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=None)
def get_limiter(kind: str, rpm: int) -> RateLimiter:
    """One shared limiter per kind ("gen" / "embed")."""
    return RateLimiter(rpm)


def _retry_delay_from_error(err: errors.APIError) -> float | None:
    """Gemini 429 errors often say 'retry in 12.3s' / retryDelay '12s'. Use it if present."""
    match = re.search(r"retry(?:Delay)?\D{0,20}?(\d+(?:\.\d+)?)\s*s", str(err), re.IGNORECASE)
    return float(match.group(1)) if match else None


def call_with_retry(fn: Callable[[], T], limiter: RateLimiter, max_attempts: int = 6,
                    base_delay: float = 2.0, max_delay: float = 60.0) -> T:
    """Run `fn` with rate limiting; retry 429/5xx and network errors with exponential backoff + jitter.

    Non-retryable errors (e.g. 400 bad request, 401/403 bad key) are raised immediately,
    because retrying them would only waste time. Network errors get MAX_NETWORK_ATTEMPTS tries,
    then raise GeminiUnavailableError with a readable message.
    """
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        try:
            return fn()
        except NETWORK_ERRORS as err:
            if attempt >= min(max_attempts, MAX_NETWORK_ATTEMPTS):
                raise GeminiUnavailableError(
                    f"Could not reach the Gemini API after {attempt} attempts "
                    f"({type(err).__name__}: {err}). Check your internet connection, DNS, VPN or proxy."
                ) from err
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  [retry] network error ({type(err).__name__}); waiting {delay:.1f}s "
                  f"(attempt {attempt}/{MAX_NETWORK_ATTEMPTS})")
            time.sleep(delay)
        except errors.APIError as err:
            if err.code not in RETRYABLE_CODES or attempt == max_attempts:
                raise
            hinted = _retry_delay_from_error(err) if err.code == 429 else None
            quota = re.search(r"quota(?:Metric|Id)['\"]?\s*[:=]\s*['\"]([^'\"]+)", str(err))
            detail = f" [{quota.group(1).rsplit('/', 1)[-1]}]" if quota else ""
            if hinted is not None and hinted > MAX_HINTED_WAIT_S:
                raise QuotaExhaustedError(
                    f"Gemini quota exhausted{detail}: the server asks to wait {hinted:.0f}s "
                    f"(> {MAX_HINTED_WAIT_S:.0f}s). Likely the daily free-tier limit; try again later."
                ) from err
            delay = hinted if hinted is not None else min(max_delay, base_delay * 2 ** (attempt - 1))
            delay += random.uniform(0, 1)
            print(f"  [retry] Gemini returned {err.code}{detail}; waiting {delay:.1f}s (attempt {attempt}/{max_attempts})")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def usage_from_response(resp: types.GenerateContentResponse, settings: Settings, latency_s: float) -> Usage:
    """Extract token counts. Thinking tokens are billed as output, so we count them as output."""
    meta = resp.usage_metadata
    tokens_in = (meta.prompt_token_count or 0) if meta else 0
    tokens_out = ((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) if meta else 0
    return Usage(tokens_in, tokens_out, latency_s, 1, estimate_cost(settings, tokens_in, tokens_out))


def generate(settings: Settings, prompt: str, system: str | None = None, temperature: float = 0.0,
             json_schema: Any = None, max_output_tokens: int | None = None) -> GenResult:
    """Call the generation model once and return its text plus usage.

    `json_schema` (a pydantic model or dict) switches on JSON mode with that response schema.
    """
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if json_schema is not None else None,
        response_schema=json_schema,
    )
    client = get_client(settings.gemini_api_key)
    limiter = get_limiter("gen", settings.gen_rpm)
    start = time.perf_counter()
    resp = call_with_retry(
        lambda: client.models.generate_content(model=settings.gen_model, contents=prompt, config=config),
        limiter,
    )
    latency = time.perf_counter() - start
    return GenResult(text=(resp.text or "").strip(), usage=usage_from_response(resp, settings, latency))
