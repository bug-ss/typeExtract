"""The network boundary: a Backend protocol and the HTTP client for Jev's ``POST /v1/systemone``."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from .errors import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    RateLimitError,
    RequestTooLargeError,
    ResponseValidationError,
    ServerError,
    TransportError,
)

log = logging.getLogger("typeextract")

DEFAULT_BASE_URL = "https://api.typesafe.ai"
SYSTEM_ONE_PATH = "/v1/systemone"
DEFAULT_MODEL = "jev-latest"
PRICE_PER_MTOK = 0.042  # USD per million input tokens; output tokens are free


@dataclass
class BackendResponse:
    answers: dict[str, Any]
    model: str | None = None
    input_tokens: int | None = None
    retries: int = 0


@runtime_checkable
class Backend(Protocol):
    """Anything that answers typed questions about a state (Jev, a local model, a fake)."""

    name: str

    async def evaluate(
        self, state: Any, questions: dict[str, dict[str, Any]], model: str
    ) -> BackendResponse: ...

    async def aclose(self) -> None: ...


class RateLimiter:
    """Token buckets for requests/minute and tokens/second, plus a shared cooldown after 429s.

    One limiter is shared by every thread and event loop using the backend (the lock only guards
    a few arithmetic operations; waiting happens with ``asyncio.sleep`` outside it).
    """

    def __init__(self, requests_per_minute: float | None, tokens_per_second: float | None):
        self.rpm = requests_per_minute or 0.0
        self.tps = tokens_per_second or 0.0
        self._req_cap = max(1.0, self.rpm / 12)  # about five seconds of burst
        self._tok_cap = self.tps  # one second of burst
        self._req, self._tok = self._req_cap, self._tok_cap
        self._last = time.monotonic()
        self._not_before = 0.0
        self._lock = threading.Lock()

    async def acquire(self, tokens: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed, self._last = now - self._last, now
                if self.rpm:
                    self._req = min(self._req_cap, self._req + elapsed * self.rpm / 60)
                if self.tps:
                    self._tok = min(self._tok_cap, self._tok + elapsed * self.tps)
                need = min(tokens, self._tok_cap) if self.tps else 0
                wait = self._not_before - now
                if self.rpm and self._req < 1:
                    wait = max(wait, (1 - self._req) * 60 / self.rpm)
                if self.tps and self._tok < need:
                    wait = max(wait, (need - self._tok) / self.tps)
                if wait <= 0:
                    if self.rpm:
                        self._req -= 1
                    if self.tps:
                        self._tok -= need
                    return
            await asyncio.sleep(wait)

    def cooldown(self, seconds: float) -> None:
        """Pause every request on this limiter (the server asked us to back off)."""
        with self._lock:
            self._not_before = max(self._not_before, time.monotonic() + seconds)


@dataclass
class _LoopState:
    client: httpx.AsyncClient
    semaphore: asyncio.Semaphore


_TOO_LARGE_RE = re.compile(
    r"too (large|long|many)|exceed|context (length|window)|token limit|max(imum)? "
    r"(number|length|size|tokens)|payload|at most \d+ questions",
    re.IGNORECASE,
)


def _message(body: Any) -> str:
    if isinstance(body, str):
        return body[:300]
    if not isinstance(body, Mapping):
        return ""
    for key in ("error", "message", "detail"):
        value = body.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("message"), str):
            return value["message"]
        if isinstance(value, list):
            parts = [
                str(d.get("msg")) + (f" at {'.'.join(map(str, d.get('loc', [])))}" if d.get("loc") else "")
                for d in value
                if isinstance(d, Mapping) and d.get("msg")
            ]
            if parts:
                return "; ".join(parts)
    return json.dumps(body)[:300]


def _retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after-ms")
    if raw is not None:
        try:
            value = float(raw) / 1000
            if math.isfinite(value) and value >= 0:
                return value
        except ValueError:
            pass
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
        return value if math.isfinite(value) and value >= 0 else None
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def status_error(status: int, body: Any, headers: httpx.Headers) -> APIError:
    msg = _message(body) or "request failed"
    rid = headers.get("x-typesafe-request-id")
    ra = _retry_after(headers)
    if status in (401, 403):
        return AuthenticationError(msg, status, rid)
    if status == 413 or (status in (400, 422) and _TOO_LARGE_RE.search(msg)):
        return RequestTooLargeError(msg, status, rid)
    if status in (400, 404, 422):
        return InvalidRequestError(msg, status, rid)
    if status == 429:
        return RateLimitError(msg, status, rid, ra)
    if status in (408, 409) or status >= 500:  # 529 = overloaded
        return ServerError(msg, status, rid, ra)
    return APIError(msg, status, rid)


class JevBackend:
    """HTTP client for TypeSafe's System One API.

    Retries 408/429/5xx/529, timeouts and connection errors with exponential backoff and jitter,
    honours ``retry-after`` / ``retry-after-ms``, and paces requests with a client-side rate
    limiter so a large batch does not stampede into 429s. Safe to use from several event loops
    (each loop gets its own connection pool).
    """

    name = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 6,
        max_concurrency: int = 16,
        requests_per_minute: float | None = 1200,
        tokens_per_second: float | None = 250_000,
        backoff_initial: float = 0.5,
        backoff_max: float = 30.0,
        max_retry_after: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "No Jev API key: pass api_key=... or set TYPESAFE_API_KEY "
                "(create one at https://console.typesafe.ai)"
            )
        if max_concurrency < 1 or max_retries < 0 or timeout <= 0:
            raise ConfigurationError("max_concurrency >= 1, max_retries >= 0 and timeout > 0 required")
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.requests_per_minute = requests_per_minute
        self.tokens_per_second = tokens_per_second
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.max_retry_after = max_retry_after
        self._transport = transport
        self.limiter = RateLimiter(requests_per_minute, tokens_per_second)
        self._loops: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopState] = (
            weakref.WeakKeyDictionary()
        )

    def __repr__(self) -> str:  # never print the key
        return f"JevBackend(base_url={self.base_url!r})"

    def _state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        st = self._loops.get(loop)
        if st is None:
            from . import __version__

            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": f"typeextract/{__version__}",
                },
            )
            st = _LoopState(client, asyncio.Semaphore(self.max_concurrency))
            self._loops[loop] = st
        return st

    def _backoff(self, attempt: int) -> float:
        delay = min(self.backoff_max, self.backoff_initial * (2**attempt))
        return delay * (1 - random.random() * 0.25)

    async def evaluate(
        self, state: Any, questions: dict[str, dict[str, Any]], model: str
    ) -> BackendResponse:
        body = json.dumps(
            {"state": state, "model": model, "questions": questions}, ensure_ascii=False
        ).encode("utf-8")
        est_tokens = len(body) // 3 + 1
        st = self._state()
        attempt = 0
        while True:
            await self.limiter.acquire(est_tokens)
            async with st.semaphore:
                try:
                    resp = await st.client.post(SYSTEM_ONE_PATH, content=body)
                except httpx.TimeoutException as exc:
                    error: APIError = TransportError(f"timeout after {self.timeout}s ({type(exc).__name__})")
                except httpx.RequestError as exc:
                    error = TransportError(f"{type(exc).__name__}: {exc}")
                else:
                    if resp.status_code < 300:
                        return self._parse(resp, attempt)
                    try:
                        payload: Any = resp.json()
                    except ValueError:
                        payload = resp.text
                    error = status_error(resp.status_code, payload, resp.headers)
            if not error.retryable or attempt >= self.max_retries:
                error.retries = attempt
                raise error
            delay = self._backoff(attempt)
            if error.retry_after is not None:
                delay = min(max(delay, error.retry_after), self.max_retry_after)
            if isinstance(error, RateLimitError) or error.status == 529:
                self.limiter.cooldown(delay)
            log.info("Jev request failed (%s); retry %d/%d in %.1fs", error, attempt + 1, self.max_retries, delay)
            attempt += 1
            await asyncio.sleep(delay)

    @staticmethod
    def _parse(resp: httpx.Response, retries: int) -> BackendResponse:
        try:
            data = resp.json()
        except ValueError as exc:
            raise ResponseValidationError(f"response is not JSON: {resp.text[:200]!r}") from exc
        if not isinstance(data, Mapping) or not isinstance(data.get("answers"), Mapping):
            raise ResponseValidationError(f"response has no 'answers' object: {str(data)[:200]}")
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
        tokens = usage.get("input_tokens")
        return BackendResponse(
            answers=dict(data["answers"]),
            model=data.get("model") if isinstance(data.get("model"), str) else None,
            input_tokens=tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None,
            retries=retries,
        )

    async def aclose(self) -> None:
        """Close the connection pool of the running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        st = self._loops.pop(loop, None)
        if st is not None:
            await st.client.aclose()
