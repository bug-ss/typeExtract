import asyncio
import json
import time

import httpx
import pytest

from typeextract.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    RateLimitError,
    RequestTooLargeError,
    ResponseValidationError,
    ServerError,
    TransportError,
)
from typeextract.jev import JevBackend, RateLimiter

OK = {
    "model": "jev-1.13.0",
    "answers": {"q": {"type": "noul", "noul": 0.95}},
    "usage": {"input_tokens": 296, "output_tokens": 20},
}


def backend(handler, **kw):
    kw.setdefault("backoff_initial", 0.001)
    return JevBackend("sk-test", transport=httpx.MockTransport(handler), **kw)


def call(b, questions=None):
    async def go():
        try:
            return await b.evaluate("state", questions or {"q": {"type": "noul", "instructions": "?"}}, "jev-latest")
        finally:
            await b.aclose()

    return asyncio.run(go())


def test_request_shape_and_response_parsing():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK)

    resp = call(backend(handler))
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer sk-test"
    assert set(seen["body"]) == {"state", "model", "questions"} and seen["body"]["model"] == "jev-latest"
    assert resp.model == "jev-1.13.0" and resp.input_tokens == 296 and resp.answers["q"]["noul"] == 0.95


def flaky(statuses, headers=None):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) <= len(statuses):
            return httpx.Response(statuses[len(calls) - 1], json={"error": "busy"}, headers=headers or {})
        return httpx.Response(200, json=OK)

    return handler, calls


@pytest.mark.parametrize("status", [429, 529, 500, 503, 408])
def test_transient_errors_are_retried(status):
    handler, calls = flaky([status, status])
    resp = call(backend(handler))
    assert len(calls) == 3 and resp.retries == 2


def test_retry_after_ms_is_honoured():
    handler, calls = flaky([429], headers={"retry-after-ms": "200"})
    t0 = time.perf_counter()
    call(backend(handler))
    assert time.perf_counter() - t0 >= 0.19 and len(calls) == 2


def test_retries_are_bounded():
    handler, calls = flaky([429] * 10)
    with pytest.raises(RateLimitError):
        call(backend(handler, max_retries=2))
    assert len(calls) == 3
    handler, _ = flaky([529] * 10)
    with pytest.raises(ServerError):
        call(backend(handler, max_retries=1))


@pytest.mark.parametrize(
    "status,body,error",
    [
        (401, {"error": "bad key"}, AuthenticationError),
        (403, {"error": "nope"}, AuthenticationError),
        (413, "too big", RequestTooLargeError),
        (422, {"detail": [{"msg": "too many questions", "loc": ["body", "questions"]}]}, RequestTooLargeError),
        (400, {"message": "request exceeds context length"}, RequestTooLargeError),
        (422, {"detail": [{"msg": "field required", "loc": ["body", "questions", "q", "criteria"]}]}, InvalidRequestError),
    ],
)
def test_non_retryable_errors(status, body, error):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, text=body)

    with pytest.raises(error):
        call(backend(handler))
    assert len(calls) == 1


def test_timeouts_and_connection_errors_are_retried_then_raised():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(TransportError):
        call(backend(handler, max_retries=2))
    assert len(calls) == 3


def test_invalid_success_body():
    with pytest.raises(ResponseValidationError):
        call(backend(lambda r: httpx.Response(200, text="<html>")))
    with pytest.raises(ResponseValidationError):
        call(backend(lambda r: httpx.Response(200, json={"model": "x"})))


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        JevBackend()
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-env")
    b = JevBackend()
    assert b.api_key == "sk-env" and "sk-env" not in repr(b)


def test_backend_is_reusable_across_event_loops():
    b = backend(lambda r: httpx.Response(200, json=OK))
    assert call(b).model == call(b).model == "jev-1.13.0"


def test_rate_limiter_paces_tokens_and_honours_cooldown():
    async def go():
        lim = RateLimiter(None, tokens_per_second=1000)
        t0 = time.perf_counter()
        await lim.acquire(1000)
        await lim.acquire(500)
        paced = time.perf_counter() - t0
        lim.cooldown(0.2)
        t1 = time.perf_counter()
        await lim.acquire(1)
        return paced, time.perf_counter() - t1

    paced, cooled = asyncio.run(go())
    assert paced >= 0.45 and cooled >= 0.19


def test_rate_limiter_paces_requests():
    async def go():
        lim = RateLimiter(requests_per_minute=600, tokens_per_second=None)  # 10/s, burst 50
        t0 = time.perf_counter()
        for _ in range(55):
            await lim.acquire(1)
        return time.perf_counter() - t0

    assert asyncio.run(go()) >= 0.4
