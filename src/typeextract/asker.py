"""Turns "answer these N questions about this state" into as few valid Jev requests as possible.

* answers already in the cache are not asked again;
* the rest are packed into requests that respect Jev's token and question limits
  (64k tokens per request, 32k for state + the longest question);
* a request the server still rejects as too large is bisected and retried, and the learned
  limit is lowered for every later request;
* every answer is validated against its question; missing or invalid answers are re-asked once;
* each request is charged against an optional budget before it is sent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .cache import Cache
from .data import Metrics
from .errors import BudgetExceededError, RequestTooLargeError, ResponseValidationError
from .jev import Backend
from .questions import Answer, Question, parse_answer

log = logging.getLogger("typeextract")


@dataclass(frozen=True)
class Limits:
    """Server limits. Defaults are Jev 1.13's published limits; ``safety`` leaves headroom
    because tokens are estimated client-side."""

    request_tokens: int = 64_000
    state_plus_question_tokens: int = 32_000
    max_questions: int = 200
    max_choice_options: int = 255
    safety: float = 0.85


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def estimate_tokens(obj: Any) -> int:
    """Conservative token estimate: ~3 UTF-8 bytes per token (English averages ~4 chars/token;
    CJK characters are 3 bytes and about one token each)."""
    s = obj if isinstance(obj, str) else canonical(obj)
    return len(s.encode("utf-8")) // 3 + 1


class Budget:
    """A spend ceiling shared by every request of an extractor."""

    def __init__(self, max_usd: float | None, price_per_mtok: float):
        self.max_usd = max_usd
        self.price_per_mtok = price_per_mtok
        self.spent = 0.0
        self._lock = threading.Lock()

    def cost(self, tokens: int) -> float:
        return tokens * self.price_per_mtok / 1_000_000

    def reserve(self, tokens: int) -> float:
        cost = self.cost(tokens)
        with self._lock:
            if self.max_usd is not None and self.spent + cost > self.max_usd:
                raise BudgetExceededError(
                    f"budget of ${self.max_usd:.4f} would be exceeded (spent ${self.spent:.4f}, "
                    f"next request ~${cost:.6f})"
                )
            self.spent += cost
        return cost

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.spent += actual - reserved


class Asker:
    def __init__(
        self,
        backend: Backend,
        model: str,
        *,
        limits: Limits | None = None,
        cache: Cache | None = None,
        budget: Budget,
    ):
        self.backend = backend
        self.model = model
        self.limits = limits or Limits()
        self.cache = cache
        self.budget = budget
        self.served_model: str | None = None
        self._request_tokens = int(self.limits.request_tokens * self.limits.safety)
        self._max_questions = self.limits.max_questions

    def _key(self, state_json: str, question: Question) -> str:
        raw = f"{self.model}\x00{state_json}\x00{canonical(question)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def ask(
        self, state: Any, questions: Mapping[str, Question], metrics: Metrics
    ) -> dict[str, Answer]:
        if not questions:
            return {}
        state_json = canonical(state)
        state_tokens = estimate_tokens(state_json)
        out: dict[str, Answer] = {}
        pending: list[tuple[str, Question, str, int]] = []
        for key, q in questions.items():
            ck = self._key(state_json, q)
            hit = self.cache.get(ck) if self.cache is not None else None
            if hit is not None:
                try:
                    out[key] = Answer.from_dict(hit)
                    metrics.cached_questions += 1
                    continue
                except (KeyError, TypeError):
                    pass  # corrupt entry: ask again
            pending.append((key, q, ck, estimate_tokens(q)))
        metrics.questions += len(questions)
        if pending:
            batches = self._pack(state_tokens, pending)
            results = await asyncio.gather(
                *(self._send(state, state_tokens, b, metrics) for b in batches),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, BaseException):
                    raise r
                out.update(r)
        return out

    def _pack(
        self, state_tokens: int, pending: list[tuple[str, Question, str, int]]
    ) -> list[list[tuple[str, Question, str, int]]]:
        sq_budget = int(self.limits.state_plus_question_tokens * self.limits.safety)
        batches: list[list[tuple[str, Question, str, int]]] = []
        current: list[tuple[str, Question, str, int]] = []
        used = state_tokens
        for item in pending:
            tokens = item[3]
            if state_tokens + tokens > sq_budget:
                raise RequestTooLargeError(
                    f"state (~{state_tokens} tokens) plus question {item[0]!r} (~{tokens} tokens) "
                    f"exceeds the {self.limits.state_plus_question_tokens}-token limit; "
                    "reduce window_chars / context_chars or the number of options"
                )
            if current and (used + tokens > self._request_tokens or len(current) >= self._max_questions):
                batches.append(current)
                current, used = [], state_tokens
            current.append(item)
            used += tokens
        if current:
            batches.append(current)
        return batches

    def _learn(self, tokens: int, n_questions: int, exc: RequestTooLargeError, floor: int) -> None:
        """The server rejected a request we thought was valid: lower the limit it named (to the
        value it stated, if any). If the message named neither, both shrink a little (not by
        half: batches packed under the old limit keep failing for a while and would ratchet it
        down). The token limit never drops below ``floor``, what one question needs with this state."""
        new_tokens, new_questions = self._request_tokens, self._max_questions
        if exc.limit_kind != "tokens":
            target = exc.limit if exc.limit_kind == "questions" and exc.limit else int(n_questions * 0.7)
            new_questions = max(1, min(new_questions, target, n_questions - 1))
        if exc.limit_kind != "questions":
            target = int(exc.limit * self.limits.safety) if exc.limit_kind == "tokens" and exc.limit else int(tokens * 0.8)
            new_tokens = max(floor, min(new_tokens, target))
        if (new_tokens, new_questions) != (self._request_tokens, self._max_questions):
            self._request_tokens, self._max_questions = new_tokens, new_questions
            log.warning(
                "Jev rejected a request as too large; now packing at most ~%d tokens / %d questions per request",
                new_tokens,
                new_questions,
            )

    async def _send(
        self,
        state: Any,
        state_tokens: int,
        batch: list[tuple[str, Question, str, int]],
        metrics: Metrics,
        retry_invalid: bool = True,
    ) -> dict[str, Answer]:
        est = state_tokens + sum(item[3] for item in batch)
        reserved = self.budget.reserve(est)
        rejection: RequestTooLargeError | None = None
        try:
            resp = await self.backend.evaluate(state, {k: q for k, q, _, _ in batch}, self.model)
        except BaseException as exc:
            self.budget.settle(reserved, 0.0)
            metrics.retries += getattr(exc, "retries", 0)
            if not isinstance(exc, RequestTooLargeError) or len(batch) == 1:
                raise
            rejection = exc
        if rejection is not None:  # rejected as too large: lower the named limit and re-pack
            self._learn(est, len(batch), rejection, state_tokens + max(item[3] for item in batch))
            metrics.splits += 1
            parts = self._pack(state_tokens, batch)
            if len(parts) == 1:
                mid = len(batch) // 2
                parts = [batch[:mid], batch[mid:]]
            results = await asyncio.gather(
                *(self._send(state, state_tokens, part, metrics, retry_invalid) for part in parts),
                return_exceptions=True,
            )
            merged: dict[str, Answer] = {}
            for r in results:
                if isinstance(r, BaseException):
                    raise r
                merged.update(r)
            return merged
        tokens = resp.input_tokens if resp.input_tokens is not None else est
        cost = self.budget.cost(tokens)
        self.budget.settle(reserved, cost)
        metrics.requests += 1
        metrics.retries += resp.retries
        metrics.input_tokens += tokens
        metrics.cost_usd += cost
        if resp.model:
            self.served_model = resp.model

        out: dict[str, Answer] = {}
        invalid: list[tuple[str, Question, str, int]] = []
        problems: list[str] = []
        to_cache: list[tuple[str, dict[str, Any]]] = []
        for item in batch:
            key, q, ck, _ = item
            try:
                ans = parse_answer(resp.answers.get(key), q, key)
            except ResponseValidationError as exc:
                invalid.append(item)
                problems.append(str(exc))
                continue
            out[key] = ans
            to_cache.append((ck, ans.to_dict()))
        if self.cache is not None and to_cache:
            set_many = getattr(self.cache, "set_many", None)
            if set_many is not None:
                set_many(to_cache)
            else:
                for ck, value in to_cache:
                    self.cache.set(ck, value)
        if invalid:
            if not retry_invalid:
                raise ResponseValidationError(
                    f"{len(invalid)} answer(s) missing or invalid after a retry: {problems[0]}"
                )
            log.warning("%d answer(s) missing or invalid (%s); asking again", len(invalid), problems[0])
            out.update(await self._send(state, state_tokens, invalid, metrics, retry_invalid=False))
        return out
