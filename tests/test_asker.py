import asyncio

import pytest

from typeextract.asker import Asker, Budget, Limits
from typeextract.cache import MemoryCache, SQLiteCache
from typeextract.data import Metrics
from typeextract.errors import BudgetExceededError, RequestTooLargeError, ResponseValidationError
from typeextract.jev import BackendResponse
from typeextract.questions import noul


class Recorder:
    """Answers every Noul with 0.7; optional server-side question cap and dropped answers."""

    name = "recorder"

    def __init__(self, server_max_questions=None, drop=0, tokens=100):
        self.server_max_questions = server_max_questions
        self.drop = drop
        self.tokens = tokens
        self.sizes = []

    async def evaluate(self, state, questions, model):
        if self.server_max_questions and len(questions) > self.server_max_questions:
            raise RequestTooLargeError("too many questions", 422)
        self.sizes.append(len(questions))
        keys = list(questions)
        if self.drop:
            keys, self.drop = keys[self.drop :], 0
        return BackendResponse({k: {"type": "noul", "noul": 0.7} for k in keys}, "jev-test", self.tokens)

    async def aclose(self):
        pass


def ask(asker, questions, state="some state"):
    m = Metrics()
    return asyncio.run(asker.ask(state, questions, m)), m


def qs(n):
    return {f"q{i}": noul(f"is statement {i} true?") for i in range(n)}


def test_questions_are_packed_under_the_question_limit():
    backend = Recorder()
    asker = Asker(backend, "jev", limits=Limits(max_questions=10), budget=Budget(None, 0.042))
    answers, m = ask(asker, qs(35))
    assert len(answers) == 35 and backend.sizes == [10, 10, 10, 5] and m.requests == 4


def test_questions_are_packed_under_the_token_limit():
    backend = Recorder()
    limits = Limits(request_tokens=300, state_plus_question_tokens=300, safety=1.0)
    asker = Asker(backend, "jev", limits=limits, budget=Budget(None, 0.042))
    answers, _ = ask(asker, qs(40))
    assert len(answers) == 40 and len(backend.sizes) > 1 and sum(backend.sizes) == 40


def test_state_too_large_for_any_request_raises_before_sending():
    backend = Recorder()
    asker = Asker(backend, "jev", limits=Limits(state_plus_question_tokens=100), budget=Budget(None, 0.042))
    with pytest.raises(RequestTooLargeError):
        ask(asker, qs(1), state="x" * 1000)
    assert backend.sizes == []


def test_server_rejection_bisects_and_learns():
    backend = Recorder(server_max_questions=8)
    asker = Asker(backend, "jev", limits=Limits(max_questions=200), budget=Budget(None, 0.042))
    answers, m = ask(asker, qs(50))
    assert len(answers) == 50 and m.splits >= 1 and max(backend.sizes) <= 8
    before = m.splits
    _, m2 = ask(asker, {f"z{i}": noul(f"z {i}") for i in range(50)}, state="other")
    assert m2.splits < before  # the learned limit avoids most rejections next time


def test_missing_answers_are_asked_again_once():
    backend = Recorder(drop=3)
    asker = Asker(backend, "jev", budget=Budget(None, 0.042))
    answers, m = ask(asker, qs(10))
    assert len(answers) == 10 and m.requests == 2


def test_persistently_missing_answers_raise():
    class Broken(Recorder):
        async def evaluate(self, state, questions, model):
            return BackendResponse({}, "jev", 10)

    with pytest.raises(ResponseValidationError):
        ask(Asker(Broken(), "jev", budget=Budget(None, 0.042)), qs(3))


def test_cache_prevents_repeat_calls_and_is_keyed_by_state_and_model(tmp_path):
    for cache in (MemoryCache(), SQLiteCache(tmp_path / "c.sqlite")):
        backend = Recorder()
        asker = Asker(backend, "jev", cache=cache, budget=Budget(None, 0.042))
        ask(asker, qs(5))
        _, m = ask(asker, qs(5))
        assert m.cached_questions == 5 and m.requests == 0
        ask(asker, qs(5), state="different")
        ask(Asker(backend, "jev-other", cache=cache, budget=Budget(None, 0.042)), qs(5))
        assert len(backend.sizes) == 3


def test_budget_blocks_requests_and_tracks_cost():
    backend = Recorder(tokens=1_000_000)
    budget = Budget(max_usd=0.05, price_per_mtok=0.042)
    asker = Asker(backend, "jev", budget=budget)
    _, m = ask(asker, qs(2))
    assert m.cost_usd == pytest.approx(0.042) and budget.spent == pytest.approx(0.042)
    with pytest.raises(BudgetExceededError):
        ask(Asker(Recorder(), "jev", budget=Budget(max_usd=0.0, price_per_mtok=0.042)), qs(1))
