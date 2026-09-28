"""The whole stack over HTTP against a mock Jev server that enforces the API's rules."""

import asyncio
import json

import httpx

import typeextract as tx
from typeextract.jev import JevBackend
from typeextract.testing import FakeBackend

TEXT = (
    "ACME SERVICES AGREEMENT\n"
    "This agreement is made between Acme Tecnologia Ltda. (the Client) and Beta Consultoria S.A. "
    "(the Provider). The Client shall pay R$ 180.000,00 in twelve monthly installments.\n"
    "Late payments accrue a fine of 2% per month. Contact: legal@acme.com.br.\n\n"
    + "The Provider shall deliver monthly reports to the Client. " * 40
)


class MockJev:
    """Validates every request like the real API would, then answers from a FakeBackend."""

    def __init__(self, server_max_questions=40):
        self.brain = FakeBackend(
            entities={
                "Acme Tecnologia Ltda.": "party",
                "Beta Consultoria S.A.": "party",
                "R$ 180.000,00": "amount",
                "2%": "rate",
            },
            attributes={("Acme Tecnologia Ltda.", "role"): "client", ("Beta Consultoria S.A.", "role"): "provider"},
            sentence_labels={"obligation": ["shall pay", "shall deliver"]},
            fields={"contract_value": "R$ 180.000,00"},
        )
        self.server_max_questions = server_max_questions
        self.requests = 0
        self.status_log = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.requests == 1:
            return self._log(httpx.Response(429, json={"error": "slow down"}, headers={"retry-after-ms": "5"}))
        if self.requests == 2:
            return self._log(httpx.Response(529, json={"error": "overloaded"}))
        assert request.headers["authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert set(body) == {"state", "model", "questions"} and body["model"] == "jev-1.13.0"
        questions = body["questions"]
        if len(questions) > self.server_max_questions:
            return self._log(httpx.Response(422, json={"detail": [{"msg": "too many questions in one request"}]}))
        for q in questions.values():
            assert q["type"] in ("choice", "noul", "score")
            if q["type"] == "choice":
                assert 2 <= len(q["criteria"]) <= 255
            if q["type"] == "score":
                assert 2 <= len(q["criteria"]) <= 10
        answers = {k: self.brain._answer(body["state"], q) for k, q in questions.items()}
        tokens = len(request.content) // 4
        return self._log(httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 10}}))

    def _log(self, resp):
        self.status_log.append(resp.status_code)
        return resp


SCHEMA = tx.Schema(
    description="Service agreement",
    entities=[
        tx.Entity("party", "a company or person bound by the contract, by its proper name", attributes=[tx.Attribute("role", ["client", "provider"])]),
        tx.Entity("amount", "an amount of money to be paid"),
        tx.Entity("rate", "an interest or penalty rate"),
    ],
    sentence_labels=[tx.SentenceLabel("obligation", "creates a duty for one of the parties")],
    fields=[tx.Field("contract_value", "the total value of the contract", source="money")],
)


def test_full_stack_over_http():
    server = MockJev()
    backend = JevBackend("sk-test", transport=httpx.MockTransport(server), backoff_initial=0.001)
    ex = tx.Extractor(SCHEMA, model="jev-1.13.0", backend=backend, cache=":memory:")
    doc = ex.extract(TEXT, document_id="contract-1")

    assert doc.errors == [] and doc.ungrounded() == []
    parties = {e.extraction_text: e.attributes["role"] for e in doc.by_class("party")}
    assert parties == {"Acme Tecnologia Ltda.": "client", "Beta Consultoria S.A.": "provider"}
    assert [e.extraction_text for e in doc.by_class("amount")] == ["R$ 180.000,00"]
    assert [e.extraction_text for e in doc.by_class("rate")] == ["2%"]
    assert len(doc.by_class("obligation")) == 41
    assert doc.fields["contract_value"].extraction_text == "R$ 180.000,00"
    assert doc.model == "jev-1.13.0"

    m = doc.metrics
    assert m.retries == 2  # the 429 and the 529 were retried transparently
    assert 422 not in server.status_log or m.splits >= 1  # oversized requests were split
    assert m.input_tokens > 0 and m.cost_usd > 0

    # identical rerun: served from the cache, no HTTP at all
    before = server.requests
    again = ex.extract(TEXT, document_id="contract-1")
    assert server.requests == before and again.metrics.cached_questions == again.metrics.questions
    assert [e.to_dict() for e in again.extractions] == [e.to_dict() for e in doc.extractions]


def test_many_documents_concurrently_over_http():
    server = MockJev(server_max_questions=500)
    backend = JevBackend("sk-test", transport=httpx.MockTransport(server), backoff_initial=0.001, max_concurrency=4)

    async def go():
        async with tx.Extractor(SCHEMA, model="jev-1.13.0", backend=backend, max_concurrent_documents=3) as ex:
            return [d async for d in ex.aextract_many({"text": TEXT, "document_id": str(i)} for i in range(6))]

    docs = asyncio.run(go())
    assert [d.document_id for d in docs] == [str(i) for i in range(6)]
    assert all(len(d.by_class("party")) == 2 and not d.errors for d in docs)
