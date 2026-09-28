"""A deterministic, offline stand-in for Jev, for testing schemas and pipelines without a network.

>>> fake = FakeBackend(entities={"Tim Cook": "person", "Apple": "organization"})
>>> ex = Extractor(schema, backend=fake)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .errors import APIError
from .jev import BackendResponse

_REF_RE = re.compile(r"`(text(?:\.S\d+)?)`")


class FakeBackend:
    """Answers the questions typeextract asks from simple lookup tables.

    * ``entities``: span text -> class id (classification and verification);
    * ``attributes``: (span text, attribute name) -> value (option, bool, or score level index);
    * ``sentence_labels``: label id -> substrings; a sentence containing one carries the label;
    * ``fields``: field id -> the exact value text;
    * ``reject``: spans that pass classification but fail verification.

    ``fail`` may raise an ``APIError`` (or return None) per call to simulate outages.
    Every call is recorded in ``calls`` as ``(state, questions)``.
    """

    name = "fake"

    def __init__(
        self,
        entities: Mapping[str, str] | None = None,
        attributes: Mapping[tuple[str, str], Any] | None = None,
        sentence_labels: Mapping[str, Sequence[str]] | None = None,
        fields: Mapping[str, str] | None = None,
        reject: Sequence[str] = (),
        confidence: float = 0.9,
        fail: Callable[[int, dict[str, Any]], None] | None = None,
        model: str = "fake-1",
        report_tokens: bool = True,
    ):
        self.entities = dict(entities or {})
        self.attributes = dict(attributes or {})
        self.sentence_labels = {k: list(v) for k, v in (sentence_labels or {}).items()}
        self.fields = dict(fields or {})
        self.reject = set(reject)
        self.confidence = confidence
        self.fail = fail
        self.model = model
        self.report_tokens = report_tokens
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def aclose(self) -> None:
        return None

    async def evaluate(self, state: Any, questions: dict[str, dict[str, Any]], model: str) -> BackendResponse:
        self.calls.append((state, questions))
        if self.fail is not None:
            self.fail(len(self.calls), questions)
        answers = {key: self._answer(state, q) for key, q in questions.items()}
        tokens = (len(str(state)) + len(str(questions))) // 4 if self.report_tokens else None
        return BackendResponse(answers=answers, model=self.model, input_tokens=tokens)

    # ------------------------------------------------------------------ answering

    def _choice(self, criteria: Mapping[str, Any], pick: str | None) -> dict[str, Any]:
        options = list(criteria)
        if pick not in criteria:
            pick = next((o for o in ("none", "unknown") if o in criteria), options[0])
        rest = (1 - self.confidence) / max(1, len(options) - 1)
        probs = {o: (self.confidence if o == pick else rest) for o in options}
        return {"type": "choice", "choice": pick, "probabilities": probs, "confidence": self.confidence}

    def _noul(self, yes: bool) -> dict[str, Any]:
        return {"type": "noul", "noul": self.confidence if yes else 1 - self.confidence}

    @staticmethod
    def _sentence(state: Any, instructions: Mapping[str, Any]) -> str:
        text = state.get("text") if isinstance(state, Mapping) else state
        m = _REF_RE.search(str(instructions.get("question", "")))
        if isinstance(text, Mapping):
            key = m.group(1).split(".", 1)[1] if m and "." in m.group(1) else next(iter(text))
            return str(text.get(key, ""))
        return str(text or "")

    def _answer(self, state: Any, q: dict[str, Any]) -> dict[str, Any]:
        qtype = q["type"]
        instr = q.get("instructions") if isinstance(q.get("instructions"), Mapping) else {}
        span = instr.get("span")
        if "field" in instr:
            value = self.fields.get(instr["field"])
            if qtype == "noul":
                text = state.get("text") if isinstance(state, Mapping) else state
                return self._noul(value is not None and value in str(text))
            pick = next(
                (k for k, v in q["criteria"].items() if isinstance(v, Mapping) and v.get("value") == value),
                None,
            )
            return self._choice(q["criteria"], pick)
        if "sentence_type" in instr:
            sentence = self._sentence(state, instr)
            subs = self.sentence_labels.get(instr["sentence_type"], [])
            return self._noul(any(s in sentence for s in subs))
        if "word" in instr:  # span tagging: which entity type is this word part of?
            sentence = self._sentence(state, instr)
            word = instr["word"]
            hits = [
                (len(ent), cls)
                for ent, cls in self.entities.items()
                if ent in sentence and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", ent)
            ]
            return self._choice(q["criteria"], max(hits)[1] if hits else None)
        if "attribute" in instr:
            value = self.attributes.get((span, instr["attribute"]))
            if qtype == "noul":
                return self._noul(bool(value))
            if qtype == "score":
                n = len(q["criteria"])
                level = int(value) if isinstance(value, int) else 0
                probs = {str(i): (1.0 if i == level else 0.0) for i in range(n)}
                return {"type": "score", "score": float(level), "legend": {}, "probabilities": probs, "confidence": 1.0}
            return self._choice(q["criteria"], value)
        if span is not None:
            cls = self.entities.get(span)
            if qtype == "noul":  # verification
                return self._noul(cls is not None and cls == instr.get("entity_type") and span not in self.reject)
            return self._choice(q["criteria"], cls)
        if qtype == "choice":
            return self._choice(q["criteria"], None)
        if qtype == "score":
            n = len(q["criteria"])
            return {"type": "score", "score": 0.0, "legend": {}, "probabilities": {str(i): float(i == 0) for i in range(n)}, "confidence": 1.0}
        return self._noul(False)


def failing(times: int, error: APIError) -> Callable[[int, dict[str, Any]], None]:
    """A ``fail`` hook for :class:`FakeBackend` that raises ``error`` on the first ``times`` calls."""

    def hook(call: int, _questions: dict[str, Any]) -> None:
        if call <= times:
            raise error

    return hook
