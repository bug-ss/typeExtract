"""Question builders, answer parsing/validation, and the tournament that lifts the 255-option cap."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import ResponseValidationError

MAX_CHOICE_OPTIONS = 255

Question = dict[str, Any]


def choice(instructions: Any, criteria: Mapping[str, Any]) -> Question:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul(instructions: Any, true: Any = None, false: Any = None) -> Question:
    q: Question = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    return q


def score(instructions: Any, levels: Sequence[Any]) -> Question:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


@dataclass(frozen=True)
class Answer:
    """A validated answer. ``probabilities`` always covers every option of a Choice/Score."""

    type: str
    choice: str | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    noul: float | None = None
    score: float | None = None

    def top2(self) -> tuple[str | None, float, float]:
        ranked = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return None, 0.0, 0.0
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return ranked[0][0], ranked[0][1], second

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "noul": self.noul,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Answer:
        return cls(
            type=data["type"],
            choice=data.get("choice"),
            probabilities=dict(data.get("probabilities") or {}),
            confidence=data.get("confidence"),
            noul=data.get("noul"),
            score=data.get("score"),
        )


def _num(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseValidationError(f"{where}: expected a number, got {value!r}")
    v = float(value)
    if v != v:  # NaN
        raise ResponseValidationError(f"{where}: NaN")
    return min(1.0, max(0.0, v))


def _distribution(raw: Any, options: Sequence[str], where: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ResponseValidationError(f"{where}: probabilities missing")
    probs = {o: 0.0 for o in options}
    for k, v in raw.items():
        if str(k) in probs:
            probs[str(k)] = _num(v, f"{where}.probabilities[{k}]")
    total = sum(probs.values())
    if total <= 0:
        raise ResponseValidationError(f"{where}: probabilities sum to zero")
    return {k: v / total for k, v in probs.items()}


def parse_answer(raw: Any, question: Question, key: str = "?") -> Answer:
    """Validate a raw answer against the question it answers; normalise probabilities."""
    if not isinstance(raw, Mapping):
        raise ResponseValidationError(f"answer {key!r} is not an object")
    qtype = question["type"]
    if raw.get("type", qtype) != qtype:
        raise ResponseValidationError(f"answer {key!r}: type {raw.get('type')!r} != {qtype!r}")
    if qtype == "noul":
        return Answer("noul", noul=_num(raw.get("noul"), f"answer {key!r}.noul"))
    if qtype == "choice":
        options = list(question["criteria"])
        probs = _distribution(raw.get("probabilities"), options, f"answer {key!r}")
        top = max(probs, key=lambda o: probs[o])
        picked = raw.get("choice")
        if picked is not None and picked not in probs:
            raise ResponseValidationError(f"answer {key!r}: unknown option {picked!r}")
        conf = raw.get("confidence")
        return Answer(
            "choice",
            choice=top,
            probabilities=probs,
            confidence=_num(conf, f"answer {key!r}.confidence") if conf is not None else None,
        )
    if qtype == "score":
        levels = [str(i) for i in range(len(question["criteria"]))]
        probs = _distribution(raw.get("probabilities"), levels, f"answer {key!r}")
        value = raw.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = sum(int(k) * v for k, v in probs.items())
        conf = raw.get("confidence")
        return Answer(
            "score",
            probabilities=probs,
            score=float(value),
            confidence=_num(conf, f"answer {key!r}.confidence") if conf is not None else None,
        )
    raise ResponseValidationError(f"unknown question type {qtype!r}")


# --------------------------------------------------------------------------- tournament


@dataclass
class ChoiceTask:
    """A Choice over any number of options.

    Up to 254 options (plus the escape option) are asked directly. Beyond that the options are
    split into groups, each group's winner advances, and a final round decides among the winners.
    ``escape`` (e.g. ``none`` / ``unknown``) is offered in every round.
    """

    key: str
    instructions: Any
    options: dict[str, Any]
    escape: tuple[str, Any] | None = None
    max_options: int = MAX_CHOICE_OPTIONS
    # results
    result: str | None = None
    probability: float = 0.0
    margin: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    done: bool = False
    _round: list[list[str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"ChoiceTask {self.key!r} has no options")
        if self.escape and self.escape[0] in self.options:
            raise ValueError(f"option {self.escape[0]!r} collides with the escape option")
        if self.max_options - (1 if self.escape else 0) < 2:
            raise ValueError("max_options leaves fewer than 2 options per group; the tournament could not finish")
        self._round = self._groups(list(self.options))

    def _groups(self, keys: list[str]) -> list[list[str]]:
        size = self.max_options - (1 if self.escape else 0)
        return [keys[i : i + size] for i in range(0, len(keys), size)]

    def questions(self) -> dict[str, Question]:
        out = {}
        for g, group in enumerate(self._round):
            criteria = {k: self.options[k] for k in group}
            if self.escape:
                criteria[self.escape[0]] = self.escape[1]
            out[f"{self.key}~{g}" if len(self._round) > 1 else self.key] = choice(
                self.instructions, criteria
            )
        return out

    def update(self, answers: Mapping[str, Answer]) -> None:
        """Consume this round's answers; either finish or set up the next round."""
        if len(self._round) == 1:
            ans = answers[self.key if self.key in answers else f"{self.key}~0"]
            top, p1, p2 = ans.top2()
            self.result = None if self.escape and top == self.escape[0] else top
            self.probability, self.margin = p1, p1 - p2
            self.probabilities = dict(ans.probabilities)
            self.done = True
            return
        winners, escape_p = [], []
        for g in range(len(self._round)):
            ans = answers[f"{self.key}~{g}"]
            top, p1, _ = ans.top2()
            if self.escape and top == self.escape[0]:
                escape_p.append(p1)
            else:
                winners.append((p1, top))
        if not winners:
            self.result, self.probability = None, min(escape_p)
            self.probabilities = {self.escape[0]: self.probability} if self.escape else {}
            self.margin, self.done = self.probability, True
            return
        winners.sort(reverse=True)
        self._round = self._groups([w for _, w in winners])
