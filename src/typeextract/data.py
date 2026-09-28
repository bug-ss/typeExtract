"""Result types (named after LangExtract's) and JSONL IO."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENTITY = "entity"
SENTENCE = "sentence"
FIELD = "field"


@dataclass(frozen=True)
class CharInterval:
    """Half-open character interval ``[start_pos, end_pos)`` into the source text."""

    start_pos: int
    end_pos: int


@dataclass
class Extraction:
    """One grounded extraction. Constructible like ``langextract.data.Extraction`` for examples."""

    extraction_class: str
    extraction_text: str
    char_interval: CharInterval | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    """Probability of the chosen class (or option, for fields)."""
    probabilities: dict[str, float] = field(default_factory=dict)
    """Full distribution over the options Jev chose between."""
    attribute_confidence: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    """Auxiliary probabilities: ``verify``, ``exists``, ``margin``."""
    needs_review: bool = False
    normalized: Any = None
    kind: str = ENTITY
    sources: tuple[str, ...] = ()
    """Candidate generators that proposed the span."""
    reason: str | None = None
    """Why a rejected extraction was dropped (only set in ``AnnotatedDocument.rejected``)."""

    @property
    def start(self) -> int:
        return self.char_interval.start_pos if self.char_interval else -1

    @property
    def end(self) -> int:
        return self.char_interval.end_pos if self.char_interval else -1

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["sources"] = list(self.sources)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Extraction:
        data = dict(data)
        interval = data.pop("char_interval", None)
        data["sources"] = tuple(data.get("sources", ()))
        known = {f.name for f in dataclasses.fields(cls)}
        ex = cls(**{k: v for k, v in data.items() if k in known})
        if interval is not None:
            ex.char_interval = CharInterval(interval["start_pos"], interval["end_pos"])
        return ex


@dataclass
class ExampleData:
    """A few-shot example in LangExtract's format; used to derive a schema."""

    text: str
    extractions: list[Extraction] = field(default_factory=list)


@dataclass
class Metrics:
    requests: int = 0
    questions: int = 0
    cached_questions: int = 0
    input_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    splits: int = 0
    windows: int = 0
    candidates: int = 0
    latency_s: float = 0.0

    def add(self, other: Metrics) -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


@dataclass
class AnnotatedDocument:
    text: str
    document_id: str | None = None
    extractions: list[Extraction] = field(default_factory=list)
    fields: dict[str, Extraction | None] = field(default_factory=dict)
    rejected: list[Extraction] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    model: str | None = None

    @property
    def entities(self) -> list[Extraction]:
        return [e for e in self.extractions if e.kind == ENTITY]

    @property
    def sentence_labels(self) -> list[Extraction]:
        return [e for e in self.extractions if e.kind == SENTENCE]

    def by_class(self, cls: str) -> list[Extraction]:
        return [e for e in self.extractions if e.extraction_class == cls]

    def ungrounded(self) -> list[Extraction]:
        """Extractions whose text is not the source slice. Always empty for typeextract."""
        found = [*self.extractions, *(f for f in self.fields.values() if f is not None)]
        return [e for e in found if self.text[e.start : e.end] != e.extraction_text]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "model": self.model,
            "extractions": [e.to_dict() for e in self.extractions],
            "fields": {k: (v.to_dict() if v else None) for k, v in self.fields.items()},
            "rejected": [e.to_dict() for e in self.rejected],
            "errors": self.errors,
            "metrics": dataclasses.asdict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotatedDocument:
        return cls(
            text=data["text"],
            document_id=data.get("document_id"),
            model=data.get("model"),
            extractions=[Extraction.from_dict(e) for e in data.get("extractions", [])],
            fields={
                k: (Extraction.from_dict(v) if v else None)
                for k, v in data.get("fields", {}).items()
            },
            rejected=[Extraction.from_dict(e) for e in data.get("rejected", [])],
            errors=list(data.get("errors", [])),
            metrics=Metrics(**data.get("metrics", {})),
        )


def save_jsonl(documents: Iterable[AnnotatedDocument], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> Iterator[AnnotatedDocument]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield AnnotatedDocument.from_dict(json.loads(line))
