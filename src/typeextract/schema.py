"""What to extract: entities (spans), sentence labels, fields (single-valued slots)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data import ExampleData
from .errors import TypeExtractError
from .text import PATTERN_KINDS

NONE = "none"
UNKNOWN = "unknown"
RESERVED = frozenset({NONE, UNKNOWN, "any"})
ATTRIBUTE_KINDS = ("choice", "bool", "score")


class SchemaError(TypeExtractError, ValueError):
    """The schema is invalid."""


def _check_id(kind: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{kind} id must be a non-empty string, got {value!r}")
    if value.strip().lower() in RESERVED:
        raise SchemaError(f"{kind} id {value!r} is reserved")
    if len(value) > 64:
        raise SchemaError(f"{kind} id {value!r} is longer than 64 characters")
    return value.strip()


def _check_description(owner: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{owner} needs a description")
    return value


def _strings(value: Any, owner: str) -> tuple[str, ...]:
    """A list of strings; a lone string is one item (not one item per character)."""
    if value is None:
        return ()
    items = (value,) if isinstance(value, str) else tuple(value)
    if not all(isinstance(v, str) for v in items):
        raise SchemaError(f"{owner} must be strings")
    return items


def _build(cls: type, data: Mapping[str, Any], where: str) -> Any:
    try:
        return cls(**data)
    except TypeError as exc:  # a missing or unknown key in a dict/YAML schema
        raise SchemaError(f"{where}: {exc}") from exc


def _check_threshold(name: str, value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise SchemaError(f"{name} must be between 0 and 1, got {value}")


def _compile(patterns: Sequence[str], owner: str) -> tuple[re.Pattern[str], ...]:
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error as exc:
            raise SchemaError(f"invalid regex {p!r} in {owner}: {exc}") from exc
    return tuple(compiled)


@dataclass
class Attribute:
    """A closed-set property of an extracted entity.

    ``kind="choice"`` picks one of ``options`` (a list, or a mapping option -> description);
    ``kind="bool"`` is a yes/no; ``kind="score"`` places the entity on 2-10 ordered ``options``
    (levels) and returns a float that may land between levels.
    Jev cannot write text, so free-form attributes are not possible; normalise in code instead.
    """

    name: str
    options: Sequence[str] | Mapping[str, str | None] | None = None
    kind: str = "choice"
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError(f"attribute name must be a non-empty string, got {self.name!r}")
        if self.kind not in ATTRIBUTE_KINDS:
            raise SchemaError(f"attribute {self.name!r}: kind must be one of {ATTRIBUTE_KINDS}")
        if self.kind == "bool":
            if self.options:
                raise SchemaError(f"bool attribute {self.name!r} takes no options")
            return
        if not self.options:
            raise SchemaError(
                f"attribute {self.name!r} needs options: Jev chooses among options and cannot "
                "write free text (use kind='bool', or normalise the span in code)"
            )
        if self.kind == "choice":
            opts = (
                dict(self.options)
                if isinstance(self.options, Mapping)
                else {str(o): None for o in self.options}
            )
            if len(opts) != len(self.options):
                raise SchemaError(f"attribute {self.name!r} has duplicate options")
            if UNKNOWN in opts:
                raise SchemaError(f"attribute {self.name!r}: 'unknown' is added automatically")
            self.options = opts
        else:
            levels = list(self.options)
            if not 2 <= len(levels) <= 10:
                raise SchemaError(f"score attribute {self.name!r} needs 2-10 levels")
            self.options = [str(level) for level in levels]


@dataclass
class Entity:
    """A span class. ``description`` is the whole specification Jev sees; say what is excluded."""

    id: str
    description: str
    examples: Sequence[str] = ()
    """Shown to Jev in the definition and proposed as candidates when they occur verbatim."""
    counter_examples: Sequence[str] = ()
    terms: Sequence[str] = ()
    """A gazetteer of known surface forms, proposed as candidates (not sent to Jev)."""
    patterns: Sequence[str] = ()
    """Regexes that propose candidates for this class (group 1 if present, else the match)."""
    attributes: Sequence[Attribute] = ()
    min_confidence: float | None = None
    normalize: Callable[[str], Any] | None = None

    def __post_init__(self) -> None:
        self.id = _check_id("entity", self.id)
        _check_description(f"entity {self.id!r}", self.description)
        for name in ("examples", "counter_examples", "terms", "patterns"):
            setattr(self, name, _strings(getattr(self, name), f"entity {self.id!r} {name}"))
        _check_threshold(f"entity {self.id!r} min_confidence", self.min_confidence)
        names = [a.name for a in self.attributes]
        if len(set(names)) != len(names):
            raise SchemaError(f"entity {self.id!r} has duplicate attribute names")
        self.compiled_patterns = _compile(self.patterns, f"entity {self.id!r}")

    def definition(self) -> str | dict[str, Any]:
        if not self.examples and not self.counter_examples:
            return self.description
        out: dict[str, Any] = {"definition": self.description}
        if self.examples:
            out["examples"] = list(self.examples)[:8]
        if self.counter_examples:
            out["not_this_type"] = list(self.counter_examples)[:8]
        return out


@dataclass
class SentenceLabel:
    """A sentence-level class (e.g. an obligation clause). A sentence may carry several labels."""

    id: str
    description: str
    min_confidence: float | None = None

    def __post_init__(self) -> None:
        self.id = _check_id("sentence label", self.id)
        _check_description(f"sentence label {self.id!r}", self.description)
        _check_threshold(f"sentence label {self.id!r} min_confidence", self.min_confidence)


@dataclass
class Field:
    """A single-valued slot ("the total due"). Jev picks one of the candidate spans, or none.

    ``source`` says where the options come from: ``"any"`` (every candidate), a built-in pattern
    kind (``"money"``, ``"date"``, ...), or an entity id (that class's extractions).
    """

    id: str
    description: str
    source: str | Sequence[str] = "any"
    patterns: Sequence[str] = ()
    min_confidence: float | None = None
    exists_threshold: float = 0.5
    normalize: Callable[[str], Any] | None = None

    def __post_init__(self) -> None:
        self.id = _check_id("field", self.id)
        _check_description(f"field {self.id!r}", self.description)
        self.patterns = _strings(self.patterns, f"field {self.id!r} patterns")
        self.source = (self.source,) if isinstance(self.source, str) else tuple(self.source)
        if not self.source and not self.patterns:
            raise SchemaError(f"field {self.id!r} needs a source or patterns")
        _check_threshold(f"field {self.id!r} min_confidence", self.min_confidence)
        _check_threshold(f"field {self.id!r} exists_threshold", self.exists_threshold)
        self.compiled_patterns = _compile(self.patterns, f"field {self.id!r}")


@dataclass
class Schema:
    entities: Sequence[Entity] = ()
    description: str = ""
    """What the documents are ("Service agreement", "Clinical note"). Sent to Jev as context."""
    instructions: str = ""
    """Extra guidelines, like LangExtract's ``prompt_description``."""
    sentence_labels: Sequence[SentenceLabel] = ()
    fields: Sequence[Field] = field(default_factory=tuple)
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        self.entities = tuple(self.entities)
        self.sentence_labels = tuple(self.sentence_labels)
        self.fields = tuple(self.fields)
        if not (self.entities or self.sentence_labels or self.fields):
            raise SchemaError("schema needs at least one entity, sentence label or field")
        _check_threshold("min_confidence", self.min_confidence)
        for kind, items in (
            ("entity", self.entities),
            ("sentence label", self.sentence_labels),
            ("field", self.fields),
        ):
            ids = [i.id for i in items]
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise SchemaError(f"duplicate {kind} ids: {sorted(dupes)}")
        entity_ids = {e.id for e in self.entities}
        for f in self.fields:
            for src in f.source:
                if src != "any" and src not in PATTERN_KINDS and src not in entity_ids:
                    raise SchemaError(
                        f"field {f.id!r}: unknown source {src!r}; use 'any', an entity id, or "
                        f"one of {sorted(PATTERN_KINDS)}"
                    )

    def entity(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise KeyError(entity_id)

    def threshold(self, item: Entity | SentenceLabel | Field) -> float:
        return item.min_confidence if item.min_confidence is not None else self.min_confidence

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Schema:
        def items(raw: Any, key: str) -> list[dict[str, Any]]:
            if raw is None:
                return []
            if isinstance(raw, Mapping):  # {id: description} or {id: {...}}
                return [
                    {"id": k, **(v if isinstance(v, Mapping) else {"description": v})}
                    for k, v in raw.items()
                ]
            if isinstance(raw, list):
                return [dict(x) for x in raw]
            raise SchemaError(f"{key} must be a list or a mapping")

        def attributes(raw: Any) -> list[Attribute]:
            out = []
            for a in items(raw, "attributes") if isinstance(raw, Mapping) else (raw or []):
                a = dict(a)
                if "id" in a and "name" not in a:
                    a["name"] = a.pop("id")
                desc = a.get("description")
                if isinstance(desc, list):  # shorthand: {role: [client, provider]}
                    a["options"] = a.pop("description")
                out.append(_build(Attribute, a, f"attribute {a.get('name')!r}"))
            return out

        entities = []
        for e in items(data.get("entities"), "entities"):
            e["attributes"] = attributes(e.get("attributes"))
            entities.append(_build(Entity, e, f"entity {e.get('id')!r}"))
        return cls(
            entities=entities,
            description=data.get("description", ""),
            instructions=data.get("instructions", ""),
            sentence_labels=[
                _build(SentenceLabel, s, f"sentence label {s.get('id')!r}")
                for s in items(data.get("sentence_labels"), "sentence_labels")
            ],
            fields=[_build(Field, f, f"field {f.get('id')!r}") for f in items(data.get("fields"), "fields")],
            min_confidence=data.get("min_confidence", 0.5),
        )

    @classmethod
    def load(cls, path: str | Path) -> Schema:
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise SchemaError("YAML schemas need PyYAML: pip install typeextract[yaml]") from exc
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        if not isinstance(data, Mapping):
            raise SchemaError(f"{path} must contain a mapping")
        return cls.from_dict(data)

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[ExampleData],
        prompt_description: str = "",
        description: str = "",
    ) -> Schema:
        """Derive a schema from LangExtract-style few-shot examples.

        Classes come from ``extraction_class``; example texts become examples and candidates;
        attribute values seen in the examples become the closed option set of that attribute.
        """
        order: list[str] = []
        texts: dict[str, list[str]] = {}
        attrs: dict[str, dict[str, list[Any]]] = {}
        for example in examples:
            for ex in example.extractions:
                name = ex.extraction_class
                if name not in texts:
                    order.append(name)
                    texts[name], attrs[name] = [], {}
                if ex.extraction_text and ex.extraction_text not in texts[name]:
                    texts[name].append(ex.extraction_text)
                for key, value in (ex.attributes or {}).items():
                    values = value if isinstance(value, list) else [value]
                    seen = attrs[name].setdefault(key, [])
                    seen.extend(v for v in values if v not in seen)
        if not order:
            raise SchemaError("examples contain no extractions")
        entities = []
        for name in order:
            attributes = []
            for key, values in attrs[name].items():
                if values and all(isinstance(v, bool) for v in values):
                    attributes.append(Attribute(key, kind="bool"))
                else:
                    opts = list(dict.fromkeys(str(v) for v in values if v is not None))
                    if opts:
                        attributes.append(Attribute(key, opts))
            entities.append(
                Entity(
                    id=name,
                    description=f'a mention of "{name.replace("_", " ")}" as described in the guidelines',
                    examples=texts[name],
                    attributes=attributes,
                )
            )
        return cls(entities=entities, description=description, instructions=prompt_description)
