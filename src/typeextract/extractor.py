"""The extraction pipeline: candidates from code, decisions from Jev, policy in code."""

from __future__ import annotations

import asyncio
import bisect
import logging
import math
import re
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .asker import Asker, Budget, Limits
from .cache import Cache, open_cache
from .data import ENTITY, FIELD, SENTENCE, AnnotatedDocument, CharInterval, ExampleData, Extraction, Metrics
from .errors import FATAL_ERRORS, ConfigurationError, ExtractionError, TypeExtractError
from .jev import DEFAULT_MODEL, PRICE_PER_MTOK, Backend, JevBackend
from .questions import Answer, ChoiceTask, Question, noul, score
from .schema import NONE, UNKNOWN, Attribute, Entity, Field, Schema
from .text import (
    BUILTIN_PATTERNS,
    TOKEN_RE,
    STOPWORDS,
    Candidate,
    CustomGenerator,
    Window,
    context_after,
    context_before,
    find_spans,
    generate_candidates,
    make_windows,
    snippet,
    split_sentences,
)

log = logging.getLogger("typeextract")
T = TypeVar("T")

NONE_DESC = (
    "Not exactly one complete mention of any of these entity types: a generic word, only part of "
    "a longer mention, or the span includes words that are not part of the mention."
)
UNKNOWN_DESC = "The text does not say or clearly imply it."
FIELD_NONE_DESC = "None of the options is the value of the field."
VERIFY_TRUE = "The span is a complete mention that fits the definition, with nothing missing and nothing extra."
VERIFY_FALSE = (
    "The definition does not cover it (for example a generic word, role, pronoun or label), it is "
    "only part of a mention, or it includes words that are not part of the mention."
)
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OVERLAP_MODES = ("none", "nested", "all")
SPAN_SOURCES = ("rules", "jev", "hybrid")
TAG_NONE_DESC = "Not part of a mention of any of these entity types."
_TAG_GAP = frozenset({"", "&", "-", "–", "/", ".", ",", "'", "’"})
_SURROGATES = re.compile("[\ud800-\udfff]")


def run_sync(factory: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine from sync code, even when an event loop is already running (notebooks,
    web handlers): in that case it runs on a private loop in a worker thread."""

    async def main() -> T:
        return await factory()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, main()).result()


def resolve_overlaps(
    items: Sequence[Extraction], mode: str = "none", tie: float = 0.1
) -> tuple[list[Extraction], list[Extraction]]:
    """Keep the most probable spans; at near-ties (same ``tie`` bucket) the longer span wins.

    ``mode="none"``: no two kept spans overlap. ``"nested"``: a span may sit fully inside a span
    of another class ("America" inside "Bank of America"). ``"all"``: keep everything.
    """
    if mode == "all":
        return list(items), []
    ordered = sorted(
        items,
        key=lambda e: (-math.floor((e.confidence or 0.0) / tie + 1e-9), -(e.end - e.start), e.start),
    )
    kept: list[Extraction] = []
    dropped: list[Extraction] = []
    for e in ordered:
        clash = False
        for k in kept:
            if e.start < k.end and k.start < e.end:
                nested = (k.start <= e.start and e.end <= k.end) or (e.start <= k.start and k.end <= e.end)
                if mode == "nested" and nested and k.extraction_class != e.extraction_class:
                    continue
                clash = True
                break
        (dropped if clash else kept).append(e)
    return kept, dropped


@dataclass
class _Input:
    text: str
    document_id: str | None


def _as_input(item: Any, index: int) -> _Input:
    if isinstance(item, str):
        return _Input(item, str(index))
    if isinstance(item, Mapping):
        return _Input(item["text"], item.get("document_id", str(index)))
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return _Input(text, getattr(item, "document_id", None) or str(index))
    raise TypeError(f"cannot extract from {type(item).__name__}: pass str, {{'text': ...}} or an object with .text")


class Extractor:
    """Reusable, thread-safe extractor for one schema.

    >>> ex = Extractor(schema, model="jev-1.13.0", cache=True)
    >>> doc = ex.extract(text)
    >>> for e in doc.extractions: print(e.extraction_class, e.extraction_text, e.start, e.end, e.confidence)
    """

    def __init__(
        self,
        schema: Schema,
        *,
        model: str = DEFAULT_MODEL,
        backend: Backend | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: Cache | str | Path | bool | None = None,
        limits: Limits | None = None,
        window_chars: int = 800,
        context_chars: int = 250,
        max_sentence_chars: int = 600,
        max_ngram: int = 4,
        max_candidates_per_sentence: int = 60,
        ngrams: bool = True,
        candidate_generators: Sequence[CustomGenerator] = (),
        stopwords: str | Collection[str] = "en",
        verify: bool = True,
        verify_threshold: float = 0.5,
        overlap: str = "none",
        review_margin: float = 0.15,
        review_threshold: float = 0.7,
        field_state_chars: int = 30_000,
        max_cost_usd: float | None = None,
        price_per_mtok: float = PRICE_PER_MTOK,
        on_error: str = "raise",
        max_concurrent_documents: int = 4,
        span_source: str = "rules",
        tag_threshold: float = 0.3,
    ):
        if not isinstance(schema, Schema):
            raise ConfigurationError("schema must be a typeextract.Schema (see Schema.load / from_dict)")
        if overlap not in OVERLAP_MODES:
            raise ConfigurationError(f"overlap must be one of {OVERLAP_MODES}")
        if span_source not in SPAN_SOURCES:
            raise ConfigurationError(f"span_source must be one of {SPAN_SOURCES}")
        if on_error not in ("raise", "skip"):
            raise ConfigurationError("on_error must be 'raise' or 'skip'")
        if min(window_chars, max_sentence_chars, max_ngram, max_candidates_per_sentence) < 1:
            raise ConfigurationError("window_chars, max_sentence_chars, max_ngram and max_candidates_per_sentence must be >= 1")
        if isinstance(stopwords, str):
            if stopwords not in STOPWORDS:
                raise ConfigurationError(f"unknown stopword list {stopwords!r}; use {sorted(STOPWORDS)} or a set")
            stopwords = STOPWORDS[stopwords]
        self.schema = schema
        self.model = model
        self.backend: Backend = backend if backend is not None else JevBackend(api_key, base_url=base_url)
        self.cache = open_cache(cache)
        self.limits = limits or Limits()
        self.budget = Budget(max_cost_usd, price_per_mtok)
        self.asker = Asker(self.backend, model, limits=self.limits, cache=self.cache, budget=self.budget)
        self.window_chars = window_chars
        self.context_chars = context_chars
        self.max_sentence_chars = max_sentence_chars
        self.max_ngram = max_ngram
        self.max_candidates_per_sentence = max_candidates_per_sentence
        self.ngrams = ngrams
        self.candidate_generators = tuple(candidate_generators)
        self.stopwords = frozenset(stopwords)
        self.verify = verify
        self.verify_threshold = verify_threshold
        self.overlap = overlap
        self.review_margin = review_margin
        self.review_threshold = review_threshold
        self.field_state_chars = field_state_chars
        self.on_error = on_error
        self.span_source = span_source
        self.tag_threshold = tag_threshold
        self.max_concurrent_documents = max(1, max_concurrent_documents)
        self.metrics = Metrics()
        """Cumulative metrics over every document this extractor has processed."""
        self._metrics_lock = threading.Lock()
        self._class_patterns = {e.id: e.compiled_patterns for e in schema.entities if e.compiled_patterns}
        self._gazetteers = {e.id: [*e.examples, *e.terms] for e in schema.entities if e.examples or e.terms}

    # ------------------------------------------------------------------ public API

    def extract(self, text: str, document_id: str | None = None) -> AnnotatedDocument:
        return run_sync(lambda: self._closing(self.aextract(text, document_id)))

    def extract_many(self, documents: Iterable[Any]) -> list[AnnotatedDocument]:
        async def collect() -> list[AnnotatedDocument]:
            return [doc async for doc in self.aextract_many(documents)]

        return run_sync(lambda: self._closing(collect()))

    async def aextract(self, text: str, document_id: str | None = None) -> AnnotatedDocument:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        started = time.perf_counter()
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:  # lone surrogates from a bad decode; one-for-one, offsets unchanged
            log.warning("document %s contains invalid code points; replaced with U+FFFD", document_id)
            text = _SURROGATES.sub("\ufffd", text)
        doc = AnnotatedDocument(text=text, document_id=document_id)
        if text.strip():
            await _Run(self, doc).execute()
        doc.metrics.latency_s = time.perf_counter() - started
        doc.model = self.asker.served_model or self.model
        with self._metrics_lock:
            self.metrics.add(doc.metrics)
        return doc

    async def aextract_many(self, documents: Iterable[Any]) -> AsyncIterator[AnnotatedDocument]:
        """Yield documents in input order, with at most ``max_concurrent_documents`` in flight.
        Works for unbounded iterables (only the in-flight documents are held in memory)."""
        pending: deque[asyncio.Future[AnnotatedDocument]] = deque()
        try:
            for i, item in enumerate(documents):
                inp = _as_input(item, i)
                pending.append(asyncio.ensure_future(self.aextract(inp.text, inp.document_id)))
                if len(pending) >= self.max_concurrent_documents:
                    yield await pending.popleft()
            while pending:
                yield await pending.popleft()
        finally:
            for fut in pending:
                fut.cancel()

    async def aclose(self) -> None:
        await self.backend.aclose()
        close = getattr(self.cache, "close", None)
        if close:
            close()

    async def __aenter__(self) -> Extractor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __enter__(self) -> Extractor:
        return self

    def __exit__(self, *exc: Any) -> None:
        close = getattr(self.cache, "close", None)
        if close:
            close()

    async def _closing(self, coro: Awaitable[T]) -> T:
        # sync calls run on a short-lived loop: release its connection pool before it closes
        try:
            return await coro
        finally:
            await self.backend.aclose()

    # ------------------------------------------------------------------ shared helpers

    async def run_tasks(
        self,
        state: Any,
        tasks: Sequence[ChoiceTask],
        extra: Mapping[str, Question],
        metrics: Metrics,
    ) -> dict[str, Answer]:
        """Ask ``tasks`` (tournament Choices) plus ``extra`` questions about one state.
        The first round carries everything; later rounds only the tournaments still open."""
        pending = list(tasks)
        extra_answers: dict[str, Answer] = {}
        first = True
        while first or pending:
            questions: dict[str, Question] = {}
            for t in pending:
                questions.update(t.questions())
            if first:
                questions.update(extra)
            if not questions:
                break
            answers = await self.asker.ask(state, questions, metrics)
            if first:
                extra_answers = {k: answers[k] for k in extra}
            for t in pending:
                t.update(answers)
            pending = [t for t in pending if not t.done]
            first = False
        return extra_answers

    def needs_review(self, probability: float, margin: float) -> bool:
        return probability < self.review_threshold or margin < self.review_margin


class _Run:
    """State for one document."""

    def __init__(self, ex: Extractor, doc: AnnotatedDocument):
        self.ex = ex
        self.schema = ex.schema
        self.doc = doc
        self.text = doc.text
        self.sentences = split_sentences(self.text, ex.max_sentence_chars)
        self.windows = make_windows(self.sentences, ex.window_chars)
        needed = bool(self.schema.entities) or any("any" in f.source for f in self.schema.fields)
        self.candidates = [[] for _ in self.sentences] if not needed else generate_candidates(
            self.text,
            self.sentences,
            class_patterns=ex._class_patterns,
            gazetteers=ex._gazetteers,
            stopwords=ex.stopwords,
            max_ngram=ex.max_ngram,
            max_per_sentence=ex.max_candidates_per_sentence,
            ngrams=ex.ngrams,
            custom=ex.candidate_generators,
        )
        doc.metrics.windows = len(self.windows)
        doc.metrics.candidates = sum(len(c) for c in self.candidates)

    async def execute(self) -> None:
        entity_ids = {e.id for e in self.schema.entities}
        late = [f for f in self.schema.fields if set(f.source) & entity_ids]
        early = [f for f in self.schema.fields if f not in late]
        await _gather(
            *(self._guard("window", f"window {w.index}", self._window(w)) for w in self.windows),
            *(self._guard("field", f.id, self._field(f)) for f in early),
        )
        await _gather(*(self._guard("field", f.id, self._field(f)) for f in late))
        self.doc.extractions.sort(key=lambda e: (e.start, -e.end, e.kind, e.extraction_class))

    async def _guard(self, stage: str, where: str, coro: Awaitable[None]) -> None:
        try:
            await coro
        except FATAL_ERRORS:
            raise
        except TypeExtractError as exc:
            if self.ex.on_error == "raise":
                raise ExtractionError(f"{where}: {exc}", document=self.doc) from exc
            log.warning("skipping %s: %s", where, exc)
            self.doc.errors.append({"stage": stage, "where": where, "error": f"{type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------------ windows

    def _ref(self, w: Window, sentence: int) -> str:
        return "`text`" if len(w.sentences) == 1 else f"`text.S{w.sentences.index(sentence) + 1}`"

    def _state(self, w: Window) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.schema.description:
            state["document_type"] = self.schema.description
        if self.schema.instructions:
            state["guidelines"] = self.schema.instructions
        if self.schema.entities:
            state["entity_types"] = {e.id: e.definition() for e in self.schema.entities}
        before = context_before(self.text, w.start, self.ex.context_chars)
        if before:
            state["preceding_text"] = before
        spans = [self.sentences[i] for i in w.sentences]
        state["text"] = (
            self.text[spans[0][0] : spans[0][1]]
            if len(spans) == 1
            else {f"S{k + 1}": self.text[s:e] for k, (s, e) in enumerate(spans)}
        )
        after = context_after(self.text, w.end, self.ex.context_chars)
        if after:
            state["following_text"] = after
        return state

    def _span_instructions(
        self, w: Window, c: Candidate, question: str, name: str = "span", **fields: Any
    ) -> dict[str, Any]:
        ref = self._ref(w, c.sentence)
        out: dict[str, Any] = {name: c.text, **fields, "question": question.replace("{ref}", ref)}
        s_start, s_end = self.sentences[c.sentence]
        sentence_text = self.text[s_start:s_end]
        positions, at = [], sentence_text.find(c.text)
        while at != -1:
            positions.append(s_start + at)
            at = sentence_text.find(c.text, at + 1)
        if len(positions) > 1:  # the same words appear twice: say which one, never by counting
            n = positions.index(c.start) if c.start in positions else 0
            ordinal = _ORDINALS[n] if n < len(_ORDINALS) else f"#{n + 1}"
            out["occurrence"] = f"the {ordinal} of {len(positions)} occurrences in {ref}"
            out["context"] = snippet(self.text, c.start, c.end, 25)
        return out

    async def _window(self, w: Window) -> None:
        ex, schema, metrics = self.ex, self.schema, self.doc.metrics
        cands = [c for i in w.sentences for c in self.candidates[i]]
        state = self._state(w)
        if schema.entities and ex.span_source != "rules":
            tagged = await self._tag_spans(w, state)
            if ex.span_source == "jev":
                cands = tagged
            else:  # hybrid: union, rules first
                seen = {(c.start, c.end) for c in cands}
                cands = cands + [c for c in tagged if (c.start, c.end) not in seen]

        # round 1: one Choice per candidate (+ one Noul per sentence and sentence label)
        tasks: list[ChoiceTask] = []
        if schema.entities:
            options = {e.id: None for e in schema.entities}
            for i, c in enumerate(cands):
                q = (
                    "Which entity type in `entity_types` is `span` exactly one complete mention of, "
                    "as it is used in {ref}?"
                )
                tasks.append(
                    ChoiceTask(
                        f"c{i}",
                        self._span_instructions(w, c, q),
                        dict(options),
                        escape=(NONE, NONE_DESC),
                        max_options=ex.limits.max_choice_options,
                    )
                )
        label_q: dict[str, Question] = {}
        for i in w.sentences:
            for j, label in enumerate(schema.sentence_labels):
                label_q[f"s{i}_{j}"] = noul(
                    {
                        "sentence_type": label.id,
                        "definition": label.description,
                        "question": f"Is {self._ref(w, i)} a `sentence_type` sentence, as described in `definition`?",
                    }
                )
        if not tasks and not label_q:
            return
        label_answers = await ex.run_tasks(state, tasks, label_q, metrics)

        for i in w.sentences:
            s, e = self.sentences[i]
            for j, label in enumerate(schema.sentence_labels):
                p = label_answers[f"s{i}_{j}"].noul or 0.0
                if p >= schema.threshold(label):
                    self.doc.extractions.append(
                        Extraction(
                            label.id,
                            self.text[s:e],
                            CharInterval(s, e),
                            confidence=p,
                            needs_review=p < ex.review_threshold,
                            kind=SENTENCE,
                        )
                    )

        accepted: list[tuple[Candidate, Entity, ChoiceTask]] = []
        for c, t in zip(cands, tasks):
            if t.result is None:
                continue
            entity = schema.entity(t.result)
            if t.probability >= schema.threshold(entity):
                accepted.append((c, entity, t))
            else:
                self.doc.rejected.append(self._extraction(c, entity, t, reason="low_confidence"))
        if not accepted:
            return

        # round 2: verification + attributes (asked speculatively for every accepted span)
        extra: dict[str, Question] = {}
        attr_tasks: list[ChoiceTask] = []
        for j, (c, entity, _) in enumerate(accepted):
            if ex.verify:
                extra[f"v{j}"] = noul(
                    self._span_instructions(
                        w,
                        c,
                        'As it is used in {ref}, is `span` exactly one complete mention of the entity type "'
                        + entity.id
                        + '" defined in `entity_types`?',
                        entity_type=entity.id,
                    ),
                    true=VERIFY_TRUE,
                    false=VERIFY_FALSE,
                )
            for k, attr in enumerate(entity.attributes):
                key = f"a{j}_{k}"
                instr = self._span_instructions(w, c, self._attribute_question(entity, attr), entity_type=entity.id, attribute=attr.name)
                if attr.kind == "choice":
                    attr_tasks.append(
                        ChoiceTask(
                            key,
                            instr,
                            dict(attr.options),  # type: ignore[arg-type]
                            escape=(UNKNOWN, UNKNOWN_DESC),
                            max_options=ex.limits.max_choice_options,
                        )
                    )
                elif attr.kind == "bool":
                    extra[key] = noul(instr)
                else:
                    extra[key] = score(instr, list(attr.options))  # type: ignore[arg-type]
        answers = await ex.run_tasks(state, attr_tasks, extra, metrics) if (extra or attr_tasks) else {}
        by_key = {t.key: t for t in attr_tasks}

        found: list[Extraction] = []
        for j, (c, entity, t) in enumerate(accepted):
            e = self._extraction(c, entity, t)
            if ex.verify:
                v = answers[f"v{j}"].noul or 0.0
                e.scores["verify"] = v
                if v < ex.verify_threshold:
                    e.reason = "verification"
                    self.doc.rejected.append(e)
                    continue
                e.needs_review = e.needs_review or v < ex.review_threshold
            for k, attr in enumerate(entity.attributes):
                key = f"a{j}_{k}"
                if attr.kind == "choice":
                    at = by_key[key]
                    e.attributes[attr.name] = at.result
                    e.attribute_confidence[attr.name] = at.probability
                elif attr.kind == "bool":
                    p = answers[key].noul or 0.0
                    e.attributes[attr.name] = p >= 0.5
                    e.attribute_confidence[attr.name] = p if p >= 0.5 else 1 - p
                else:
                    a = answers[key]
                    e.attributes[attr.name] = a.score
                    e.attribute_confidence[attr.name] = a.confidence if a.confidence is not None else a.top2()[1]
            found.append(e)
        kept, dropped = resolve_overlaps(found, ex.overlap)
        for d in dropped:
            d.reason = "overlap"
        self.doc.extractions.extend(kept)
        self.doc.rejected.extend(dropped)

    async def _tag_spans(self, w: Window, state: dict[str, Any]) -> list[Candidate]:
        """Let Jev find the spans: one Choice per word ("which entity type is this word part of,
        or none?"), then runs of words tagged with the same type become candidates.
        No capitalisation, stopword or length heuristics; costs one question per word."""
        ex = self.ex
        options = {e.id: None for e in self.schema.entities}
        words: list[Candidate] = []
        for i in w.sentences:
            ss, se = self.sentences[i]
            words += [Candidate(m.start(), m.end(), m.group(), i) for m in TOKEN_RE.finditer(self.text, ss, se)]
        q = "As it is used in {ref}, which entity type in `entity_types` is `word` part of?"
        tasks = [
            ChoiceTask(
                f"t{k}",
                self._span_instructions(w, word, q, name="word"),
                dict(options),
                escape=(NONE, TAG_NONE_DESC),
                max_options=ex.limits.max_choice_options,
            )
            for k, word in enumerate(words)
        ]
        await ex.run_tasks(state, tasks, {}, self.doc.metrics)

        spans: dict[tuple[int, int], Candidate] = {}

        def emit(run: list[Candidate], cls: str) -> None:
            s, e = run[0].start, run[-1].end
            for end in (e, e + 1) if self.text[e : e + 1] == "." else (e,):  # "Ltd." / "Ltd"
                if (s, end) not in spans:
                    spans[(s, end)] = Candidate(s, end, self.text[s:end], run[0].sentence, {"jev_tagger"}, set(), {cls}, 0)

        run: list[Candidate] = []
        run_cls: str | None = None
        for word, task in zip(words, tasks):
            cls = task.result if task.result and task.probability >= ex.tag_threshold else None
            joins = (
                run
                and cls == run_cls
                and word.sentence == run[-1].sentence
                and self.text[run[-1].end : word.start].strip() in _TAG_GAP
            )
            if joins:
                run.append(word)
                continue
            if run and run_cls:
                emit(run, run_cls)
            run, run_cls = ([word], cls) if cls else ([], None)
        if run and run_cls:
            emit(run, run_cls)
        return sorted(spans.values(), key=lambda c: (c.start, c.end))

    @staticmethod
    def _attribute_question(entity: Entity, attr: Attribute) -> str:
        what = attr.name.replace("_", " ") + (f" ({attr.description})" if attr.description else "")
        if attr.kind == "bool":
            return f"As it is used in {{ref}}, is the {entity.id} `span` {what}?"
        if attr.kind == "score":
            return f"As it is used in {{ref}}, where does the {entity.id} `span` fall on {what}?"
        return f"As it is used in {{ref}}, which option is the {what} of the {entity.id} `span`?"

    def _extraction(self, c: Candidate, entity: Entity, t: ChoiceTask, reason: str | None = None) -> Extraction:
        e = Extraction(
            entity.id,
            c.text,
            CharInterval(c.start, c.end),
            confidence=t.probability,
            probabilities={k: v for k, v in t.probabilities.items()},
            scores={"margin": t.margin},
            needs_review=self.ex.needs_review(t.probability, t.margin),
            kind=ENTITY,
            sources=tuple(sorted(c.sources)),
            reason=reason,
        )
        if entity.normalize is not None and reason is None:
            e.normalized = _safe(entity.normalize, c.text)
        return e

    # ------------------------------------------------------------------ fields

    def _field_spans(self, f: Field) -> list[tuple[int, int]]:
        spans: set[tuple[int, int]] = set()
        for src in f.source:
            if src == "any":
                spans.update((c.start, c.end) for cs in self.candidates for c in cs)
            elif src in BUILTIN_PATTERNS:
                spans.update(find_spans(self.text, [BUILTIN_PATTERNS[src]]))
            else:
                spans.update((e.start, e.end) for e in self.doc.extractions if e.extraction_class == src)
        spans.update(find_spans(self.text, f.compiled_patterns))
        return sorted(spans)

    def _chunks(self) -> list[tuple[int, int]]:
        """Contiguous regions of at most ``field_state_chars`` (sentence-aligned)."""
        limit = self.ex.field_state_chars
        if len(self.text) <= limit:
            return [(0, len(self.text))]
        chunks: list[tuple[int, int]] = []
        for s, e in self.sentences:
            if chunks and e - chunks[-1][0] <= limit:
                chunks[-1] = (chunks[-1][0], e)
            else:
                chunks.append((s, e))
        return chunks

    def _field_question(self, f: Field, where: str) -> dict[str, Any]:
        return {
            "field": f.id,
            "definition": f.description,
            "question": f"Which option is the value of `field` (as described in `definition`) in {where}?",
        }

    async def _field(self, f: Field) -> None:
        ex, metrics = self.ex, self.doc.metrics
        spans = self._field_spans(f)
        if not spans:
            self.doc.fields[f.id] = None
            return
        index = {sp: i for i, sp in enumerate(spans)}
        base: dict[str, Any] = {"document_type": self.schema.description} if self.schema.description else {}

        def options(items: Sequence[tuple[int, int]]) -> dict[str, Any]:
            return {
                f"o{index[sp]}": {"value": self.text[sp[0] : sp[1]], "context": snippet(self.text, sp[0], sp[1])}
                for sp in items
            }

        def new_task(items: Sequence[tuple[int, int]]) -> ChoiceTask:
            return ChoiceTask(
                "f",
                self._field_question(f, "`text`"),
                options(items),
                escape=(NONE, FIELD_NONE_DESC),
                max_options=ex.limits.max_choice_options,
            )

        # Round 1, per chunk of text: a (tournament) Choice over the chunk's candidates, plus
        # a Noul asking whether the chunk states the field at all (a Choice always ranks something).
        chunks = self._chunks()
        starts = [cs for cs, _ in chunks]
        groups: dict[int, list[tuple[int, int]]] = {}
        for sp in spans:
            groups.setdefault(max(0, bisect.bisect_right(starts, sp[0]) - 1), []).append(sp)
        exists_q = noul(
            {
                "field": f.id,
                "definition": f.description,
                "question": "Does `text` state the value of `field` (as described in `definition`)?",
            }
        )
        jobs = []
        for k, inside in sorted(groups.items()):
            cs = min(chunks[k][0], inside[0][0])
            ce = max(chunks[k][1], max(e for _, e in inside))
            task = new_task(inside)
            jobs.append((task, ex.run_tasks({**base, "text": self.text[cs:ce]}, [task], {"exists": exists_q}, metrics)))
        answers = await _gather(*(coro for _, coro in jobs))
        exist_p = max(a["exists"].noul or 0.0 for a in answers)
        winners = [(spans[int(t.result[1:])], t) for t, _ in jobs if t.result is not None]

        # Final round across chunks, on excerpts around each chunk's winner.
        final: ChoiceTask | None = winners[0][1] if len(winners) == 1 else None
        if len(winners) > 1:
            final = new_task([sp for sp, _ in winners])
            excerpts = {f"E{k + 1}": snippet(self.text, sp[0], sp[1], 300) for k, (sp, _) in enumerate(winners)}
            await ex.run_tasks({**base, "text": excerpts}, [final], {}, metrics)
        if final is None or final.result is None:
            self.doc.fields[f.id] = None
            return

        def label(key: str) -> str:
            if key == NONE:
                return NONE
            a, b = spans[int(key[1:])]
            return f"{self.text[a:b]} @{a}"

        sp = spans[int(final.result[1:])]
        e = Extraction(
            f.id,
            self.text[sp[0] : sp[1]],
            CharInterval(*sp),
            confidence=final.probability,
            probabilities={label(k): v for k, v in final.probabilities.items()},
            scores={"exists": exist_p, "margin": final.margin},
            needs_review=ex.needs_review(final.probability, final.margin) or exist_p < ex.review_threshold,
            kind=FIELD,
        )
        if exist_p < f.exists_threshold or final.probability < self.schema.threshold(f):
            e.reason = "not_stated" if exist_p < f.exists_threshold else "low_confidence"
            self.doc.rejected.append(e)
            self.doc.fields[f.id] = None
            return
        if f.normalize is not None:
            e.normalized = _safe(f.normalize, e.extraction_text)
        self.doc.fields[f.id] = e


def _safe(fn: Callable[[str], Any], value: str) -> Any:
    try:
        return fn(value)
    except Exception as exc:  # a user normaliser must never break extraction
        log.warning("normalize(%r) failed: %s", value, exc)
        return None


async def _gather(*aws: Awaitable[Any]) -> list[Any]:
    """Like asyncio.gather, but waits for everything before raising the first error
    (so no task is left running unobserved)."""
    results = await asyncio.gather(*aws, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results


# ---------------------------------------------------------------------- LangExtract-style API


def _schema_from(
    schema: Schema | Mapping[str, Any] | str | Path | None,
    prompt_description: str | None,
    examples: Sequence[ExampleData] | None,
) -> Schema:
    if schema is None:
        if not examples:
            raise ConfigurationError("pass a schema, or LangExtract-style prompt_description + examples")
        return Schema.from_examples(examples, prompt_description or "")
    if isinstance(schema, Schema):
        return schema
    if isinstance(schema, Mapping):
        return Schema.from_dict(schema)
    return Schema.load(schema)


def extract(
    text_or_documents: str | Iterable[Any],
    schema: Schema | Mapping[str, Any] | str | Path | None = None,
    *,
    prompt_description: str | None = None,
    examples: Sequence[ExampleData] | None = None,
    model_id: str = DEFAULT_MODEL,
    api_key: str | None = None,
    **options: Any,
) -> AnnotatedDocument | list[AnnotatedDocument]:
    """One-call extraction, shaped like ``langextract.extract``.

    ``schema`` may be a :class:`Schema`, a dict, or a path to a JSON/YAML file; alternatively pass
    ``prompt_description`` and LangExtract ``examples`` and the schema is derived from them.
    """
    ex = Extractor(_schema_from(schema, prompt_description, examples), model=model_id, api_key=api_key, **options)
    with ex:
        if isinstance(text_or_documents, str):
            return ex.extract(text_or_documents)
        return ex.extract_many(text_or_documents)


async def aextract(
    text: str,
    schema: Schema | Mapping[str, Any] | str | Path | None = None,
    *,
    prompt_description: str | None = None,
    examples: Sequence[ExampleData] | None = None,
    model_id: str = DEFAULT_MODEL,
    api_key: str | None = None,
    **options: Any,
) -> AnnotatedDocument:
    ex = Extractor(_schema_from(schema, prompt_description, examples), model=model_id, api_key=api_key, **options)
    async with ex:
        return await ex.aextract(text)
