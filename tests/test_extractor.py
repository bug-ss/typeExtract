import asyncio
import json

import pytest

import typeextract as tx
from typeextract.errors import AuthenticationError, BudgetExceededError, ExtractionError, ServerError
from typeextract.testing import FakeBackend, failing

TEXT = (
    "Tim Cook said Apple will open a new office in Austin, Texas. The deal is worth $1.2 billion.\n"
    "Cook met Bank of America executives in New York."
)
ENTITIES = {
    "Tim Cook": "person",
    "Cook": "person",
    "Apple": "organization",
    "Austin": "location",
    "Texas": "location",
    "Bank of America": "organization",
    "America": "location",
    "New York": "location",
}


def schema(**kw):
    return tx.Schema(
        description="Business news",
        entities=[
            tx.Entity(
                "person",
                "a named person",
                attributes=[
                    tx.Attribute("role", ["executive", "politician"]),
                    tx.Attribute("quoted", kind="bool"),
                    tx.Attribute("prominence", ["minor", "notable", "famous"], kind="score"),
                ],
            ),
            tx.Entity("organization", "a named company or institution"),
            tx.Entity("location", "a named place", normalize=str.upper),
        ],
        **kw,
    )


def fake(**kw):
    kw.setdefault("entities", ENTITIES)
    return FakeBackend(**kw)


def run(sch=None, backend=None, text=TEXT, **kw):
    return tx.Extractor(sch or schema(), backend=backend or fake(), **kw).extract(text)


def test_end_to_end_grounded_entities_attributes_and_normalisation():
    backend = fake(attributes={("Tim Cook", "role"): "executive", ("Tim Cook", "quoted"): True, ("Tim Cook", "prominence"): 2})
    doc = run(backend=backend)
    got = [(e.extraction_class, e.extraction_text) for e in doc.extractions]
    assert got == [
        ("person", "Tim Cook"),
        ("organization", "Apple"),
        ("location", "Austin"),
        ("location", "Texas"),
        ("person", "Cook"),
        ("organization", "Bank of America"),
        ("location", "New York"),
    ]
    assert doc.ungrounded() == [] and doc.errors == []
    tim = doc.extractions[0]
    assert tim.attributes == {"role": "executive", "quoted": True, "prominence": 2.0}
    assert tim.char_interval == tx.CharInterval(0, 8) and tim.confidence == pytest.approx(0.9)
    assert "verify" in tim.scores and not tim.needs_review
    assert doc.by_class("location")[0].normalized == "AUSTIN"
    # "Cook" inside "Tim Cook" and "America" inside "Bank of America" lose the overlap
    assert {(r.extraction_text, r.reason) for r in doc.rejected} >= {("Cook", "overlap"), ("America", "overlap")}
    assert doc.model == "fake-1" and doc.metrics.requests >= 2 and doc.metrics.cost_usd > 0


def test_every_choice_respects_the_option_cap_and_every_request_the_question_cap():
    backend = fake()
    run(backend=backend, limits=tx.Limits(max_questions=7))
    for _state, questions in backend.calls:
        assert len(questions) <= 7
        assert all(len(q.get("criteria") or {}) <= 255 for q in questions.values())


def test_many_classes_use_the_tournament():
    classes = [tx.Entity(f"type_{i}", f"entity type number {i}") for i in range(400)]
    backend = FakeBackend(entities={"Apple": "type_399"})
    doc = run(tx.Schema(entities=classes), backend, text="We met Apple today.")
    assert [(e.extraction_class, e.extraction_text) for e in doc.extractions] == [("type_399", "Apple")]
    assert all(len(q.get("criteria") or {}) <= 255 for _, qs in backend.calls for q in qs.values())


def test_overlap_modes():
    nested = run(overlap="nested").extractions
    assert {e.extraction_text for e in nested} >= {"Bank of America", "America"}  # different classes may nest
    assert [e.start for e in nested if e.extraction_text == "Cook"] == [93]  # same class may not
    doc = run(overlap="all")
    assert sum(e.extraction_text == "Cook" for e in doc.extractions) == 2


def test_verification_and_thresholds_reject_with_reasons():
    doc = run(backend=fake(reject=["Apple"]))
    assert "Apple" not in {e.extraction_text for e in doc.extractions}
    assert ("Apple", "verification") in {(r.extraction_text, r.reason) for r in doc.rejected}
    low = run(backend=fake(confidence=0.4))
    assert low.extractions == [] and all(r.reason == "low_confidence" for r in low.rejected)
    no_verify = run(backend=fake(reject=["Apple"]), verify=False)
    assert "Apple" in {e.extraction_text for e in no_verify.extractions}


def test_review_flags():
    doc = run(backend=fake(confidence=0.6), review_threshold=0.7)
    assert doc.extractions and all(e.needs_review for e in doc.extractions)


def test_sentence_labels_and_fields():
    sch = schema(
        sentence_labels=[tx.SentenceLabel("announcement", "announces a future action")],
        fields=[
            tx.Field("deal_value", "the value of the deal", source="money", normalize=lambda s: s.lstrip("$")),
            tx.Field("ceo", "the chief executive quoted", source="person"),
            tx.Field("launch_date", "the launch date", source="date"),
            tx.Field("missing", "the stock ticker", patterns=[r"\b[A-Z]{4}\b"]),
        ],
    )
    backend = fake(sentence_labels={"announcement": ["will open"]}, fields={"deal_value": "$1.2 billion", "ceo": "Tim Cook"})
    doc = run(sch, backend, text=TEXT + " Ticker AAPL is up.")
    labels = doc.sentence_labels
    assert [e.extraction_text for e in labels] == ["Tim Cook said Apple will open a new office in Austin, Texas."]
    assert doc.fields["deal_value"].extraction_text == "$1.2 billion"
    assert doc.fields["deal_value"].normalized == "1.2 billion"
    assert doc.fields["ceo"].extraction_text == "Tim Cook" and doc.fields["ceo"].start == 0
    assert doc.fields["launch_date"] is None  # no date in the text: no call made
    assert doc.fields["missing"] is None
    assert doc.ungrounded() == []


def test_field_with_hundreds_of_candidates_and_long_document():
    amounts = [f"${i}.00" for i in range(1, 700)]
    text = " ".join(f"Line {i} costs {a}." for i, a in enumerate(amounts))
    sch = tx.Schema(fields=[tx.Field("total", "the grand total", source="money")])
    backend = FakeBackend(fields={"total": "$654.00"})
    doc = tx.Extractor(sch, backend=backend, field_state_chars=5_000).extract(text)
    assert doc.fields["total"].extraction_text == "$654.00"
    assert all(len(q.get("criteria") or {}) <= 255 for _, qs in backend.calls for q in qs.values())


def test_field_not_stated_is_none_and_audited():
    sch = tx.Schema(fields=[tx.Field("total", "the grand total", source="money")])
    doc = tx.Extractor(sch, backend=FakeBackend()).extract("Tax was $5.00.")
    assert doc.fields["total"] is None


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_empty_documents_make_no_calls(text):
    backend = fake()
    doc = run(backend=backend, text=text)
    assert doc.extractions == [] and backend.calls == [] and doc.metrics.requests == 0


def test_errors_are_isolated_per_window_with_skip():
    text = "\n\n".join(["Tim Cook visited Austin."] * 6)
    backend = fake(fail=failing(1, ServerError("overloaded", 529)))
    doc = run(backend=backend, text=text, on_error="skip", window_chars=30)
    assert len(doc.errors) == 1 and doc.errors[0]["stage"] == "window"
    assert len(doc.by_class("person")) == 5


def test_errors_raise_with_partial_document():
    with pytest.raises(ExtractionError) as info:
        run(backend=fake(fail=failing(1, ServerError("down", 503))))
    assert isinstance(info.value.document, tx.AnnotatedDocument)


def test_fatal_errors_stop_even_when_skipping():
    with pytest.raises(AuthenticationError):
        run(backend=fake(fail=failing(99, AuthenticationError("bad key", 401))), on_error="skip")
    with pytest.raises(BudgetExceededError):
        run(max_cost_usd=0.0, on_error="skip")


def test_cache_makes_reruns_free_and_identical(tmp_path):
    backend = fake()
    ex = tx.Extractor(schema(), backend=backend, cache=tmp_path / "cache.sqlite")
    first = ex.extract(TEXT)
    calls = len(backend.calls)
    second = ex.extract(TEXT)
    assert len(backend.calls) == calls and second.metrics.requests == 0
    assert [e.to_dict() for e in first.extractions] == [e.to_dict() for e in second.extractions]


def test_repeated_words_are_disambiguated_without_counting():
    backend = fake(entities={"Paris": "location"})
    doc = run(backend=backend, text="Paris is not Paris, Texas.")
    assert [e.start for e in doc.extractions] == [0, 13]
    instr = [q["instructions"] for _, qs in backend.calls for q in qs.values() if isinstance(q["instructions"], dict)]
    assert any("second of 2 occurrences" in i.get("occurrence", "") for i in instr)


def test_windows_reference_sentences_by_key():
    backend = fake()
    run(backend=backend)
    state, questions = backend.calls[0]
    assert isinstance(state["text"], dict) and set(state["text"]) == {"S1", "S2", "S3"}
    assert state["entity_types"]["person"] == "a named person"
    assert any("`text.S3`" in q["instructions"]["question"] for q in questions.values())


def test_extract_many_is_ordered_and_accepts_several_input_shapes():
    class Doc:
        def __init__(self, text, document_id):
            self.text, self.document_id = text, document_id

    ex = tx.Extractor(schema(), backend=fake(), max_concurrent_documents=2)
    docs = ex.extract_many(["Tim Cook.", {"text": "Apple.", "document_id": "b"}, Doc("Austin.", "c"), "Texas."])
    assert [d.document_id for d in docs] == ["0", "b", "c", "3"]
    assert [d.extractions[0].extraction_text for d in docs] == ["Tim Cook", "Apple", "Austin", "Texas"]
    assert ex.metrics.requests == sum(d.metrics.requests for d in docs)


def test_sync_api_works_inside_a_running_event_loop():
    async def notebook_cell():
        return tx.Extractor(schema(), backend=fake()).extract("Tim Cook.")

    assert asyncio.run(notebook_cell()).extractions[0].extraction_text == "Tim Cook"


def test_async_api():
    async def go():
        async with tx.Extractor(schema(), backend=fake()) as ex:
            return [d async for d in ex.aextract_many(["Apple.", "Austin."])]

    assert [d.extractions[0].extraction_text for d in asyncio.run(go())] == ["Apple", "Austin"]


def test_langextract_style_call_with_examples(monkeypatch):
    examples = [
        tx.ExampleData(
            text="ROMEO. But soft! What light through yonder window breaks?",
            extractions=[
                tx.Extraction("character", "ROMEO", attributes={"emotional_state": "wonder"}),
                tx.Extraction("emotion", "But soft!", attributes={"feeling": "gentle awe"}),
            ],
        )
    ]
    backend = FakeBackend(entities={"JULIET": "character"}, attributes={("JULIET", "emotional_state"): "wonder"})
    doc = tx.extract(
        "JULIET. O Romeo, Romeo!",
        prompt_description="Extract characters and emotions.",
        examples=examples,
        backend=backend,
    )
    juliet = doc.by_class("character")[0]
    assert juliet.extraction_text == "JULIET" and juliet.attributes["emotional_state"] == "wonder"
    state = backend.calls[0][0]
    assert state["guidelines"] == "Extract characters and emotions."
    assert state["entity_types"]["character"]["examples"] == ["ROMEO"]


def test_normaliser_errors_do_not_break_extraction():
    def boom(_):
        raise ValueError("nope")

    sch = tx.Schema(entities=[tx.Entity("location", "a place", normalize=boom)])
    doc = run(sch, fake(), text="Austin.")
    assert doc.extractions[0].normalized is None


def test_io_roundtrip_and_html(tmp_path):
    doc = run(backend=fake(fields={}))
    path = tmp_path / "out.jsonl"
    tx.save_jsonl([doc], path)
    (back,) = list(tx.load_jsonl(path))
    assert [e.to_dict() for e in back.extractions] == [e.to_dict() for e in doc.extractions]
    assert json.loads(path.read_text())["metrics"]["requests"] == doc.metrics.requests
    html = tx.to_html(run(backend=fake(), text="<b>Tim Cook</b> & Apple"))
    assert "&lt;b&gt;" in html and "<mark" in html and "<b>Tim" not in html
    assert '<mark data-class="person"' in html and '<tr data-class="person">' in html  # styleable per class


def test_invalid_code_points_keep_offsets():
    doc = run(backend=fake(), text="Tim Cook \udcff met Apple.")
    assert [e.extraction_text for e in doc.extractions] == ["Tim Cook", "Apple"]
    assert doc.text[doc.extractions[1].start : doc.extractions[1].end] == "Apple"
    json.dumps(doc.to_dict(), ensure_ascii=False).encode("utf-8")  # serialisable


def test_one_extractor_is_safe_to_share_between_threads():
    from concurrent.futures import ThreadPoolExecutor

    ex = tx.Extractor(schema(), backend=fake())
    with ThreadPoolExecutor(8) as pool:
        docs = list(pool.map(ex.extract, ["Tim Cook."] * 16))
    assert all(d.extractions[0].extraction_text == "Tim Cook" for d in docs)
    assert ex.metrics.requests == sum(d.metrics.requests for d in docs)


def test_field_only_schema_skips_span_classification():
    backend = FakeBackend(fields={"total": "$9.00"})
    doc = tx.Extractor(tx.Schema(fields=[tx.Field("total", "the total", source="money")]), backend=backend).extract(
        "Tim Cook paid $9.00 in Austin."
    )
    assert doc.fields["total"].extraction_text == "$9.00" and doc.metrics.candidates == 0
    assert len(backend.calls) == 1


LONG = "The patient has acute chronic obstructive pulmonary disease and takes metformin."


def test_jev_span_source_finds_what_rules_miss():
    ents = {"acute chronic obstructive pulmonary disease": "condition", "metformin": "drug"}
    sch = tx.Schema(entities=[tx.Entity("condition", "a disease"), tx.Entity("drug", "a medicine")])
    rules = tx.Extractor(sch, backend=FakeBackend(entities=ents)).extract(LONG)
    assert "acute chronic obstructive pulmonary disease" not in {e.extraction_text for e in rules.extractions}

    backend = FakeBackend(entities=ents)
    jev = tx.Extractor(sch, backend=backend, span_source="jev").extract(LONG)
    assert [(e.extraction_class, e.extraction_text) for e in jev.extractions] == [
        ("condition", "acute chronic obstructive pulmonary disease"),
        ("drug", "metformin"),
    ]
    assert jev.ungrounded() == [] and jev.extractions[0].sources == ("jev_tagger",)
    word_questions = [q for _, qs in backend.calls for q in qs.values() if "word" in q["instructions"]]
    assert len(word_questions) == 11  # one per word

    hybrid = tx.Extractor(sch, backend=FakeBackend(entities=ents), span_source="hybrid").extract(LONG)
    assert {e.extraction_text for e in hybrid.extractions} == set(ents)


def test_span_source_is_validated():
    with pytest.raises(tx.ConfigurationError):
        tx.Extractor(schema(), backend=fake(), span_source="magic")


# ------------------------------------------------------------------ span_source review regressions

GEO = tx.Schema(
    entities=[
        tx.Entity("location", "a named place"),
        tx.Entity("organization", "a named company or institution"),
        tx.Entity("amount", "an amount of money"),
        tx.Entity("rate", "a percentage rate"),
        tx.Entity("email", "an email address"),
        tx.Entity("ticket", "a ticket number", patterns=[r"#\d+"]),
        tx.Entity("drug", "a medicine", terms=["metformin"]),
    ]
)


def jev_run(text, entities, mode="jev", **kw):
    backend = FakeBackend(entities=entities)
    return tx.Extractor(GEO, backend=backend, span_source=mode, **kw).extract(text), backend


@pytest.mark.parametrize("mode", ["jev", "hybrid"])
def test_tagger_keeps_list_items_separate(mode):
    ents = {c: "location" for c in ["Paris", "London", "Berlin", "Rome", "Austin", "Texas"]}
    doc, _ = jev_run("We visited Paris, London, Berlin and Rome, then Austin, Texas.", ents, mode)
    assert [e.extraction_text for e in doc.extractions] == ["Paris", "London", "Berlin", "Rome", "Austin", "Texas"]


def test_tagger_handles_table_rows_cjk_and_nesting():
    doc, _ = jev_run("Paris   London   Berlin", {"Paris": "location", "London": "location", "Berlin": "location"})
    assert [e.extraction_text for e in doc.extractions] == ["Paris", "London", "Berlin"]
    doc, _ = jev_run("我们去了北京上海。", {"北京": "location", "上海": "location"})
    assert [e.extraction_text for e in doc.extractions] == ["北京", "上海"]
    doc, _ = jev_run("The Bank of America office.", {"Bank of America": "organization", "America": "location"}, overlap="nested")
    assert [(e.extraction_class, e.extraction_text) for e in doc.extractions] == [
        ("organization", "Bank of America"),
        ("location", "America"),
    ]


def test_jev_mode_keeps_symbols_and_the_users_own_candidate_sources():
    ents = {"$1.2 billion": "amount", "12%": "rate", "bob@acme.com": "email", "#4521": "ticket", "metformin": "drug"}
    text = "It cost $1.2 billion, up 12%; mail bob@acme.com about #4521 and metformin."
    doc, _ = jev_run(text, ents)
    assert {e.extraction_text for e in doc.extractions} == set(ents)
    # the ticket pattern and the drug term still propose their spans even if Jev tags nothing
    quiet = FakeBackend(entities=ents)
    quiet._answer = lambda state, q, _orig=quiet._answer: (
        {"type": "noul", "noul": 0.0} if "word" in (q.get("instructions") or {}) else _orig(state, q)
    )
    doc = tx.Extractor(GEO, backend=quiet, span_source="jev").extract(text)
    assert {"#4521", "metformin", "$1.2 billion", "12%", "bob@acme.com"} <= {e.extraction_text for e in doc.extractions}


def test_no_sentence_final_period_variants():
    doc, backend = jev_run("The patient takes metformin.", {"metformin": "drug"}, mode="hybrid")
    assert [e.extraction_text for e in doc.extractions] == ["metformin"]
    asked = [q["instructions"]["span"] for _, qs in backend.calls for q in qs.values() if "span" in q["instructions"]]
    assert "metformin." not in asked


def test_tag_threshold_is_a_recall_knob_and_is_validated():
    class Unsure(FakeBackend):  # "is this word part of a mention?" -> 0.45 for every word
        def _answer(self, state, q):
            if "word" in (q.get("instructions") or {}):
                return {"type": "noul", "noul": 0.45}
            return super()._answer(state, q)

    doc = tx.Extractor(GEO, backend=Unsure(entities={"Rome": "location"}), span_source="jev", tag_threshold=0.4).extract("I love Rome.")
    assert [e.extraction_text for e in doc.extractions] == ["Rome"]
    doc = tx.Extractor(GEO, backend=Unsure(entities={"Rome": "location"}), span_source="jev", tag_threshold=0.5).extract("I love Rome.")
    assert doc.extractions == []
    for bad in (None, -0.1, 1.5, "0.3"):
        with pytest.raises(tx.ConfigurationError):
            tx.Extractor(GEO, backend=FakeBackend(), span_source="jev", tag_threshold=bad)


def test_hybrid_survives_a_failed_tagging_round():
    def fail_tagging(_call, questions):
        if any("word" in (q.get("instructions") or {}) for q in questions.values()):
            raise ServerError("overloaded", 529)

    backend = FakeBackend(entities={"Rome": "location", "Paris": "location"}, fail=fail_tagging)
    doc = tx.Extractor(GEO, backend=backend, span_source="hybrid", on_error="skip").extract("Rome and Paris.")
    assert [e.extraction_text for e in doc.extractions] == ["Rome", "Paris"]
    assert [err["stage"] for err in doc.errors] == ["tag"]


def test_hybrid_metrics_sources_and_any_fields():
    sch = tx.Schema(
        entities=[tx.Entity("condition", "a disease"), tx.Entity("drug", "a medicine")],
        fields=[tx.Field("main_condition", "the main condition", source="any")],
    )
    ents = {"acute chronic obstructive pulmonary disease": "condition", "metformin": "drug"}
    backend = FakeBackend(entities=ents, fields={"main_condition": "acute chronic obstructive pulmonary disease"})
    doc = tx.Extractor(sch, backend=backend, span_source="hybrid").extract(LONG)
    classified = sum(
        1 for _, qs in backend.calls for q in qs.values()
        if q["type"] == "choice" and "span" in q["instructions"] and "entity_type" not in q["instructions"]
    )
    assert doc.metrics.candidates == classified
    by_text = {e.extraction_text: e for e in doc.extractions}
    assert "jev_tagger" in by_text["metformin"].sources and "ngram" in by_text["metformin"].sources
    assert doc.fields["main_condition"].extraction_text == "acute chronic obstructive pulmonary disease"


def test_hybrid_tags_and_classifies_concurrently():
    class Slow(FakeBackend):
        in_flight = peak = 0

        async def evaluate(self, state, questions, model):
            Slow.in_flight += 1
            Slow.peak = max(Slow.peak, Slow.in_flight)
            await asyncio.sleep(0.05)
            Slow.in_flight -= 1
            return await super().evaluate(state, questions, model)

    backend = Slow(entities={"Rome": "location"})
    doc = tx.Extractor(GEO, backend=backend, span_source="hybrid", verify=False).extract("I love Rome.")
    assert [e.extraction_text for e in doc.extractions] == ["Rome"]
    assert Slow.peak == 2  # the tagging request overlapped the code candidates' classification


def test_failing_custom_generator_follows_on_error():
    def broken(text, start, end):
        raise RuntimeError("spaCy model not loaded")

    ex = tx.Extractor(schema(), backend=fake(), candidate_generators=[broken], on_error="skip")
    docs = ex.extract_many(["Tim Cook.", "Apple."])
    assert [d.extractions[0].extraction_text for d in docs] == ["Tim Cook", "Apple"]
    assert all(d.errors == [{"stage": "candidates", "where": "broken", "error": "RuntimeError: spaCy model not loaded"}] for d in docs)
    with pytest.raises(ExtractionError):
        tx.Extractor(schema(), backend=fake(), candidate_generators=[broken]).extract("Tim Cook.")


def test_fields_get_guidelines_and_typed_kind_rules():
    sch = tx.Schema(
        instructions="Amounts are in thousands.",
        fields=[tx.Field("count", "the number of units", source="number")],
    )
    backend = FakeBackend(fields={"count": "40"})
    doc = tx.Extractor(sch, backend=backend).extract("Shipped March 3, 2026 with 40 units.")
    assert doc.fields["count"].extraction_text == "40"
    state, questions = backend.calls[0]
    assert state["guidelines"] == "Amounts are in thousands."
    values = {o["value"] for q in questions.values() if q["type"] == "choice" for o in q["criteria"].values() if isinstance(o, dict)}
    assert values == {"40"}  # "3" and "2026" (inside the date) are not offered


def test_word_occurrence_hints_are_whole_words():
    backend = FakeBackend(entities={"cat": "location"})
    tx.Extractor(GEO, backend=backend, span_source="jev").extract("A man and a woman had a cat.")
    hints = {
        (q["instructions"]["word"], q["instructions"].get("occurrence"))
        for _, qs in backend.calls for q in qs.values() if "word" in (q.get("instructions") or {})
    }
    assert ("man", None) in hints  # not "inside" woman
    assert ("a", "the first of 2 occurrences in `text`") in hints and ("a", "the second of 2 occurrences in `text`") in hints


def test_html_view_scales_to_thousands_of_extractions():
    import time as _time

    text = " ".join(f"w{i}" for i in range(4000))
    doc = tx.AnnotatedDocument(text=text)
    pos = 0
    for i in range(4000):
        word = f"w{i}"
        doc.extractions.append(tx.Extraction("x", word, tx.CharInterval(pos, pos + len(word)), confidence=0.9))
        pos += len(word) + 1
    t0 = _time.perf_counter()
    html = tx.to_html(doc)
    assert _time.perf_counter() - t0 < 1.5 and html.count("<mark") == 4000


@pytest.mark.parametrize("mode", ["jev", "hybrid"])
def test_long_mentions_joined_by_a_connector_are_both_found(mode):
    ents = {"acute chronic obstructive pulmonary disease": "condition", "type 2 diabetes": "condition"}
    sch = tx.Schema(entities=[tx.Entity("condition", "a disease")])
    doc = tx.Extractor(sch, backend=FakeBackend(entities=ents), span_source=mode).extract(
        "Admitted with acute chronic obstructive pulmonary disease and type 2 diabetes."
    )
    assert [e.extraction_text for e in doc.extractions] == list(ents)
