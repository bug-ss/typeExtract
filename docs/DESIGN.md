# typeextract design

typeextract does what LangExtract does (schema in, grounded extractions with character
offsets out) but uses a **typed-decision model (Jev)** instead of a generative LLM. Jev never
writes text. It only answers `Choice`, `Noul` and `Score` questions with probabilities. So the
framework turns extraction from *generation* into *selection*:

> Code enumerates what the answer could be (spans with exact offsets). Jev decides which of
> those options is right. Code assembles the result.

Every extraction is therefore a substring of the input by construction:
`text[e.start:e.end] == e.extraction_text` always holds.

## Pipeline

```
text ─► segment ─► windows ─► candidates ─► round 1: classify ─► round 2: verify + attributes ─► resolve ─► document
                                                  │                                                     ▲
                                                  └──── sentence labels ───────────────────────────────┘
          fields (single-valued slots): candidates ─► locate (tournament) + exists ──────────────────────┘
```

1. **Segment** (`text.py`). Sentences with exact offsets, never modifying the text. Blank lines are
   hard breaks. Hard-wrapped lines are re-joined unless the line ends a sentence, is short (a
   heading or list item) or the next line starts with a bullet. Abbreviations, initials and
   decimals do not end sentences. Over-long sentences (tables, logs) are force-split at clause
   punctuation or whitespace.
2. **Window**. Consecutive sentences are grouped into windows of about 800 characters. One
   window is one Jev `state`, sent with a little text before and after as context. Sentences are
   keyed `S1..Sn` so questions can point at them by path (`text.S2`).
3. **Candidates** (`text.py`). Recall is decided here, deterministically:
   - typed patterns (money, percent, dates, times, emails, URLs, phones, quantities, IDs, numbers);
   - per-entity regexes and gazetteers (`patterns`, `examples`, `terms`);
   - proper-noun runs (including connectors like "of"/"de" and "Inc."-style suffixes);
   - content n-grams that do not start or end on a stopword or cross punctuation;
   - quoted text;
   - custom generators (a hook for spaCy, GLiNER or anything else).

   Candidates are de-duplicated by offsets and capped per sentence by priority.

   With `span_source="jev"`, Jev finds the spans instead. It gets one `Choice` per word: *which
   entity type is this word part of, or `none`?*. Runs of words with the same type (allowing
   `&`, `-`, `/`, `.`, `,` between them) become candidates. `"hybrid"` merges both sources.
   Either way, the candidates still go through rounds 1 and 2.
4. **Round 1: classify.** For every candidate there is one `Choice`: *which entity type is this
   exact span a complete mention of, or `none`?* A second set of questions, one `Noul` per
   sentence and sentence label, asks *is this sentence an X?*. All of a window's questions go
   out together (speculative fan-out) and are packed into as few requests as the limits allow.
5. **Round 2: verify + attributes.** This round runs only for spans accepted in round 1. It asks
   one verification `Noul` per span (*is it exactly one complete mention that fits the
   definition?*). It also asks, speculatively, each attribute of the span's class: `Choice`,
   `Noul` (bool) or `Score`.
6. **Resolve** (policy, in code). Spans are accepted by per-class threshold, and ones that fail
   verification are dropped. Overlaps are resolved by probability; near-ties prefer the longer
   span. Close calls get `needs_review`. Rejected spans are kept with a reason for auditing.
7. **Fields** (single-valued slots such as "the total due"). This is where "point at the index"
   is the right question. The field's candidates (a pattern kind, an entity class, regexes, or
   everything) become the **options** of a `Choice`: *which option is X?*. A `Noul` in the same
   request asks whether the text states X at all.

## Jev shortcomings and how the framework handles them

| Jev limitation | Mitigation |
|---|---|
| Cannot generate text, cannot find spans on its own | Candidates come from code, and Jev only chooses among them. Output is offsets into the source, so there is nothing to hallucinate and nothing to align. |
| A `Choice` has at most 255 options | A **tournament** (`ChoiceTask`) splits options into groups of 254 plus `none`, takes each group's winner, then runs a final round. The same mechanism covers entity classes, attribute options and field candidates. |
| A `Choice` has one winner, and its probabilities sum to 1, so several entities compete | Extraction asks *what is this span?* once per candidate, not *where is the entity?*, so any number of mentions can be found. *Where* questions are used only for single-valued fields, paired with an existence `Noul` (since a Choice always ranks something first). |
| Cannot count and is weak with numeric positions | Never asks for an index. Spans are quoted inside structured instructions, and sentences are referenced by key (`text.S2`). A repeated span gets an explicit occurrence number and snippet. Offsets live in code. |
| 64k tokens per request, 32k for state plus the longest question | A token estimator and **packer** split a window's questions across as many requests as needed, all with the same state. If the server still rejects a request as too large, the limits are lowered and the rejected batch is **re-packed** under them (bisected as a fallback) and retried; later requests use the lower limits. Windows keep states small. |
| Accuracy drops with large, irrelevant state ("context rot") | Small windows (about 800 characters) with bounded neighbour context. Class definitions are sent once in the state, not repeated in every question. |
| Literal reading and generic mentions ("the Client") | `none` is defined explicitly ("generic, only part of a mention, or extra words"). Definitions carry examples and counter-examples. Accepted spans are re-checked with a class-aware verification `Noul`. |
| Probabilities are not exact, can be over-confident, and vary run to run | Per-class thresholds, and `needs_review` from the top-2 margin and low confidence. A content-addressed answer **cache** makes reruns reproducible and free. Model versions can be pinned. |
| Rate limits (1,200 req/min, 250k tokens/s, changing dynamically), 429 and 529 responses | Client-side token bucket for requests and tokens. A global cooldown honours `retry-after` / `retry-after-ms`. Exponential backoff with jitter, and bounded concurrency. |
| Transient failures (timeouts, 5xx, connection resets) | Retries on 408, 429, 5xx, 529, timeouts and connection errors. Per-window error isolation (`on_error="skip"`) keeps partial results and records the errors. Authentication and budget errors always stop the run. |
| Malformed or partial responses | Every answer is validated: its type matches, the chosen option exists, and probabilities are normalised. Missing answers are re-asked once, then reported as an error. |
| No normalisation or free-text attributes | Attributes are closed sets (`Choice`), booleans (`Noul`) or ordinal scales (`Score`). Value normalisation is a code hook (`normalize=`). |
| Text only, English-first | Offsets are Unicode code points. CJK text is tokenised per character. Stopwords are configurable per language. |
| Cost control | Lean state, candidate caps, packing, the cache, a `max_cost_usd` budget guard, and per-document metrics (requests, questions, tokens, cost, retries, splits). |
| Adversarial content in documents | Document text is only ever state data, never part of an instruction. Outputs are always grounded, so injected text cannot create values that are not in the document. |

## Module map

| Module | Responsibility |
|---|---|
| `schema.py` | `Schema`, `Entity`, `Attribute`, `SentenceLabel`, `Field`; loading from dict/JSON/YAML; LangExtract-style `from_examples` |
| `data.py` | `CharInterval`, `Extraction`, `AnnotatedDocument`, `Metrics`; JSONL IO |
| `text.py` | Sentence segmentation, tokens, windows, candidate generators |
| `questions.py` | Question builders, answer parsing and validation, `ChoiceTask` tournament |
| `jev.py` | HTTP backend for `POST /v1/systemone`: retries, rate limiting, error taxonomy |
| `asker.py` | Cache lookup, packing under token limits, adaptive splitting, budget, metrics |
| `cache.py` | Memory and SQLite answer caches |
| `extractor.py` | The pipeline above; sync/async APIs; batch processing |
| `visualize.py` | Self-contained HTML highlighting |
| `testing.py` | `FakeBackend` for offline tests of your own schemas |
| `cli.py` | `typeextract extract / visualize / check` |
