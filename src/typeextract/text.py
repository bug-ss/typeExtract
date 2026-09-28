"""Segmentation and candidate generation. Pure functions; offsets always index the original text."""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# Hiragana/Katakana, CJK ideographs (ext. A + unified + compatibility), half-width katakana.
# These scripts do not separate words with spaces, so each character is a token.
CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
_WORD = f"[^\\W_{CJK}]"
TOKEN_RE = re.compile(f"[{CJK}]|{_WORD}+(?:[.,'’\\-/&]{_WORD}+)*")
_CJK_RE = re.compile(f"[{CJK}]")

# --------------------------------------------------------------------------- stopwords

_EN = """a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each either else
etc even ever every few for from further had has have having he her here hers herself him himself
his how however i if in into is it its itself just let may me might more most must my myself no
nor not now of off on once only or other our ours ourselves out over own per same shall she
should so some such than that the their theirs them themselves then there therefore these they
this those through thus to too under until up upon us very via was we were what when where
whether which while who whom whose why will with within without would yet you your yours
yourself yourselves one two three said says say according including""".split()
_MULTI = """o a os as um uma uns umas de do da dos das em no na nos nas por pelo pela para com
sem e ou que se ao aos à às el la los las un una unos unas del al y en con sin para por le les
des du au aux et ou dans sur pour avec sans der die das ein eine einer eines dem den des und
oder mit von zu im am auf für""".split()
STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(_EN),
    "multi": frozenset(_EN) | frozenset(_MULTI),
}

# Abbreviations that never end a sentence (titles, "No.", "e.g.", month names before a day).
_NEVER_FINAL = frozenset(
    """mr mrs ms dr prof sr sra st mt gen col capt rev hon sen rep gov no nos vs fig figs eq approx
    dept cf al art arts sec secs para ref refs vol pp op e.g i.e u.s u.k jan feb mar apr jun jul
    aug sep sept oct nov dec""".split()
)
_TITLES = frozenset("mr mrs ms dr prof sr sra st gen col capt rev hon sen rep gov".split())
_CORP_SUFFIXES = frozenset(
    "inc ltd llc corp co plc gmbh ag sa s.a ltda lp llp nv bv oy ab sas srl".split()
)
_CONNECTORS = frozenset(
    "of de da do dos das del della di du la le van von der den y e and & for the".split()
)

# --------------------------------------------------------------------------- patterns

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_NUM = r"\d(?:[\d,.]*\d)?"
_UNIT = (
    r"mg|mcg|µg|μg|g|kg|lbs?|oz|ml|mL|L|cm|mm|m|km|mi|ft|hrs?|hours?|mins?|minutes?|secs?|seconds?"
    r"|days?|weeks?|months?|years?|yrs?|tablets?|tabs?|capsules?|caps?|units?|IU|mmHg|bpm|°[CF]|kWh|MW|GB|MB|TB"
)
BUILTIN_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])"),
    "url": re.compile(r"(?<!\w)(?:https?://|www\.)[^\s<>\"'()]*[^\s<>\"'().,;:!?]"),
    "money": re.compile(
        r"(?:(?:R\$|US\$|[$€£¥₹₩])\s?|(?<!\w)(?:USD|EUR|GBP|JPY|INR|BRL|CAD|AUD|CHF|CNY|MXN)\s?)"
        rf"{_NUM}(?:\s?(?:million|billion|thousand|mn|bn|[kKmMbB])(?!\w))?"
        rf"|(?<![\w.,]){_NUM}\s?(?:USD|EUR|GBP|BRL|dollars|euros|pounds|reais)(?!\w)"
    ),
    "percent": re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?\s?(?:%|percent(?!\w)|per cent(?!\w))"),
    "date": re.compile(
        r"(?<!\w)\d{4}-\d{1,2}-\d{1,2}(?!\w)"
        r"|(?<![\w/.-])\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})(?![\w/-])"
        rf"|(?<!\w){_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?(?!\w)"
        rf"|(?<!\w)\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\.?(?:,?\s+\d{{4}})?(?!\w)"
        rf"|(?<!\w){_MONTH}\.?\s+\d{{4}}(?!\w)"
        r"|(?<!\w)\d{1,2}\s+de\s+[a-zç]{4,9}\s+de\s+\d{4}(?!\w)"
    ),
    "time": re.compile(
        r"(?<![\w:])\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?(?![\w:])"
        r"|(?<![\w:])\d{1,2}\s?[AaPp]\.?[Mm]\.?(?!\w)"
    ),
    "phone": re.compile(r"(?<![\w+])\+?\(?\d[\d\s().-]{5,18}\d(?!\w)"),
    "quantity": re.compile(rf"(?<![\w.,])\d+(?:[.,]\d+)?\s?(?:{_UNIT})(?!\w)"),
    "identifier": re.compile(
        r"(?<![\w-])(?=[A-Za-z0-9./-]*\d)(?=[A-Za-z0-9./-]*[A-Za-z])[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)+(?![\w-])"
        r"|(?<!\w)[A-Z]{1,6}\d{2,}[A-Z0-9]*(?!\w)"
    ),
    "number": re.compile(r"(?<![\w.,$€£¥₹₩])\d+(?:[.,]\d+)*(?![\w%])"),
    "quoted": re.compile(r"[\"“«]([^\"”»\n]{2,80})[\"”»]"),
}
PATTERN_KINDS = frozenset(BUILTIN_PATTERNS)
_WEAK_KINDS = ("number", "phone")  # dropped when fully inside a more specific match


def _rank(kind: str) -> int:
    """number < phone < every other kind."""
    return _WEAK_KINDS.index(kind) if kind in _WEAK_KINDS else len(_WEAK_KINDS)


class _Cover:
    """Union of intervals with O(log n) "is (s, e) strictly inside it" queries."""

    def __init__(self, spans: Iterable[tuple[int, int]]):
        merged: list[list[int]] = []
        self.exact = set(spans)
        for s, e in sorted(self.exact):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.starts = [s for s, _ in merged]
        self.ends = [e for _, e in merged]

    def strictly_contains(self, s: int, e: int) -> bool:
        i = bisect.bisect_right(self.starts, s) - 1
        return i >= 0 and self.ends[i] >= e and (s, e) not in self.exact


def _pattern_spans(pattern: re.Pattern[str], text: str) -> Iterable[tuple[int, int]]:
    for m in pattern.finditer(text):
        s, e = (m.start(1), m.end(1)) if m.re.groups and m.group(1) is not None else m.span()
        s, e = _strip(text, s, e)
        if e > s:
            yield s, e


def _strip(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


# --------------------------------------------------------------------------- segmentation

_BULLET_RE = re.compile(r"\s*(?:[-*•·▪◦‣–—]|\d{1,3}[.)]|[A-Za-z][.)]|\(\w{1,3}\))\s")
_TABLE_RE = re.compile(r"\t| {3,}|\|")
_FINAL = ".!?…:;。！？"
_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)|[。！？]+[”’」』)）]*")
_SOFT_SPLIT_RE = re.compile(r"[;:,，、；：]\s*|\s+")


def _lines(text: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in re.finditer("\n", text):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(text)))
    return out


def _blocks(text: str) -> list[tuple[int, int]]:
    """Group lines into logical blocks, re-joining hard-wrapped paragraphs."""
    blocks: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    prev, prev_bullet = "", False
    for s, e in _lines(text):
        raw = text[s:e]
        line = raw.strip()
        if not line:
            if current:
                blocks.append(current)
            current, prev = None, ""
            continue
        bullet = bool(_BULLET_RE.match(raw))
        joins = (
            current is not None
            and len(prev) >= 40
            and prev[-1] not in _FINAL
            and not bullet
            # a list item continues when indented or when the line clearly continues it
            and not (prev_bullet and not raw[:1].isspace() and line[:1].isupper())
            and not _TABLE_RE.search(prev)
            and not _TABLE_RE.search(line)
        )
        if not joins:
            prev_bullet = bullet
        if joins and current is not None:
            current = (current[0], e)
        else:
            if current:
                blocks.append(current)
            current = (s, e)
        prev = line
    if current:
        blocks.append(current)
    return blocks


def _is_boundary(text: str, seg_start: int, block_end: int, punct_start: int, end: int) -> bool:
    if text[punct_start] in "。！？":
        return True
    if text[punct_start] == "." and re.fullmatch(r"\s*(?:\d{1,3}|[A-Za-z]|[IVXivx]{1,4})", text[seg_start:punct_start]):
        return False  # a list marker ("1.", "a.", "iv.") opening the segment
    nxt = end
    while nxt < block_end and text[nxt].isspace():
        nxt += 1
    if nxt >= block_end:
        return True
    if text[nxt].islower():
        return False
    if text[punct_start] == ".":
        m = re.search(r"([\w.]+)$", text[max(0, punct_start - 12) : punct_start])
        word = m.group(1).lower() if m else ""
        if word in _NEVER_FINAL or (len(word) == 1 and word.isalpha()):
            return False
    return True


def _force_split(text: str, s: int, e: int, max_chars: int) -> list[tuple[int, int]]:
    """Split an over-long sentence, preferring the last clause punctuation, then whitespace."""
    out = []
    while e - s > max_chars:
        floor, limit = s + max_chars // 3, s + max_chars
        punct = space = None
        for m in _SOFT_SPLIT_RE.finditer(text, floor, limit):
            if text[m.start()] in ";:,，、；：":
                punct = m.end()
            else:
                space = m.end()
        cut = punct or space or limit  # no break opportunity: hard cut (CJK, one giant token)
        a, b = _strip(text, s, cut)
        if b > a:
            out.append((a, b))
        s = cut
    a, b = _strip(text, s, e)
    if b > a:
        out.append((a, b))
    return out


def split_sentences(text: str, max_chars: int = 600) -> list[tuple[int, int]]:
    """Sentence spans ``(start, end)`` over ``text``; never longer than ``max_chars``."""
    sentences: list[tuple[int, int]] = []
    for bs, be in _blocks(text):
        start = bs
        for m in _BOUNDARY_RE.finditer(text, bs, be):
            if _is_boundary(text, start, be, m.start(), m.end()):
                a, b = _strip(text, start, m.end())
                if b > a:
                    sentences.extend(_force_split(text, a, b, max_chars))
                start = m.end()
        a, b = _strip(text, start, be)
        if b > a:
            sentences.extend(_force_split(text, a, b, max_chars))
    return sentences


@dataclass
class Window:
    """Consecutive sentences sent to Jev as one state."""

    index: int
    sentences: list[int]  # global sentence indices
    start: int
    end: int


def make_windows(
    sentences: Sequence[tuple[int, int]], max_chars: int = 800, max_sentences: int = 8
) -> list[Window]:
    windows: list[Window] = []
    for i, (s, e) in enumerate(sentences):
        w = windows[-1] if windows else None
        if w is not None and e - w.start <= max_chars and len(w.sentences) < max_sentences:
            w.sentences.append(i)
            w.end = e
        else:
            windows.append(Window(len(windows), [i], s, e))
    return windows


def context_before(text: str, start: int, chars: int) -> str:
    if chars <= 0 or start <= 0:
        return ""
    s = max(0, start - chars)
    if s > 0 and not text[s - 1].isspace():
        m = re.search(r"\s", text[s:start])
        s = s + m.end() if m else start
    return text[s:start].strip()


def context_after(text: str, end: int, chars: int) -> str:
    if chars <= 0 or end >= len(text):
        return ""
    e = min(len(text), end + chars)
    if e < len(text) and not text[e].isspace():
        m = re.search(r"\s\S*$", text[end:e])
        e = end + m.start() if m else end
    return text[end:e].strip()


# --------------------------------------------------------------------------- candidates


@dataclass
class Candidate:
    start: int
    end: int
    text: str
    sentence: int
    sources: set[str] = field(default_factory=set)
    kinds: set[str] = field(default_factory=set)
    """Built-in pattern kinds that matched exactly this span."""
    classes: set[str] = field(default_factory=set)
    """Entity ids whose own pattern, gazetteer or a custom generator proposed this span."""
    priority: int = 9


CustomGenerator = Callable[[str, int, int], Iterable[tuple[int, int] | tuple[int, int, str]]]
"""``fn(text, sentence_start, sentence_end)`` -> spans ``(start, end)`` or ``(start, end, class_id)``."""


class _Collector:
    def __init__(self, text: str, sentences: Sequence[tuple[int, int]]):
        self.text = text
        self.sentences = sentences
        self.starts = [s for s, _ in sentences]
        self.found: list[dict[tuple[int, int], Candidate]] = [{} for _ in sentences]

    def sentence_of(self, s: int, e: int) -> int | None:
        i = bisect.bisect_right(self.starts, s) - 1
        if i < 0 or e > self.sentences[i][1]:
            return None
        return i

    def add(
        self,
        s: int,
        e: int,
        source: str,
        priority: int,
        kind: str | None = None,
        cls: str | None = None,
        sentence: int | None = None,
    ) -> None:
        s, e = _strip(self.text, s, e)
        if e <= s:
            return
        i = self.sentence_of(s, e) if sentence is None else sentence
        if i is None:
            return
        cand = self.found[i].get((s, e))
        if cand is None:
            cand = Candidate(s, e, self.text[s:e], i, priority=priority)
            self.found[i][(s, e)] = cand
        cand.sources.add(source)
        cand.priority = min(cand.priority, priority)
        if kind:
            cand.kinds.add(kind)
        if cls:
            cand.classes.add(cls)


def _gazetteer_re(terms: Iterable[str]) -> re.Pattern[str] | None:
    words = sorted({t.strip() for t in terms if t and len(t.strip()) >= 2}, key=len, reverse=True)
    if not words:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\w)", re.IGNORECASE)


def generate_candidates(
    text: str,
    sentences: Sequence[tuple[int, int]],
    *,
    class_patterns: Mapping[str, Sequence[re.Pattern[str]]] | None = None,
    gazetteers: Mapping[str, Sequence[str]] | None = None,
    stopwords: Collection[str] = STOPWORDS["en"],
    max_ngram: int = 4,
    max_cjk_chars: int = 6,
    max_per_sentence: int = 60,
    builtin: bool = True,
    ngrams: bool = True,
    custom: Sequence[CustomGenerator] = (),
) -> list[list[Candidate]]:
    """Propose candidate spans per sentence, de-duplicated and capped by priority.

    Priority: class-specific patterns, gazetteers and custom generators (0), typed patterns (1),
    proper-noun runs (2), plain numbers (3), n-grams with a capitalised word (4), other n-grams (5+n).
    """
    col = _Collector(text, sentences)
    stop = {w.lower() for w in stopwords}

    # class-specific proposals
    for cls, patterns in (class_patterns or {}).items():
        for p in patterns:
            for s, e in _pattern_spans(p, text):
                col.add(s, e, f"pattern:{cls}", 0, cls=cls)
    for cls, terms in (gazetteers or {}).items():
        gz = _gazetteer_re(terms)
        if gz is not None:
            for m in gz.finditer(text):
                col.add(m.start(), m.end(), f"gazetteer:{cls}", 0, cls=cls)
    for i, (ss, se) in enumerate(sentences):
        for gen in custom:
            for item in gen(text, ss, se):
                s, e = int(item[0]), int(item[1])
                if ss <= s < e <= se:
                    col.add(s, e, "custom", 0, cls=item[2] if len(item) > 2 else None, sentence=i)

    # typed patterns
    if builtin:
        typed: list[tuple[int, int, str]] = []
        for kind, p in BUILTIN_PATTERNS.items():
            for s, e in _pattern_spans(p, text):
                if kind == "phone" and not 7 <= sum(c.isdigit() for c in text[s:e]) <= 15:
                    continue
                typed.append((s, e, kind))
        # A bare number (or a phone-looking digit run) inside a more specific match is noise.
        covers = {
            weak: _Cover([(s, e) for s, e, k in typed if _rank(k) > _WEAK_KINDS.index(weak)])
            for weak in _WEAK_KINDS
        }
        for s, e, kind in typed:
            if kind in _WEAK_KINDS and covers[kind].strictly_contains(s, e):
                continue
            col.add(s, e, f"pattern:{kind}", 3 if kind == "number" else 1, kind=kind)

    # token-based proposals
    tokens = [m.span() for m in TOKEN_RE.finditer(text)]
    tok_starts = [s for s, _ in tokens]
    caps = []
    for i, (ss, se) in enumerate(sentences):
        lo = bisect.bisect_left(tok_starts, ss)
        hi = bisect.bisect_left(tok_starts, se)
        toks = [t for t in tokens[lo:hi] if t[1] <= se]
        _proper_nouns(text, toks, stop, col, i)
        if ngrams:
            _ngrams(text, toks, stop, col, i, max_ngram, max_cjk_chars)
        # scripts without spaces produce many more character windows per sentence
        has_cjk = any(_CJK_RE.match(text[s:e]) for s, e in toks)
        caps.append(max_per_sentence * 3 if has_cjk else max_per_sentence)

    out = []
    for found, cap in zip(col.found, caps):
        ranked = sorted(found.values(), key=lambda c: (c.priority, c.start, c.end))
        out.append(sorted(ranked[:cap], key=lambda c: (c.start, c.end)))
    return out


def _is_cjk(tok: str) -> bool:
    return len(tok) == 1 and bool(_CJK_RE.match(tok))


def _is_cap(tok: str) -> bool:
    return tok[0].isupper() or (tok[0].isalpha() and any(c.isupper() for c in tok[1:]))


_GAP_RE = re.compile(r"\s*(?:[&/+\-–]\s*)?")


def _joinable_gap(gap: str) -> bool:
    return _GAP_RE.fullmatch(gap) is not None


def _is_stop(tok: str, stop: set[str]) -> bool:
    """Stopwords are case-insensitive, except all-caps tokens ("US", "IT") which may be names."""
    return tok.lower() in stop and not (len(tok) > 1 and tok.isupper())


def _proper_nouns(
    text: str, toks: list[tuple[int, int]], stop: set[str], col: _Collector, sentence: int
) -> None:
    """Runs of capitalised tokens: "Bank of America", "Dr. Smith", "Acme, Inc.", "Windows 11".

    Emits the whole run and each capitalised segment between connectors ("Smith of Acme Ltd."
    also yields "Smith" and "Acme Ltd."), plus a variant keeping a suffix's period ("Inc.").
    """
    words = [text[s:e] for s, e in toks]
    n = len(toks)

    def emit(a: int, b: int) -> None:
        if a == b and words[a].lower() in _TITLES:
            return
        s, e = toks[a][0], toks[b][1]
        col.add(s, e, "proper_noun", 2, sentence=sentence)
        if e < len(text) and text[e] == "." and words[b].lower() in _CORP_SUFFIXES | _NEVER_FINAL:
            col.add(s, e + 1, "proper_noun", 2, sentence=sentence)

    i = 0
    while i < n:
        w = words[i]
        if _is_cjk(w) or not _is_cap(w) or _is_stop(w, stop):
            i += 1
            continue
        segments = [[i, i]]
        k = i + 1
        while k < n:
            gap = text[toks[k - 1][1] : toks[k][0]]
            nxt = words[k]
            if _is_cjk(nxt):
                break
            if gap[:1] == "." and not gap[1:].strip() and words[k - 1].lower() in _TITLES and _is_cap(nxt):
                segments[-1][1] = k  # "Dr. Smith"
            elif gap.strip() == "," and nxt.lower().rstrip(".") in _CORP_SUFFIXES:
                segments[-1][1] = k  # "Acme, Inc"
            elif not _joinable_gap(gap):
                break
            elif _is_cap(nxt) or (nxt.isdigit() and len(nxt) <= 4):
                segments[-1][1] = k
            elif (
                nxt.lower() in _CONNECTORS
                and k + 1 < n
                and _is_cap(words[k + 1])
                and not _is_cjk(words[k + 1])
                and _joinable_gap(text[toks[k][1] : toks[k + 1][0]])
            ):
                segments.append([k + 1, k + 1])  # "Bank of America"
                k += 1
            else:
                break
            k += 1
        run_end = segments[-1][1]
        for a, b in {(segments[0][0], run_end), *(tuple(seg) for seg in segments)}:
            emit(a, b)
        i = run_end + 1


def _ngrams(
    text: str,
    toks: list[tuple[int, int]],
    stop: set[str],
    col: _Collector,
    sentence: int,
    max_ngram: int,
    max_cjk_chars: int,
) -> None:
    words = [text[s:e] for s, e in toks]
    n_toks = len(toks)
    for i in range(n_toks):
        first = words[i]
        cjk = _is_cjk(first)
        if not cjk and (_is_stop(first, stop) or not first[0].isalnum()):
            continue
        limit = max_cjk_chars if cjk else max_ngram
        for n in range(1, limit + 1):
            j = i + n - 1
            if j >= n_toks:
                break
            if n > 1:
                gap = text[toks[j - 1][1] : toks[j][0]]
                if _is_cjk(words[j]) != cjk or (cjk and gap) or (not cjk and not _joinable_gap(gap)):
                    break
            last = words[j]
            if cjk:
                if n < 2:
                    continue
            elif _is_stop(last, stop) or (n == 1 and (len(first) < 2 or first.lower() in _TITLES)):
                continue
            span = words[i : j + 1]
            if not cjk and all(t.replace(",", "").replace(".", "").isdigit() for t in span):
                continue
            has_cap = not cjk and any(_is_cap(t) for t in span)
            col.add(toks[i][0], toks[j][1], "ngram", 4 if has_cap else 5 + n, sentence=sentence)


def find_spans(text: str, patterns: Sequence[re.Pattern[str]]) -> list[tuple[int, int]]:
    """All matches of ``patterns`` in ``text`` (group 1 if present), de-duplicated, in order."""
    spans = {sp for p in patterns for sp in _pattern_spans(p, text)}
    return sorted(spans)


def snippet(text: str, start: int, end: int, width: int = 40) -> str:
    """The span with ``width`` characters of context on each side, whitespace collapsed."""
    a, b = max(0, start - width), min(len(text), end + width)
    out = " ".join(text[a:b].split())
    return ("…" if a > 0 else "") + out + ("…" if b < len(text) else "")
