"""Draw docs/assets/jev-mode.svg and jev-mode-dark.svg: how ``span_source="jev"`` turns words into
extractions, on one worked sentence.

The words, regions, candidate spans and offsets are computed with typeextract's own code, so the
drawing cannot drift from the implementation. The probabilities stand in for Jev's answers and
are illustrative.

    python examples/make_diagram.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from typeextract.text import STOPWORDS, _Collector, region_candidates, tokens, typed_spans

OUT = Path(__file__).parent.parent / "docs" / "assets"
SENTENCE = "Patient has acute chronic obstructive pulmonary disease and type 2 diabetes; takes metformin 500 mg."
P_WORD = {  # illustrative "is this word part of a mention?" answers
    "Patient": 0.03, "has": 0.01, "acute": 0.91, "chronic": 0.96, "obstructive": 0.97, "pulmonary": 0.97,
    "disease": 0.95, "and": 0.12, "type": 0.88, "2": 0.85, "diabetes": 0.95, "takes": 0.02,
    "metformin": 0.97, "500": 0.83, "mg": 0.79,
}
THRESHOLD = 0.3
ACCEPTED = [  # illustrative round-1 answers
    ("acute chronic obstructive pulmonary disease", "condition", 0.93),
    ("type 2 diabetes", "condition", 0.91),
    ("metformin", "medication", 0.95),
    ("500 mg", "dosage", 0.92),
]
REJECTED = [
    ("acute chronic obstructive pulmonary disease and type 2 diabetes", "two mentions"),
    ("metformin 500 mg", "extra words"),
    ("pulmonary disease", "only part of a mention"),
    ("2", "not a mention"),
]

W = 960
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
THEMES = {
    "jev-mode.svg": dict(
        fg="#1f2328", muted="#59636e", line="#d0d7de", panel="#f6f8fa", jev="#8250df", code="#57606a",
        condition="#3b6ea8", medication="#cf222e", dosage="#1a7f37",
    ),
    "jev-mode-dark.svg": dict(
        fg="#e6edf3", muted="#9198a1", line="#3d444d", panel="#151b23", jev="#bc8cff", code="#9198a1",
        condition="#79b8ff", medication="#ff7b72", dosage="#56d364",
    ),
}


def tw(s: str, size: float, mono: bool = False) -> float:
    """Generous text width estimate (sized for wide fonts such as DejaVu Sans)."""
    if mono:
        return len(s) * size * 0.61
    em = 0.0
    for ch in s:
        if ch in "ijlI.,;:'!|’“” ":
            em += 0.32
        elif ch in "frt()-1":
            em += 0.42
        elif ch in "mwMW…":
            em += 0.95
        elif ch.isupper() or ch.isdigit():
            em += 0.68
        else:
            em += 0.62
    return em * size


class Svg:
    def __init__(self, pal: dict[str, str]):
        self.p = pal
        self.parts: list[str] = []

    def text(self, x, y, s, size=13, color="fg", weight=400, anchor="start", mono=False, italic=False):
        style = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else SANS}" font-size="{size}" '
            f'font-weight="{weight}" fill="{self.p.get(color, color)}" text-anchor="{anchor}"{style}>{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, stroke="line", fill="none", fill_opacity=1.0, rx=6, dash=False, sw=1.0):
        d = ' stroke-dasharray="4 3"' if dash else ""
        f = self.p.get(fill, fill)
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{f}" '
            f'fill-opacity="{fill_opacity}" stroke="{self.p.get(stroke, stroke)}" stroke-width="{sw}"{d}/>'
        )

    def line(self, x1, y1, x2, y2, color="muted", dash=False, arrow=False, sw=1.2):
        d = ' stroke-dasharray="4 3"' if dash else ""
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{self.p[color]}" stroke-width="{sw}"{d}{m}/>'
        )

    def pill(self, x, y, label, color) -> float:
        """A small "Jev" / "code" tag; (x, y) is the top-left corner. Returns its width."""
        w = tw(label, 11) + 14
        self.rect(x, y, w, 18, stroke=color, fill=color, fill_opacity=0.14, rx=9)
        self.text(x + w / 2, y + 13, label, size=11, color=color, weight=700, anchor="middle")
        return w

    def chip(self, x, y, s, color="line", fill_opacity=0.0, text_color="fg", dash=False, size=12.5) -> float:
        """A span of text as a chip; (x, y) is the top-left corner. Returns its width."""
        w = tw(s, size) + 16
        self.rect(x, y, w, 26, stroke=color, fill=color, fill_opacity=fill_opacity, rx=5, dash=dash)
        self.text(x + 8, y + 17.5, s, size=size, color=text_color)
        return w

    def heading(self, y, kind, label, note=None) -> None:
        """A step heading at baseline ``y``: a Jev/code pill, the step, and a right-aligned note."""
        w = self.pill(36, y - 13, "Jev" if kind == "jev" else "code", kind)
        self.text(36 + w + 10, y, label, size=13.5, weight=600)
        if note:
            self.text(924, y, note, size=12, color="muted", anchor="end")

    def arrow(self, y1, y2, label) -> None:
        self.line(64, y1, 64, y2 - 2, color="muted", arrow=True, sw=1.4)
        self.text(78, (y1 + y2) / 2 + 4, label, size=12, color="muted", italic=True)

    def render(self, height: int) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
            f'role="img" aria-label="How span_source=jev works: Jev tags every word, code turns tagged words into '
            f'regions and candidate spans, Jev classifies each span and verifies the accepted ones, code resolves '
            f'the result into grounded extractions.">'
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{self.p["muted"]}"/></marker></defs>'
            + "".join(self.parts)
            + "</svg>"
        )


def compute():
    toks = tokens(SENTENCE)
    words = [SENTENCE[s:e] for s, e in toks]
    assert words == list(P_WORD), words
    inside = [P_WORD[w] >= THRESHOLD for w in words]
    col = _Collector(SENTENCE, [(0, len(SENTENCE))])
    region_candidates(SENTENCE, 0, (0, len(SENTENCE)), toks, inside, col, STOPWORDS["en"])
    tagged = {(c.start, c.end) for c in col.found[0].values()}
    typed = {(s, e): k for s, e, k in typed_spans(SENTENCE)}
    # regions = tagger spans not contained in another tagger span
    regions = sorted(sp for sp in tagged if not any(o != sp and o[0] <= sp[0] and sp[1] <= o[1] for o in tagged))
    for text, *_ in ACCEPTED + REJECTED:  # every span drawn is one the code really proposes
        i = SENTENCE.index(text)
        assert (i, i + len(text)) in tagged or (i, i + len(text)) in typed, text
    return toks, words, inside, regions, tagged, typed


def draw(pal: dict[str, str]) -> str:
    toks, words, inside, regions, tagged, typed = compute()
    g = Svg(pal)
    span = lambda sp: SENTENCE[sp[0] : sp[1]]  # noqa: E731

    # ---- header
    g.text(24, 34, 'How span_source="jev" finds entities', size=18, weight=700)
    x = 936.0
    for kind, label, note in (("code", "code", "runs locally"), ("jev", "Jev", "a typed question to Jev")):
        x -= tw(note, 12)
        g.text(x, 34, note, size=12, color="muted")
        x -= tw(label, 11) + 14 + 6
        g.pill(x, 21, label, kind)
        x -= 18

    # ---- ① + ②: tag every word, join tagged words into regions
    g.rect(16, 52, 928, 184, fill="panel", stroke="line", rx=10)
    g.heading(80, "jev", "①  one Noul per word: “is this word part of a mention of an entity type?”", "round trip 1")
    widths = [tw(w, 12.5) + 12 for w in words]
    gap = 6
    row = sum(widths) + gap * (len(words) - 1)
    assert row <= 888, f"word row is {row:.0f}px wide; shorten SENTENCE"
    x = (W - row) / 2
    centers, edges = [], []
    for w_, wd, ins in zip(words, widths, inside):
        connector = not ins and w_ == "and"
        g.rect(x, 96, wd, 26, stroke="jev" if (ins or connector) else "line", fill="jev",
               fill_opacity=0.16 if ins else 0, rx=5, dash=connector)
        g.text(x + wd / 2, 113.5, w_, size=12.5, color="fg" if ins else "muted", anchor="middle")
        p = P_WORD[w_]
        g.text(x + wd / 2, 140, f"{p:.2f}", size=11.5, color="jev" if p >= THRESHOLD else "muted",
               weight=700 if p >= THRESHOLD else 400, anchor="middle")
        centers.append(x + wd / 2)
        edges.append((x, x + wd))
        x += wd + gap
    tok_at = {s: i for i, (s, _) in enumerate(toks)}
    tok_end = {e: i for i, (_, e) in enumerate(toks)}
    for n, (s, e) in enumerate(regions, 1):
        x0, x1 = edges[tok_at[s]][0], edges[tok_end[e]][1]
        g.parts.append(
            f'<path d="M{x0:.1f},150 V158 H{x1:.1f} V150" fill="none" stroke="{pal["fg"]}" stroke-width="1.6"/>'
        )
        g.text((x0 + x1) / 2, 176, f"region {n}", size=12, weight=600, anchor="middle")
    g.heading(214, "code", "②  words with P ≥ 0.3 join into regions; “and” between tagged words is absorbed as a connector")

    g.arrow(236, 262, "2 regions")

    # ---- ③: candidate spans
    g.rect(16, 262, 928, 296, fill="panel", stroke="line", rx=10)
    g.heading(290, "code", "③  each region proposes itself, its segments and its n-grams; precise generators still run too")
    r1, r2 = regions
    r1_spans = sorted((sp for sp in tagged if r1[0] <= sp[0] and sp[1] <= r1[1]), key=lambda sp: (sp[0], -sp[1]))
    r1_segments = [sp for sp in r1_spans if span(sp) in (ACCEPTED[0][0], ACCEPTED[1][0])]
    r1_ngrams = [sp for sp in r1_spans if sp != r1 and sp not in r1_segments]
    g.text(36, 318, "region 1", size=12, color="muted", weight=600)
    y = 326
    for sp, note in [(r1, "whole region"), *[(s, "segment (split at the connector)") for s in r1_segments]]:
        w = g.chip(36, y, span(sp), color="jev", fill_opacity=0.10)
        g.text(36 + w + 10, y + 17.5, note, size=12, color="muted", italic=True)
        y += 32
    x = 36.0
    shown = ["pulmonary disease", "chronic obstructive", "2 diabetes"]
    for s in shown:
        x += g.chip(x, y, s, color="jev", fill_opacity=0.10) + 8
    g.text(x + 2, y + 17.5, f"+ {len(r1_ngrams) - len(shown)} more n-grams (≤ 4 words)", size=12, color="muted", italic=True)

    r2_spans = [sp for sp in tagged if r2[0] <= sp[0] and sp[1] <= r2[1] and sp != r2]
    g.text(36, 468, "region 2", size=12, color="muted", weight=600)
    w = g.chip(36, 476, span(r2), color="jev", fill_opacity=0.10)
    g.text(36 + w + 10, 476 + 17.5, "whole region (its only segment)", size=12, color="muted", italic=True)
    x = 36.0
    order = sorted(r2_spans, key=lambda sp: (span(sp) == "500 mg", sp))  # "500 mg" last, next to its twin
    for sp in order:
        x += g.chip(x, 508, span(sp), color="jev", fill_opacity=0.10) + 8
    tagger_500_right = x - 8

    # precise generators, on the right, level with region 2
    g.text(640, 468, "precise generators (still on)", size=12, color="muted", weight=600)
    two = next(sp for sp, k in typed.items() if k == "number")
    five = next(sp for sp, k in typed.items() if k == "quantity")
    w = g.chip(640, 476, span(two), color="code", fill_opacity=0.10)
    g.text(640 + w + 10, 476 + 17.5, "typed: number", size=12, color="muted", italic=True)
    w = g.chip(640, 508, span(five), color="code", fill_opacity=0.10)
    g.text(640 + w + 10, 508 + 17.5, "typed: quantity", size=12, color="muted", italic=True)
    g.line(tagger_500_right + 6, 521, 634, 521, color="muted", dash=True)
    g.text((tagger_500_right + 640) / 2, 514, "same offsets: one candidate, both sources", size=11.5, color="muted", anchor="middle")
    g.text(640, 552 - 4, "+ your regexes, terms, examples, custom generators", size=11.5, color="muted", italic=True)

    n_candidates = len(tagged | set(typed))
    g.arrow(558, 584, f"{n_candidates} candidate spans, deduplicated by offsets")

    # ---- ④: classify every candidate
    g.rect(16, 584, 928, 216, fill="panel", stroke="line", rx=10)
    g.heading(612, "jev", "④  one Choice per candidate: “which entity type is this exact span, or none?”",
              "round trip 2 (precise generators' spans: round trip 1)")
    g.text(36, 640, "accepted", size=12, weight=700, color="fg")
    g.text(500, 640, "none", size=12, weight=700, color="muted")
    for i, (s, cls, p) in enumerate(ACCEPTED):
        y = 648 + i * 30
        w = g.chip(36, y, s, color=cls, fill_opacity=0.12)
        g.text(36 + w + 10, y + 17.5, f"{cls} {p:.2f}", size=12.5, color=cls, weight=600)
    for i, (s, why) in enumerate(REJECTED):
        y = 648 + i * 30
        shown_text = s if len(s) < 30 else "acute chronic … type 2 diabetes"
        w = g.chip(500, y, shown_text, color="muted", dash=True, text_color="muted")
        g.text(500 + w + 10, y + 17.5, why, size=12.5, color="muted", italic=True)
    g.text(500, 786, f"+ {n_candidates - len(ACCEPTED) - len(REJECTED)} more → none", size=12, color="muted", italic=True)

    g.arrow(800, 826, "accepted: p ≥ min_confidence")

    # ---- ⑤ + ⑥: verify, attributes, resolve
    g.rect(16, 826, 928, 242, fill="panel", stroke="line", rx=10)
    g.heading(854, "jev", "⑤  accepted spans only: a verification Noul, plus each attribute (Choice / Noul / Score)",
              "round trip 3")
    w = g.chip(36, 868, "metformin", color="medication", fill_opacity=0.12)
    g.text(36 + w + 12, 885.5, "complete mention of medication?  0.97", size=12.5, color="fg")
    g.text(36 + w + 12 + tw("complete mention of medication?  0.97", 12.5) + 24, 885.5, "status: active  0.94",
           size=12.5, color="fg")
    g.heading(928, "code", "⑥  thresholds · verification · overlaps → grounded extractions")
    rows = [(s, cls, p) for s, cls, p in ACCEPTED]
    for i, (s, cls, p) in enumerate(rows):
        y = 956 + i * 24
        a = SENTENCE.index(s)
        attr = "status=active" if s == "metformin" else ""
        g.text(36, y, cls, size=12.5, color=cls, mono=True, weight=600)
        g.text(160, y, f"[{a}, {a + len(s)})", size=12.5, mono=True, color="muted")
        g.text(262, y, s, size=12.5, mono=True)
        g.text(640, y, attr, size=12.5, mono=True, color="muted")
        g.text(924, y, f"{p:.2f}", size=12.5, mono=True, anchor="end")
    g.text(36, 1056, "Offsets index the original text: every extraction is an exact slice of it.", size=12,
           color="muted", italic=True)
    return g.render(1084)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pal in THEMES.items():
        (OUT / name).write_text(draw(pal), encoding="utf-8")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
