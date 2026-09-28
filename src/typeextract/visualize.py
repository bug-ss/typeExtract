"""Self-contained HTML view of an annotated document (no external assets)."""

from __future__ import annotations

import heapq
import html
import json
from pathlib import Path

from .data import AnnotatedDocument, Extraction

_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#86bcb6",
]


def _colors(classes: list[str]) -> dict[str, str]:
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(classes)}


def _tooltip(e: Extraction) -> str:
    parts = [f"{e.extraction_class}  p={e.confidence:.2f}" if e.confidence is not None else e.extraction_class]
    parts += [f"{k}: {v}" for k, v in e.attributes.items()]
    parts += [f"{k}={v:.2f}" for k, v in e.scores.items()]
    if e.needs_review:
        parts.append("needs review")
    return "\n".join(parts)


def to_html(doc: AnnotatedDocument, title: str | None = None) -> str:
    text = doc.text
    spans = [e for e in doc.extractions if e.kind != "sentence"]
    spans += [f for f in doc.fields.values() if f is not None]
    classes = sorted({e.extraction_class for e in [*spans, *doc.sentence_labels]})
    color = _colors(classes)

    # innermost (shortest) span wins at every character: sweep the boundaries with a heap of
    # the spans open at each point, O(N log N) instead of rescanning every span per segment
    bounds = sorted({0, len(text), *(e.start for e in spans), *(e.end for e in spans)})
    by_start = sorted(range(len(spans)), key=lambda k: spans[k].start)
    open_spans: list[tuple[int, int]] = []  # (length, index)
    nxt = 0
    body = []
    for a, b in zip(bounds, bounds[1:]):
        while nxt < len(by_start) and spans[by_start[nxt]].start <= a:
            k = by_start[nxt]
            heapq.heappush(open_spans, (spans[k].end - spans[k].start, k))
            nxt += 1
        while open_spans and spans[open_spans[0][1]].end <= a:  # lazily drop spans that ended
            heapq.heappop(open_spans)
        chunk = html.escape(text[a:b])
        if open_spans:
            e = spans[open_spans[0][1]]
            style = f"background:{color[e.extraction_class]}33;border-bottom:2px solid {color[e.extraction_class]}"
            if e.needs_review:
                style += ";outline:1px dashed #999"
            chunk = (
                f'<mark data-class="{html.escape(e.extraction_class)}" style="{style}" '
                f'title="{html.escape(_tooltip(e))}">{chunk}</mark>'
            )
        body.append(chunk)

    legend = "".join(
        f'<span class="tag" style="border-color:{color[c]};background:{color[c]}22">{html.escape(c)}</span>'
        for c in classes
    )
    rows = "".join(
        '<tr data-class="{}"><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            html.escape(e.extraction_class),
            html.escape(e.extraction_class),
            html.escape(e.extraction_text if len(e.extraction_text) < 120 else e.extraction_text[:117] + "…"),
            f"{e.start}–{e.end}",
            f"{e.confidence:.2f}" if e.confidence is not None else "",
            html.escape(json.dumps(e.attributes, ensure_ascii=False)) if e.attributes else ("review" if e.needs_review else ""),
        )
        for e in [*doc.extractions, *(f for f in doc.fields.values() if f is not None)]
    )
    m = doc.metrics
    stats = (
        f"{len(doc.extractions)} extractions · {m.requests} requests · {m.questions} questions "
        f"({m.cached_questions} cached) · {m.input_tokens:,} tokens · ${m.cost_usd:.5f} · {m.latency_s:.2f}s"
    )
    errors = "".join(f"<li>{html.escape(str(err))}</li>" for err in doc.errors)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title or doc.document_id or "typeextract")}</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#e3e3e3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#e8e8ea;--muted:#9a9aa0;--line:#2c2d31}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;margin:0;padding:24px 16px;}}
main{{max-width:960px;margin:auto}} .text{{white-space:pre-wrap;border:1px solid var(--line);border-radius:8px;padding:16px}}
mark{{color:inherit;border-radius:3px;padding:0 1px}} .tag{{display:inline-block;border:1px solid;border-radius:12px;padding:0 10px;margin:0 6px 6px 0;font-size:13px}}
.muted{{color:var(--muted);font-size:13px}} table{{border-collapse:collapse;width:100%;margin-top:16px;font-size:14px;table-layout:fixed}}
td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}
.scroll{{overflow-x:auto}}
</style></head><body><main>
<h1 style="font-size:20px">{html.escape(title or doc.document_id or "Extraction")}</h1>
<p class="muted">{html.escape(stats)} · model {html.escape(doc.model or "?")}</p>
<div>{legend}</div>
<div class="text">{"".join(body)}</div>
{f'<h2 style="font-size:16px">Errors</h2><ul>{errors}</ul>' if errors else ""}
<div class="scroll"><table><colgroup><col style="width:17%"><col style="width:43%"><col style="width:11%"><col style="width:7%"><col style="width:22%"></colgroup><tr><th>class</th><th>text</th><th>offsets</th><th>p</th><th>attributes</th></tr>{rows}</table></div>
</main></body></html>"""


def save_html(doc: AnnotatedDocument, path: str | Path, title: str | None = None) -> None:
    Path(path).write_text(to_html(doc, title), encoding="utf-8")
