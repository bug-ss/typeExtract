"""Render docs/assets/demo.gif: typeextract's HTML view for a few documents, with the highlights
appearing one class at a time.

    python examples/make_demo.py             # live, against Jev (needs TYPESAFE_API_KEY)
    python examples/make_demo.py --offline   # no key: a FakeBackend answers from lookup tables

Needs Pillow (``pip install pillow``) and a Chrome/Chromium binary: set ``CHROME=/path/to/chrome``
or have ``chromium``/``google-chrome`` on the PATH (a Playwright browser download also works).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typeextract as tx
from typeextract.testing import FakeBackend

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "assets" / "demo.gif"
SIZE = (960, 640)  # browser window, CSS pixels


@dataclass
class Example:
    title: str
    schema: tx.Schema
    text: str
    options: dict[str, Any] = field(default_factory=dict)
    offline: dict[str, Any] = field(default_factory=dict)  # FakeBackend lookup tables


EXAMPLES = [
    Example(
        "Service agreement · span_source=\"rules\"",
        tx.Schema.load(HERE / "schemas" / "contract.yaml"),
        (HERE / "texts" / "contract.txt").read_text(encoding="utf-8").split("\n", 2)[2],
        offline=dict(
            entities={
                "Acme Tecnologia Ltda.": "party",
                "Beta Consultoria S.A.": "party",
                "Maria Souza": "party",
                "$180,000.00": "amount",
                "$15,000.00": "amount",
                "2%": "rate",
                "1%": "rate",
                "30 days": "deadline",
                "Law No. 10.406/2002": "statute",
            },
            attributes={
                ("Acme Tecnologia Ltda.", "role"): "client",
                ("Beta Consultoria S.A.", "role"): "provider",
                ("Maria Souza", "role"): "witness",
            },
            sentence_labels={"obligation": ["shall"], "termination": ["terminate"]},
            fields={"contract_value": "$180,000.00", "signing_date": "March 3, 2026"},
        ),
    ),
    Example(
        "Clinical note · span_source=\"hybrid\"",
        tx.Schema(
            description="Clinical note",
            entities=[
                tx.Entity("condition", "a diagnosed disease or condition, not a symptom", counter_examples=["cough"]),
                tx.Entity(
                    "medication",
                    "a medicine's generic or brand name",
                    attributes=[tx.Attribute("status", ["active", "stopped", "planned"])],
                ),
                tx.Entity("dosage", "an amount of a medicine with its unit"),
                tx.Entity("date", "a calendar date"),
            ],
        ),
        "Admitted on 2026-09-12 with acute chronic obstructive pulmonary disease and type 2 diabetes. "
        "Takes metformin 500 mg twice daily and lisinopril 10 mg daily; aspirin was stopped last week. "
        "Plan: start prednisone 40 mg for five days.",
        options=dict(span_source="hybrid"),
        offline=dict(
            entities={
                "acute chronic obstructive pulmonary disease": "condition",
                "type 2 diabetes": "condition",
                "metformin": "medication",
                "lisinopril": "medication",
                "aspirin": "medication",
                "prednisone": "medication",
                "500 mg": "dosage",
                "10 mg": "dosage",
                "40 mg": "dosage",
                "2026-09-12": "date",
            },
            attributes={
                ("metformin", "status"): "active",
                ("lisinopril", "status"): "active",
                ("aspirin", "status"): "stopped",
                ("prednisone", "status"): "planned",
            },
        ),
    ),
    Example(
        "Business news · overlap=\"nested\"",
        tx.Schema(
            description="Business news",
            entities=[
                tx.Entity("person", "a named person"),
                tx.Entity("organization", "a named company or institution"),
                tx.Entity("location", "a named place"),
                tx.Entity("money", "an amount of money"),
            ],
        ),
        "Tim Cook said Apple will open offices in Paris, London and Berlin next year. "
        "Apple's chief executive also met Bank of America in New York about a $1.2 billion deal.",
        options=dict(overlap="nested"),
        offline=dict(
            entities={
                "Tim Cook": "person",
                "Apple": "organization",
                "Paris": "location",
                "London": "location",
                "Berlin": "location",
                "Bank of America": "organization",
                "America": "location",
                "New York": "location",
                "$1.2 billion": "money",
            }
        ),
    ),
    Example(
        "Chinese news · CJK text",
        tx.Schema(
            entities=[
                tx.Entity("person", "a person's name"),
                tx.Entity("organization", "a named organization"),
                tx.Entity("location", "a named place"),
            ]
        ),
        "昨天下午，张伟教授在清华大学主楼作了报告，随后前往北京市海淀区参观。",
        offline=dict(entities={"张伟": "person", "清华大学": "organization", "北京市海淀区": "location"}),
    ),
]


def extract(example: Example, offline: bool) -> tx.AnnotatedDocument:
    backend = FakeBackend(model="FakeBackend (offline demo)", **example.offline) if offline else None
    with tx.Extractor(example.schema, model="jev-1.13.0", backend=backend, **example.options) as ex:
        return ex.extract(example.text, document_id=example.title)


def reveal_order(doc: tx.AnnotatedDocument) -> list[list[str]]:
    """Classes to show at each step: none, then one entity class at a time, then everything
    (fields and sentence labels too)."""
    entities = list(dict.fromkeys(e.extraction_class for e in doc.entities))
    rest = [e.extraction_class for e in doc.sentence_labels] + [k for k, v in doc.fields.items() if v]
    return [entities[:n] for n in range(len(entities) + 1)] + [entities + list(dict.fromkeys(rest))]


def with_hidden(page: str, hidden: set[str]) -> str:
    """Hide some classes with CSS, so the layout never moves between frames."""
    if not hidden:
        return page
    sel = ",".join(f'mark[data-class="{c}"]' for c in sorted(hidden))
    rows = ",".join(f'tr[data-class="{c}"]' for c in sorted(hidden))
    css = f"<style>{sel}{{background:none!important;border-color:transparent!important;outline:none!important}}{rows}{{display:none}}</style>"
    return page.replace("</head>", css + "</head>", 1)


def find_chrome() -> str:
    candidates = [os.environ.get("CHROME")]
    candidates += [shutil.which(n) for n in ("chromium", "chromium-browser", "google-chrome", "chrome")]
    candidates += sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    candidates += sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    sys.exit("No Chrome/Chromium found: set CHROME=/path/to/chrome")


def screenshot(chrome: str, html: str, png: Path) -> None:
    page = png.with_suffix(".html")
    page.write_text(html, encoding="utf-8")
    subprocess.run(
        [
            chrome,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--disable-background-networking",
            "--no-first-run",
            "--force-device-scale-factor=2",
            f"--window-size={SIZE[0]},{SIZE[1] + 200}",  # headless viewports are shorter than the window
            f"--screenshot={png}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def main() -> None:
    from PIL import Image

    offline = "--offline" in sys.argv
    chrome = find_chrome()
    frames: list[Image.Image] = []
    durations: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, example in enumerate(EXAMPLES):
            doc = extract(example, offline)
            assert not doc.ungrounded()
            page = tx.to_html(doc, title=example.title)
            every = {e.extraction_class for e in doc.extractions} | set(doc.fields)
            order = reveal_order(doc)
            for j, shown in enumerate(order):
                png = Path(tmp) / f"{i:02d}_{j:02d}.png"
                screenshot(chrome, with_hidden(page, every - set(shown)), png)
                image = Image.open(png).convert("RGB").crop((0, 0, SIZE[0] * 2, SIZE[1] * 2)).resize(SIZE, Image.LANCZOS)
                frames.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
                durations.append(900 if j == 0 else 3200 if j == len(order) - 1 else 550)
            print(f"{example.title}: {len(doc.extractions)} extractions, {len(order)} steps")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
