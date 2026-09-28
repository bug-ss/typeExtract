"""Command line: ``typeextract extract | visualize | check``."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from .data import AnnotatedDocument, load_jsonl, save_jsonl
from .errors import TypeExtractError
from .extractor import SPAN_SOURCES, Extractor, run_sync
from .jev import DEFAULT_MODEL
from .schema import Schema
from .visualize import save_html


def _write_html(docs: list[AnnotatedDocument], target: str) -> None:
    """One document and a file target: that file. Otherwise (several documents, an existing
    directory, or a path ending in "/"): one uniquely named file per document in that directory."""
    out = Path(target)
    if len(docs) == 1 and not out.is_dir() and not target.endswith(("/", "\\")):
        save_html(docs[0], out)
        return
    out.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(docs):
        name = re.sub(r"[^\w.-]+", "_", d.document_id or "")[:80].strip("._") or "document"
        save_html(d, out / f"{i + 1:04d}_{name}.html")


def _extract(args: argparse.Namespace) -> int:
    schema = Schema.load(args.schema)
    inputs = []
    for p in args.inputs:
        path = Path(p)
        if path.suffix == ".jsonl":  # {"text": ..., "document_id": ...} per line
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if line.strip():
                    row = json.loads(line)
                    inputs.append({"text": row["text"], "document_id": row.get("document_id", f"{path.name}:{n}")})
        else:
            inputs.append({"text": path.read_text(encoding="utf-8"), "document_id": path.name})
    with Extractor(
        schema,
        model=args.model,
        cache=args.cache or None,
        max_cost_usd=args.max_cost,
        on_error="skip" if args.skip_errors else "raise",
        max_concurrent_documents=args.concurrency,
        span_source=args.span_source,
    ) as ex:
        docs = ex.extract_many(inputs)
    if args.out:
        save_jsonl(docs, args.out)
    else:
        for d in docs:
            print(json.dumps(d.to_dict(), ensure_ascii=False))
    if args.html:
        _write_html(docs, args.html)
    m = ex.metrics
    print(
        f"{len(docs)} document(s): {sum(len(d.extractions) for d in docs)} extractions, "
        f"{m.requests} requests, {m.questions} questions ({m.cached_questions} cached), "
        f"{m.input_tokens:,} tokens, ${m.cost_usd:.5f}, errors: {sum(len(d.errors) for d in docs)}",
        file=sys.stderr,
    )
    return 1 if any(d.errors for d in docs) else 0


def _visualize(args: argparse.Namespace) -> int:
    _write_html(list(load_jsonl(args.jsonl)), args.out)
    return 0


def _check(args: argparse.Namespace) -> int:
    from .jev import JevBackend
    from .questions import noul

    backend = JevBackend()

    async def ping() -> str:
        try:
            resp = await backend.evaluate("The sky is blue.", {"q": noul("Is the sky described as blue?")}, args.model)
            return f"ok: model={resp.model} answer={resp.answers['q']} usage={resp.input_tokens} tokens"
        finally:
            await backend.aclose()

    print(run_sync(ping))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="typeextract", description="Grounded extraction with Jev.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="extract from text files (or .jsonl with a 'text' field)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-s", "--schema", required=True, help="schema .json / .yaml")
    p.add_argument("-o", "--out", help="write annotated documents as JSONL")
    p.add_argument("--html", help="HTML file (one document) or directory (several)")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p.add_argument("--cache", help="SQLite answer cache path")
    p.add_argument("--max-cost", type=float, help="stop before spending more than this (USD)")
    p.add_argument("--concurrency", type=int, default=4, help="documents in flight")
    p.add_argument("--skip-errors", action="store_true", help="keep going when a window fails")
    p.add_argument(
        "--span-source",
        choices=SPAN_SOURCES,
        default="rules",
        help="who proposes spans: code rules, Jev word tagging, or both",
    )
    p.set_defaults(func=_extract)

    p = sub.add_parser("visualize", help="render a JSONL of annotated documents as HTML")
    p.add_argument("jsonl")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=_visualize)

    p = sub.add_parser("check", help="one tiny live call to verify the API key and model")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p.set_defaults(func=_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return int(args.func(args))
    except (TypeExtractError, OSError, json.JSONDecodeError, KeyError) as exc:  # schema/IO problems: no traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
