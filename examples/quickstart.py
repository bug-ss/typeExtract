"""Run the contract example.

    export TYPESAFE_API_KEY=...        # https://console.typesafe.ai
    python examples/quickstart.py      # live, against Jev
    python examples/quickstart.py --offline   # no key: a FakeBackend answers instead
"""

import sys
from pathlib import Path

import typeextract as tx
from typeextract.testing import FakeBackend

HERE = Path(__file__).parent
schema = tx.Schema.load(HERE / "schemas" / "contract.yaml")
text = (HERE / "texts" / "contract.txt").read_text(encoding="utf-8")

backend = None
if "--offline" in sys.argv:
    backend = FakeBackend(
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
    )

with tx.Extractor(schema, model="jev-1.13.0", backend=backend, cache=".typeextract-cache.sqlite") as ex:
    doc = ex.extract(text, document_id="contract.txt")

for e in doc.extractions:
    flag = "  <- review" if e.needs_review else ""
    print(f"{e.extraction_class:12} {e.extraction_text[:60]!r:64} [{e.start}:{e.end}] p={e.confidence:.2f} {e.attributes or ''}{flag}")
for name, value in doc.fields.items():
    print(f"field {name}: {value.extraction_text if value else None}")
m = doc.metrics
print(f"\n{m.requests} requests, {m.questions} questions, {m.input_tokens:,} tokens, ${m.cost_usd:.5f}, {m.latency_s:.2f}s")

tx.save_html(doc, "contract.html")
tx.save_jsonl([doc], "contract.jsonl")
print("wrote contract.html and contract.jsonl")
