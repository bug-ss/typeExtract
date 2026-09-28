import json

import pytest

import typeextract as tx
from typeextract import cli
from typeextract.schema import SchemaError
from typeextract.testing import FakeBackend

SCHEMA = {
    "description": "Service agreement",
    "instructions": "Only extract parties that sign.",
    "min_confidence": 0.6,
    "entities": [
        {
            "id": "party",
            "description": "a company or person bound by the contract",
            "examples": ["Acme Ltd."],
            "attributes": {"role": ["client", "provider"], "signed": {"kind": "bool"}},
        },
        {"id": "amount", "description": "an amount of money", "patterns": [r"R\$\s?[\d.,]+"]},
    ],
    "sentence_labels": {"obligation": "creates a duty for a party"},
    "fields": [{"id": "total", "description": "the total price", "source": "amount"}],
}


def test_schema_from_dict_json_and_yaml(tmp_path):
    s = tx.Schema.from_dict(SCHEMA)
    party = s.entity("party")
    assert [a.name for a in party.attributes] == ["role", "signed"]
    assert party.attributes[0].options == {"client": None, "provider": None}
    assert party.definition() == {"definition": "a company or person bound by the contract", "examples": ["Acme Ltd."]}
    assert s.threshold(party) == 0.6 and s.sentence_labels[0].id == "obligation"
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    assert tx.Schema.load(tmp_path / "s.json").entity("amount").compiled_patterns
    yaml = pytest.importorskip("yaml")
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(SCHEMA))
    assert tx.Schema.load(tmp_path / "s.yaml").fields[0].source == ("amount",)


@pytest.mark.parametrize(
    "build",
    [
        lambda: tx.Schema(),
        lambda: tx.Schema(entities=[tx.Entity("a", "x"), tx.Entity("a", "y")]),
        lambda: tx.Entity("none", "reserved"),
        lambda: tx.Entity("x", ""),
        lambda: tx.Entity("x", "d", patterns=["("]),
        lambda: tx.Attribute("dosage"),
        lambda: tx.Attribute("level", ["only-one"], kind="score"),
        lambda: tx.Attribute("x", ["a", "unknown"]),
        lambda: tx.Schema(fields=[tx.Field("f", "d", source="nonexistent")]),
        lambda: tx.Schema(entities=[tx.Entity("x", "d")], min_confidence=1.5),
    ],
)
def test_invalid_schemas_fail_fast(build):
    with pytest.raises(SchemaError):
        build()


def test_extractor_rejects_bad_options():
    s = tx.Schema.from_dict(SCHEMA)
    for kw in ({"overlap": "maybe"}, {"on_error": "ignore"}, {"stopwords": "klingon"}, {"window_chars": 0}):
        with pytest.raises(tx.ConfigurationError):
            tx.Extractor(s, backend=FakeBackend(), **kw)


def test_cli_extract_and_visualize(tmp_path, monkeypatch, capsys):
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA))
    (tmp_path / "doc.txt").write_text("Acme Ltd. pays R$ 1.000,00. The client shall pay monthly.")
    fake = FakeBackend(
        entities={"Acme Ltd.": "party", "R$ 1.000,00": "amount"},
        sentence_labels={"obligation": ["shall pay"]},
        fields={"total": "R$ 1.000,00"},
    )
    monkeypatch.setattr("typeextract.extractor.JevBackend", lambda *a, **k: fake)
    out = tmp_path / "out.jsonl"
    code = cli.main(["extract", str(tmp_path / "doc.txt"), "-s", str(tmp_path / "schema.json"), "-o", str(out), "--html", str(tmp_path / "doc.html")])
    assert code == 0
    doc = json.loads(out.read_text())
    assert {e["extraction_text"] for e in doc["extractions"]} >= {"Acme Ltd.", "R$ 1.000,00", "The client shall pay monthly."}
    assert doc["fields"]["total"]["extraction_text"] == "R$ 1.000,00"
    assert "<mark" in (tmp_path / "doc.html").read_text()
    assert cli.main(["visualize", str(out), "-o", str(tmp_path / "again.html")]) == 0
    assert "extractions" in capsys.readouterr().err


def test_cli_reports_missing_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA))
    (tmp_path / "doc.txt").write_text("text")
    assert cli.main(["extract", str(tmp_path / "doc.txt"), "-s", str(tmp_path / "schema.json")]) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
