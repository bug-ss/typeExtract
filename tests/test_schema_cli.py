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


def test_schema_errors_are_clear_not_attribute_or_type_errors():
    with pytest.raises(SchemaError):
        tx.SentenceLabel("x", None)
    with pytest.raises(SchemaError):
        tx.Schema.from_dict({"entities": [{"id": "drug"}]})  # description missing
    with pytest.raises(SchemaError):
        tx.Schema.from_dict({"entities": [{"id": "drug", "description": "d", "colour": "red"}]})
    assert tx.Entity("drug", "a medicine", examples="aspirin").examples == ("aspirin",)  # not per character
    assert isinstance(SchemaError("x"), tx.TypeExtractError)


def test_limits_that_cannot_finish_a_tournament_are_rejected():
    from typeextract.questions import ChoiceTask

    with pytest.raises(tx.ConfigurationError):
        tx.Extractor(tx.Schema.from_dict(SCHEMA), backend=FakeBackend(), limits=tx.Limits(max_choice_options=2))
    with pytest.raises(ValueError):
        ChoiceTask("k", "?", {"a": None, "b": None, "c": None}, escape=("none", "n"), max_options=2)


def test_cli_writes_one_html_per_document_and_reports_schema_errors(tmp_path, monkeypatch, capsys):
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA))
    rows = [{"text": f"Acme Ltd. pays R$ {i}.000,00."} for i in range(3)]
    (tmp_path / "docs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    fake = FakeBackend(entities={"Acme Ltd.": "party"})
    monkeypatch.setattr("typeextract.extractor.JevBackend", lambda *a, **k: fake)
    out_dir = tmp_path / "html"
    assert cli.main(["extract", str(tmp_path / "docs.jsonl"), "-s", str(tmp_path / "schema.json"), "-o", str(tmp_path / "o.jsonl"), "--html", str(out_dir), "--span-source", "hybrid"]) == 0
    assert len(list(out_dir.glob("*.html"))) == 3
    assert any("word" in q["instructions"] for _, qs in fake.calls for q in qs.values() if isinstance(q["instructions"], dict))
    # one document into an existing directory: written inside it, no IsADirectoryError
    (tmp_path / "one.txt").write_text("Acme Ltd. pays.")
    assert cli.main(["extract", str(tmp_path / "one.txt"), "-s", str(tmp_path / "schema.json"), "-o", str(tmp_path / "p.jsonl"), "--html", str(out_dir)]) == 0
    assert len(list(out_dir.glob("*.html"))) == 4
    (tmp_path / "bad.json").write_text(json.dumps({"entities": [{"id": "x"}]}))
    assert cli.main(["extract", str(tmp_path / "one.txt"), "-s", str(tmp_path / "bad.json")]) == 2
    err = capsys.readouterr().err
    assert "SchemaError" in err and "Traceback" not in err
