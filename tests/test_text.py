import re

import pytest

from typeextract.text import (
    _Collector,
    context_after,
    find_occurrences,
    region_candidates,
    tokens,
    typed_spans,
    context_before,
    find_spans,
    generate_candidates,
    make_windows,
    split_sentences,
)


def texts(text, spans):
    return [text[s:e] for s, e in spans]


def cands(text, **kw):
    sents = split_sentences(text)
    return {c.text: c for cs in generate_candidates(text, sents, **kw) for c in cs}


# ------------------------------------------------------------------ sentences


def test_abbreviations_initials_and_decimals_do_not_split():
    text = "Dr. Smith paid $3.50 on Jan. 5 to J. R. Tolkien, e.g. for books. Then he left."
    assert texts(text, split_sentences(text)) == [
        "Dr. Smith paid $3.50 on Jan. 5 to J. R. Tolkien, e.g. for books.",
        "Then he left.",
    ]


def test_sentence_ending_abbreviation_splits_before_capital():
    text = "The meeting ends at 3 p.m. The report is due tomorrow."
    assert len(split_sentences(text)) == 2


def test_hard_wrapped_lines_are_rejoined_but_headings_bullets_and_blank_lines_break():
    text = (
        "PAYMENT TERMS\n"
        "The client shall pay the provider within thirty days of the invoice and\n"
        "any late payment accrues interest.\n"
        "- first item without a full stop\n"
        "- second item\n"
        "\n"
        "New paragraph."
    )
    assert texts(text, split_sentences(text)) == [
        "PAYMENT TERMS",
        "The client shall pay the provider within thirty days of the invoice and\nany late payment accrues interest.",
        "- first item without a full stop",
        "- second item",
        "New paragraph.",
    ]


def test_cjk_punctuation_splits_without_spaces():
    text = "张伟在清华大学作了报告。随后前往北京！"
    assert texts(text, split_sentences(text)) == ["张伟在清华大学作了报告。", "随后前往北京！"]


def test_long_sentences_are_force_split_within_limit_and_cover_text():
    text = ("alpha beta gamma, delta epsilon; " * 60).strip()
    spans = split_sentences(text, max_chars=100)
    assert all(e - s <= 100 for s, e in spans)
    assert all(s2 >= e1 for (_, e1), (s2, _) in zip(spans, spans[1:]))
    assert re.sub(r"\s+", "", "".join(texts(text, spans))) == re.sub(r"\s+", "", text)


def test_unbreakable_text_is_hard_cut():
    text = "x" * 1000
    spans = split_sentences(text, max_chars=300)
    assert sum(e - s for s, e in spans) == 1000 and all(e - s <= 300 for s, e in spans)


@pytest.mark.parametrize("text", ["", "   \n\n  ", "\r\n"])
def test_empty_input(text):
    assert split_sentences(text) == []


def test_offsets_index_original_text_with_crlf():
    text = "First line.\r\nSecond line here.\r\n"
    for s, e in split_sentences(text):
        assert text[s:e] == text[s:e].strip() and text[s:e]


def test_windows_and_context():
    text = "One two. " * 200
    sents = split_sentences(text)
    windows = make_windows(sents, max_chars=100)
    assert all(w.end - w.start <= 100 for w in windows)
    assert [i for w in windows for i in w.sentences] == list(range(len(sents)))
    w = windows[3]
    before, after = context_before(text, w.start, 30), context_after(text, w.end, 30)
    assert text[: w.start].rstrip().endswith(before) and before[:1].isalpha()
    assert text[w.end :].lstrip().startswith(after)


# ------------------------------------------------------------------ candidates


def test_typed_patterns():
    text = (
        "Pay $1,315.50 or R$ 180.000,00 (5% fee) by 2026-10-10 or 10/10/2026 or March 3, 2026 at 3:30 p.m. "
        "Mail dana@acme-corp.com, visit https://acme.com/x?a=1, call +1 (415) 555-0177, "
        "take 500 mg, ref INV-2087."
    )
    c = cands(text)
    for span, kind in [
        ("$1,315.50", "money"),
        ("R$ 180.000,00", "money"),
        ("5%", "percent"),
        ("2026-10-10", "date"),
        ("10/10/2026", "date"),
        ("March 3, 2026", "date"),
        ("3:30 p.m.", "time"),
        ("dana@acme-corp.com", "email"),
        ("https://acme.com/x?a=1", "url"),
        ("+1 (415) 555-0177", "phone"),
        ("500 mg", "quantity"),
        ("INV-2087", "identifier"),
    ]:
        assert f"typed:{kind}" in c[span].sources, span
    # numbers inside more specific matches are not proposed on their own
    assert "415" not in c and "0177" not in c and "1,315.50" not in c


def test_proper_nouns_connectors_titles_and_suffixes():
    c = cands("Dr. Smith met the Bank of America team and Acme Tecnologia Ltda. in New York with Acme, Inc. staff.")
    for span in ["Dr. Smith", "Bank of America", "America", "Acme Tecnologia Ltda.", "New York", "Acme, Inc."]:
        assert span in c, span
    assert "Dr" not in c  # a bare title is never a candidate


def test_ngram_edges_skip_stopwords_but_not_all_caps_names():
    c = cands("the patient with type 2 diabetes flew to the US")
    assert "type 2 diabetes" in c and "diabetes" in c and "US" in c
    assert not any(t.lower().startswith("the ") or t.lower().endswith(" the") for t in c)
    assert "with type" not in c


def test_candidates_are_grounded_and_capped():
    text = " ".join(f"Word{i}" for i in range(200)) + "."
    sents = split_sentences(text, max_chars=10_000)
    out = generate_candidates(text, sents, max_per_sentence=25)
    assert all(len(cs) <= 25 for cs in out)
    assert all(text[c.start : c.end] == c.text for cs in out for c in cs)


def test_gazetteer_class_patterns_and_custom_generators():
    text = "Metformin 500 mg and ASPIRIN daily; code ZX-9."
    sents = split_sentences(text)

    def custom(t, s, e):
        i = t.index("daily", s, e)
        yield (i, i + 5, "frequency")

    out = generate_candidates(
        text,
        sents,
        gazetteers={"drug": ["metformin", "aspirin"]},
        class_patterns={"code": [re.compile(r"code (\w+-\d)")]},
        custom=[custom],
    )
    c = {x.text: x for cs in out for x in cs}
    assert "gazetteer:drug" in c["Metformin"].sources and "gazetteer:drug" in c["ASPIRIN"].sources
    assert "pattern:code" in c["ZX-9"].sources
    assert "custom:frequency" in c["daily"].sources


def test_cjk_character_windows():
    c = cands("张伟在清华大学作了报告。")
    assert "清华大学" in c and "张伟" in c


def test_find_spans_uses_group_one():
    assert find_spans("id: A1, id: B2", [re.compile(r"id: (\w\d)")]) == [(4, 6), (12, 14)]


def test_numbered_list_markers_stay_with_their_sentence():
    text = "1. The Provider shall deliver reports.\n2. The Client shall pay.\na. Minor point."
    assert texts(text, split_sentences(text)) == [
        "1. The Provider shall deliver reports.",
        "2. The Client shall pay.",
        "a. Minor point.",
    ]


def test_wrapped_list_items_continue_but_new_paragraphs_do_not():
    text = (
        "2. The Client shall pay the Provider a total of $180,000.00 in twelve monthly installments of\n"
        "$15,000.00, each due by the 10th day of the month.\n"
        "- Item two: call +1 (415) 555-0177 or email dana.whit@acme-corp.com\n"
        "Invoice INV-2087 follows."
    )
    assert texts(text, split_sentences(text)) == [
        "2. The Client shall pay the Provider a total of $180,000.00 in twelve monthly installments of\n"
        "$15,000.00, each due by the 10th day of the month.",
        "- Item two: call +1 (415) 555-0177 or email dana.whit@acme-corp.com",
        "Invoice INV-2087 follows.",
    ]



# ------------------------------------------------------------------ review regressions


def test_combining_marks_and_decomposed_accents_stay_in_one_token():
    for text, expected in [("दिल्ली में", ["दिल्ली", "में"]), ("Mu\u0308ller GmbH", ["Mu\u0308ller", "GmbH"])]:
        assert [text[s:e] for s, e in tokens(text)] == expected


def test_possessives_are_not_part_of_names():
    c = cands("Apple's CEO met Google’s team; O'Neil agreed.")
    assert "Apple" in c and "Google" in c and "O'Neil" in c
    assert not any("'s" in t or "’s" in t for t in c)


def test_occurrences_are_whole_words():
    text = "A man and a woman had a cat in Austin."
    assert len(find_occurrences(text, "a")) == 2  # not inside "man", "woman", "had", "cat"
    assert find_occurrences(text, "man") == [2]
    assert find_occurrences(text, "in") == [text.index(" in ") + 1]
    assert find_occurrences("北京上海北京", "北京") == [0, 4]  # CJK: plain substring


def test_typed_spans_apply_kind_rules():
    text = "Firmware 1.2.3.4 shipped March 3, 2026 with 40 units; hotline +1 415 555 0177."
    spans = {(text[s:e], k) for s, e, k in typed_spans(text)}
    assert ("+1 415 555 0177", "phone") in spans
    assert not any(k == "phone" and t == "1.2.3.4" for t, k in spans)
    assert not any(k == "number" and t in {"3", "2026", "40", "415"} for t, k in spans)


def _regions(text, inside_words):
    sents = split_sentences(text)
    col = _Collector(text, sents)
    toks = tokens(text, *sents[0])
    region_candidates(text, 0, sents[0], toks, [text[s:e] in inside_words for s, e in toks], col, {"of", "and", "the"})
    return {c.text for c in col.found[0].values()}


def test_regions_keep_list_items_nested_mentions_and_symbols_separable():
    got = _regions("We visited Paris, London, Berlin and Rome.", {"Paris", "London", "Berlin", "Rome"})
    assert {"Paris", "London", "Berlin", "Rome"} <= got
    got = _regions("The Bank of America office.", {"Bank", "America"})  # "of" absorbed as a connector
    assert {"Bank of America", "America", "Bank"} <= got
    got = _regions("It cost $1.2 billion, up 12% on R$ 180.000,00; mail bob@acme.com or #4521.",
                   {"1.2", "billion", "12", "R", "180.000,00", "bob", "acme.com", "4521"})
    assert {"$1.2 billion", "12%", "R$ 180.000,00", "bob@acme.com", "#4521"} <= got


def test_regions_add_a_period_only_after_abbreviations():
    assert "metformin." not in _regions("The patient takes metformin.", {"metformin"})
    assert "Acme Ltd." in _regions("We hired Acme Ltd. today.", {"Acme", "Ltd"})
