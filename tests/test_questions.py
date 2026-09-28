import pytest

from typeextract.errors import ResponseValidationError
from typeextract.questions import Answer, ChoiceTask, choice, noul, parse_answer, score


def test_choice_answer_is_normalised_and_completed():
    q = choice("which?", {"a": None, "b": None, "none": "neither"})
    ans = parse_answer({"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.2}}, q)
    assert ans.choice == "a"
    assert ans.probabilities == pytest.approx({"a": 0.75, "b": 0.25, "none": 0.0})
    assert ans.top2() == ("a", pytest.approx(0.75), pytest.approx(0.25))


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"type": "noul", "noul": 0.5},
        {"type": "choice", "choice": "zzz", "probabilities": {"a": 1.0}},
        {"type": "choice", "probabilities": {"a": 0, "b": 0}},
        {"type": "choice", "probabilities": {"a": float("nan")}},
        {"type": "choice", "probabilities": "a"},
    ],
)
def test_invalid_choice_answers_raise(raw):
    with pytest.raises(ResponseValidationError):
        parse_answer(raw, choice("which?", {"a": None, "b": None}))


def test_noul_is_clamped_and_score_is_computed_when_missing():
    assert parse_answer({"type": "noul", "noul": 1.3}, noul("?")).noul == 1.0
    ans = parse_answer({"type": "score", "probabilities": {"0": 0.5, "2": 0.5}}, score("?", ["lo", "mid", "hi"]))
    assert ans.score == pytest.approx(1.0) and set(ans.probabilities) == {"0", "1", "2"}


def test_answer_roundtrip():
    a = Answer("choice", choice="x", probabilities={"x": 1.0}, confidence=0.9)
    assert Answer.from_dict(a.to_dict()) == a


def _answer_all(task, pick):
    """Answer every open question of the task, favouring ``pick`` when offered."""
    out = {}
    for key, q in task.questions().items():
        options = list(q["criteria"])
        top = pick if pick in options else "none"
        out[key] = Answer("choice", choice=top, probabilities={o: (0.9 if o == top else 0.1 / (len(options) - 1)) for o in options})
    return out


def test_small_choice_is_a_single_question():
    task = ChoiceTask("k", "which?", {f"o{i}": None for i in range(254)}, escape=("none", "no"))
    qs = task.questions()
    assert list(qs) == ["k"] and len(qs["k"]["criteria"]) == 255
    task.update(_answer_all(task, "o7"))
    assert task.done and task.result == "o7" and task.probability == pytest.approx(0.9)


def test_tournament_lifts_the_255_option_cap():
    task = ChoiceTask("k", "which?", {f"o{i}": None for i in range(600)}, escape=("none", "no"))
    rounds = 0
    while not task.done:
        qs = task.questions()
        assert all(len(q["criteria"]) <= 255 for q in qs.values())
        if rounds == 0:
            assert len(qs) == 3
        task.update(_answer_all(task, "o523"))
        rounds += 1
    assert task.result == "o523" and rounds == 2


def test_tournament_all_escape():
    task = ChoiceTask("k", "which?", {f"o{i}": None for i in range(300)}, escape=("none", "no"))
    task.update(_answer_all(task, "missing"))
    assert task.done and task.result is None


def test_escape_collision_is_rejected():
    with pytest.raises(ValueError):
        ChoiceTask("k", "which?", {"none": None}, escape=("none", "no"))
