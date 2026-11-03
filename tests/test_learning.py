import pytest

from dactilograph.learning import LearnSession

LETTERS = ["A", "B", "C"]


def test_session_starts_at_the_first_letter_or_the_chosen_one():
    assert LearnSession(LETTERS).target == "A"
    assert LearnSession(LETTERS, start="C").target == "C"
    assert LearnSession(LETTERS, start="Z").target == "A"


def test_navigation_wraps_around():
    session = LearnSession(LETTERS)
    session.previous()
    assert session.target == "C"
    session.next()
    assert session.target == "A"


def test_only_the_target_letter_counts_as_success():
    session = LearnSession(LETTERS)
    assert not session.check(["B"], now_s=1.0)
    assert session.check(["B", "A"], now_s=2.0)
    assert session.success_s == 2.0


def test_success_is_reported_once_and_cleared_when_moving():
    session = LearnSession(LETTERS)
    session.check(["A"], now_s=1.0)
    assert not session.check(["A"], now_s=2.0)
    session.next()
    assert not session.succeeded
    session.select("A")
    assert session.target == "A"
    assert not session.succeeded


def test_session_needs_letters():
    with pytest.raises(ValueError):
        LearnSession([])
