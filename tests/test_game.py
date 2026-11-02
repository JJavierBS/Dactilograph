import numpy as np
import pytest

from dactilograph.game import (
    BUBBLE_RADIUS,
    HELP_AFTER_S,
    MAX_BUBBLES,
    MAX_X,
    MIN_X,
    POP_DURATION_S,
    POPS_PER_LEVEL,
    START_LIVES,
    START_SPAWN_INTERVAL_S,
    Bubble,
    BubbleGame,
)

LETTERS = ["A", "B", "C", "D", "E", "F"]


def new_game(letters=LETTERS, seed=0):
    return BubbleGame(letters, np.random.default_rng(seed), now_s=0.0)


def test_first_bubble_appears_at_the_bottom_on_the_first_update():
    game = new_game()
    game.update(0.0)
    [bubble] = game.bubbles
    assert bubble.letter in LETTERS
    assert bubble.y == pytest.approx(1 + BUBBLE_RADIUS)
    assert MIN_X <= bubble.x <= MAX_X


def test_bubbles_rise_over_time():
    game = new_game()
    game.update(0.0)
    start = game.bubbles[0].y
    game.update(1.0)
    assert game.bubbles[0].y == pytest.approx(start - game.speed)


def test_new_bubbles_appear_at_the_spawn_interval():
    game = new_game()
    game.update(0.0)
    game.update(START_SPAWN_INTERVAL_S - 0.1)
    assert len(game.bubbles) == 1
    game.update(START_SPAWN_INTERVAL_S)
    assert len(game.bubbles) == 2


def test_letters_on_screen_are_never_repeated():
    game = new_game(letters=["A", "B"])
    for second in range(0, 40, 4):
        game.update(float(second))
        letters = [bubble.letter for bubble in game.bubbles]
        assert len(letters) == len(set(letters))


def test_there_are_never_more_than_the_maximum_bubbles():
    game = new_game(letters=[chr(ord("A") + i) for i in range(20)])
    game.lives = 100
    for step in range(200):
        game.update(step * 0.5)
        assert len(game.bubbles) <= MAX_BUBBLES


def test_making_the_letter_pops_its_bubble():
    game = new_game()
    game.bubbles = [Bubble("A", 0.5, 0.5, 0.0), Bubble("B", 0.3, 0.6, 0.0)]
    assert game.try_pop("A", now_s=1.0)
    assert [bubble.letter for bubble in game.bubbles] == ["B"]
    assert game.score == 1
    assert [pop.letter for pop in game.pops] == ["A"]


def test_a_letter_without_bubble_does_nothing():
    game = new_game()
    game.bubbles = [Bubble("A", 0.5, 0.5, 0.0)]
    assert not game.try_pop("Z", now_s=1.0)
    assert game.score == 0
    assert len(game.bubbles) == 1


def test_pop_animation_disappears_after_its_duration():
    game = new_game()
    game.bubbles = [Bubble("A", 0.5, 0.5, 0.0)]
    game.try_pop("A", now_s=0.0)
    game.update(POP_DURATION_S / 2)
    assert len(game.pops) == 1
    game.update(POP_DURATION_S)
    assert game.pops == []


def test_escaped_bubble_costs_a_life():
    game = new_game()
    game.bubbles = [Bubble("A", 0.5, -BUBBLE_RADIUS + 0.001, 0.0)]
    game.update(1.0)
    assert game.lives == START_LIVES - 1
    assert "A" not in [bubble.letter for bubble in game.bubbles]


def test_game_ends_without_lives_and_ignores_letters():
    game = new_game()
    game.lives = 1
    game.bubbles = [Bubble("A", 0.5, -BUBBLE_RADIUS + 0.001, 0.0), Bubble("B", 0.5, 0.5, 0.0)]
    game.update(1.0)
    assert game.is_over
    assert game.bubbles == []
    assert not game.try_pop("B", now_s=1.0)
    game.update(10.0)
    assert game.bubbles == []


def test_levels_make_the_game_faster():
    game = new_game()
    speed, interval = game.speed, game.spawn_interval_s
    game.score = POPS_PER_LEVEL
    assert game.level == 2
    assert game.speed > speed
    assert game.spawn_interval_s < interval


def test_help_shows_the_highest_bubble_that_has_waited_too_long():
    game = new_game()
    game.bubbles = [
        Bubble("A", 0.5, 0.4, spawned_s=0.0),
        Bubble("B", 0.5, 0.2, spawned_s=0.0),
        Bubble("C", 0.5, 0.1, spawned_s=HELP_AFTER_S),
    ]
    assert game.help_letter(HELP_AFTER_S - 0.1) is None
    assert game.help_letter(HELP_AFTER_S) == "B"


def test_game_needs_letters():
    with pytest.raises(ValueError):
        new_game(letters=[])
