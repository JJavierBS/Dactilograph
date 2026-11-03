from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

START_LIVES = 3
POPS_PER_LEVEL = 5
START_SPEED = 0.08
SPEED_GROWTH = 1.15
START_SPAWN_INTERVAL_S = 4.0
SPAWN_INTERVAL_DECAY = 0.9
MIN_SPAWN_INTERVAL_S = 1.5
MAX_BUBBLES = 4
BUBBLE_RADIUS = 0.07
MIN_X = 0.12
MAX_X = 0.88
HELP_AFTER_S = 6.0
POP_DURATION_S = 0.4
MISS_DURATION_S = 1.2


@dataclass
class Bubble:
    letter: str
    x: float
    y: float
    spawned_s: float


@dataclass(frozen=True)
class Pop:
    letter: str
    x: float
    y: float
    popped_s: float


class BubbleGame:
    def __init__(self, letters: Sequence[str], rng: np.random.Generator, now_s: float) -> None:
        if not letters:
            raise ValueError("The game needs at least one letter")
        self._letters = list(letters)
        self._rng = rng
        self.bubbles: list[Bubble] = []
        self.pops: list[Pop] = []
        self.misses: list[Pop] = []
        self.score = 0
        self.lives = START_LIVES
        self._last_update_s = now_s
        self._next_spawn_s = now_s

    @property
    def level(self) -> int:
        return 1 + self.score // POPS_PER_LEVEL

    @property
    def speed(self) -> float:
        return START_SPEED * SPEED_GROWTH ** (self.level - 1)

    @property
    def spawn_interval_s(self) -> float:
        interval = START_SPAWN_INTERVAL_S * SPAWN_INTERVAL_DECAY ** (self.level - 1)
        return max(interval, MIN_SPAWN_INTERVAL_S)

    @property
    def is_over(self) -> bool:
        return self.lives <= 0

    def update(self, now_s: float) -> None:
        elapsed_s = now_s - self._last_update_s
        self._last_update_s = now_s
        self.pops = [pop for pop in self.pops if now_s - pop.popped_s < POP_DURATION_S]
        self.misses = [miss for miss in self.misses if now_s - miss.popped_s < MISS_DURATION_S]
        if self.is_over:
            return

        for bubble in self.bubbles:
            bubble.y -= self.speed * elapsed_s
        escaped = [bubble for bubble in self.bubbles if bubble.y < -BUBBLE_RADIUS]
        self.bubbles = [bubble for bubble in self.bubbles if bubble.y >= -BUBBLE_RADIUS]
        self.lives = max(self.lives - len(escaped), 0)
        self.misses.extend(Pop(bubble.letter, bubble.x, 0.0, now_s) for bubble in escaped)
        if self.is_over:
            self.bubbles = []
            return

        if now_s >= self._next_spawn_s:
            self._spawn(now_s)
            self._next_spawn_s = now_s + self.spawn_interval_s

    def try_pop(self, letter: str, now_s: float) -> bool:
        if self.is_over:
            return False
        matching = [bubble for bubble in self.bubbles if bubble.letter == letter]
        if not matching:
            return False
        target = min(matching, key=lambda bubble: bubble.y)
        self.bubbles.remove(target)
        self.pops.append(Pop(target.letter, target.x, target.y, now_s))
        self.score += 1
        return True

    def help_letter(self, now_s: float) -> str | None:
        waiting = [bubble for bubble in self.bubbles if now_s - bubble.spawned_s >= HELP_AFTER_S]
        if not waiting:
            return None
        return min(waiting, key=lambda bubble: bubble.y).letter

    def _spawn(self, now_s: float) -> None:
        if len(self.bubbles) >= MAX_BUBBLES:
            return
        on_screen = {bubble.letter for bubble in self.bubbles}
        available = [letter for letter in self._letters if letter not in on_screen]
        if not available:
            return
        letter = available[self._rng.integers(len(available))]
        x = self._rng.uniform(MIN_X, MAX_X)
        self.bubbles.append(Bubble(letter, x, 1 + BUBBLE_RADIUS, now_s))
