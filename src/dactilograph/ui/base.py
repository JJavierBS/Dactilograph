from dataclasses import dataclass

import cv2
import numpy as np

from dactilograph.dataset import DYNAMIC_LETTERS, STATIC_LETTERS
from dactilograph.live_recognition import Reading
from dactilograph.progress import Progress
from dactilograph.ui.chalk import (
    BLUE,
    DIM,
    PINK,
    SCRIPT,
    WHITE,
    YELLOW,
    Chalk,
    hand_in_box,
    text_size,
)
from dactilograph.ui.widgets import Button

ALPHABET = tuple(
    sorted(
        (*STATIC_LETTERS, *DYNAMIC_LETTERS),
        key=lambda letter: "NZ" if letter == "Ñ" else letter,
    )
)
LETTER_COLORS = (WHITE, YELLOW, PINK, BLUE)
RING_DRAW_S = 0.35
LEFT_KEYS = {65361, 2424832, 81}
RIGHT_KEYS = {65363, 2555904, 83}
BACKSPACE_KEYS = {8, 65288}
SPACE_KEYS = {32}
Mouse = tuple[int, int] | None


@dataclass
class Context:
    now: float
    frame: np.ndarray | None
    reading: Reading | None
    mouse: Mouse
    references: dict[str, np.ndarray]
    progress: Progress


class Screen:
    tray_active = ""
    tray_items: tuple[tuple[str, str, str], ...] = ()
    uses_camera = False

    def enter(self, context: Context) -> None:
        pass

    def key_action(self, key: int) -> str | None:
        return None

    def handle(self, action: str, context: Context) -> str | None:
        return action

    def update(self, context: Context) -> None:
        pass

    def draw(self, canvas: np.ndarray, chalk: Chalk, context: Context) -> list[Button]:
        return []


def letter_color(letter: str) -> tuple[int, int, int]:
    return LETTER_COLORS[ALPHABET.index(letter) % len(LETTER_COLORS)]


def ring_progress(since_s: float | None, now_s: float) -> float:
    if since_s is None:
        return 0.0
    return min((now_s - since_s) / RING_DRAW_S, 1.0)


def letter_scale(letter: str, single_scale: float) -> float:
    return single_scale if len(letter) == 1 else single_scale * 0.62


def place_camera(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def chalk_user_hand(
    chalk: Chalk, reading: Reading | None, top_left: tuple[int, int], size: tuple[int, int]
) -> None:
    if reading is None or reading.hand is None:
        return
    chalk.hand(hand_in_box(reading.hand.landmarks, top_left, size), YELLOW, 3)


def no_hand_note(chalk: Chalk, reading: Reading | None, center: tuple[int, int]) -> None:
    if reading is not None and reading.hand is None:
        chalk.text("no veo tu mano", center, 1.0, DIM, 2, SCRIPT, "center")


def chalk_word_button(
    chalk: Chalk,
    label: str,
    key: str,
    center: tuple[int, int],
    color: tuple[int, int, int],
    action: str,
    mouse: Mouse,
    hover_since: dict[str, float],
    now_s: float,
    emphasized: bool = False,
) -> Button:
    width, height = text_size(label, 1.15, 2, SCRIPT)
    x, y = center
    button = Button(x - width // 2 - 24, y - height - 24, width + 48, height + 64, action)
    hovered = button.contains(mouse)
    if hovered:
        hover_since.setdefault(action, now_s)
    else:
        hover_since.pop(action, None)
    chalk.text(label, (x, y), 1.15, YELLOW if emphasized else color, 2, SCRIPT, "center")
    chalk.text(key, (x, y + 30), 0.6, DIM, 1, align="center")
    if hovered or emphasized:
        progress = ring_progress(hover_since.get(action), now_s) if hovered else 1.0
        chalk.ring(
            (x, y - height // 2 + 12), (width // 2 + 26, height // 2 + 34), YELLOW, progress, 2
        )
    return button
