import numpy as np

from dactilograph.dataset import DYNAMIC_LETTERS
from dactilograph.ui.base import ALPHABET, Context, Screen, letter_scale, ring_progress
from dactilograph.ui.chalk import (
    BLUE,
    DIM,
    PINK,
    SANS,
    SCRIPT,
    WHITE,
    YELLOW,
    Chalk,
    fit_hand,
)
from dactilograph.ui.widgets import Button

OPTIONS = (
    ("1", "Aprender", "las 30 letras, una a una", BLUE, "aprender"),
    ("2", "Reconocer", "haz letras y mira cuál ve", WHITE, "reconocer"),
    ("3", "Jugar", "explota burbujas signando", PINK, "jugar"),
)
OPTION_KEYS = {ord("1"): "aprender", ord("2"): "reconocer", ord("3"): "jugar"}
FIRST_OPTION_Y = 365
OPTION_STEP = 95
SHOWCASE_S = 1.6
SHOWCASE_TOP_LEFT = (850, 110)
SHOWCASE_SIZE = 360


class MenuScreen(Screen):
    tray_active = "menu"

    def __init__(self) -> None:
        self._focus = 0
        self._focus_since: float | None = None

    def enter(self, context: Context) -> None:
        self._focus_since = context.now

    def key_action(self, key: int) -> str | None:
        return OPTION_KEYS.get(key)

    def draw(self, canvas: np.ndarray, chalk: Chalk, context: Context) -> list[Button]:
        chalk.text("Dactilograph", (90, 190), 4.0, WHITE, 6, SCRIPT)
        chalk.line((95, 215), (720, 222), YELLOW, 3)
        chalk.text("el alfabeto dactilológico de la LSE, con tu mano", (100, 268), 0.8, DIM, 1)

        buttons = [
            Button(70, FIRST_OPTION_Y + index * OPTION_STEP - 62, 690, 84, option[-1])
            for index, option in enumerate(OPTIONS)
        ]
        hovered = next((i for i, b in enumerate(buttons) if b.contains(context.mouse)), None)
        if hovered is not None and hovered != self._focus:
            self._focus, self._focus_since = hovered, context.now

        for index, (key, name, note, color, _) in enumerate(OPTIONS):
            y = FIRST_OPTION_Y + index * OPTION_STEP
            chalk.text(key, (95, y), 1.2, DIM, 2)
            width, _ = chalk.text(name, (140, y), 2.0, color, 3, SCRIPT)
            chalk.text(note, (470, y - 8), 0.7, DIM, 1)
            if index == self._focus:
                progress = ring_progress(self._focus_since, context.now)
                chalk.ring((140 + width // 2 - 20, y - 20), (width // 2 + 70, 48), YELLOW, progress)

        self._draw_showcase(chalk, context)
        learned = len(context.progress.learned & set(ALPHABET))
        chalk.text(
            f"aprendidas: {learned} de {len(ALPHABET)}", (1230, 575), 0.7, WHITE, 1, SANS, "right"
        )
        chalk.text(
            f"récord: {context.progress.high_score}", (1230, 610), 0.7, WHITE, 1, SANS, "right"
        )
        return buttons

    def _draw_showcase(self, chalk: Chalk, context: Context) -> None:
        shown = [letter for letter in ALPHABET if letter in context.references]
        if not shown:
            return
        letter = shown[int(context.now / SHOWCASE_S) % len(shown)]
        points = fit_hand(context.references[letter], SHOWCASE_TOP_LEFT, SHOWCASE_SIZE)
        chalk.hand(points, WHITE, 3)
        x, y = SHOWCASE_TOP_LEFT[0] + SHOWCASE_SIZE, SHOWCASE_TOP_LEFT[1] + SHOWCASE_SIZE + 30
        chalk.text(letter, (x, y), letter_scale(letter, 3.0), YELLOW, 4, SCRIPT, "right")
        if letter in DYNAMIC_LETTERS:
            chalk.text("con movimiento", (x, y + 32), 0.6, BLUE, 1, SANS, "right")
