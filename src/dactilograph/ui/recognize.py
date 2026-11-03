from collections import deque

import numpy as np

from dactilograph.ui.base import (
    Context,
    Screen,
    chalk_user_hand,
    letter_scale,
    no_hand_note,
    place_camera,
    ring_progress,
)
from dactilograph.ui.chalk import DIM, SANS, SCRIPT, WHITE, YELLOW, Chalk, pin_photo
from dactilograph.ui.widgets import Button

CAMERA_TOP_LEFT = (60, 120)
CAMERA_SIZE = (680, 510)
COLUMN_X = 1000
HISTORY_LENGTH = 6
HISTORY_STEP = 72


class RecognizeScreen(Screen):
    tray_active = "reconocer"
    uses_camera = True

    def __init__(self) -> None:
        self._history: deque[tuple[str, float]] = deque(maxlen=HISTORY_LENGTH)

    def update(self, context: Context) -> None:
        if context.reading is None:
            return
        for letter in context.reading.confirmed:
            self._history.append((letter, context.now))

    def draw(self, canvas: np.ndarray, chalk: Chalk, context: Context) -> list[Button]:
        chalk.text("Haz cualquier letra delante de la cámara", (60, 78), 1.2, WHITE, 2, SCRIPT)
        if context.frame is not None:
            pin_photo(canvas, place_camera(context.frame, CAMERA_SIZE), CAMERA_TOP_LEFT)
        chalk_user_hand(chalk, context.reading, CAMERA_TOP_LEFT, CAMERA_SIZE)
        no_hand_note(chalk, context.reading, (CAMERA_TOP_LEFT[0] + 340, CAMERA_TOP_LEFT[1] + 265))

        chalk.text("veo", (COLUMN_X, 150), 0.6, DIM, 1, SANS, "center")
        prediction = context.reading.prediction if context.reading else None
        if prediction is None:
            chalk.text("-", (COLUMN_X, 330), 6.0, DIM, 8, SCRIPT, "center")
        else:
            scale = letter_scale(prediction.letter, 6.0)
            chalk.text(prediction.letter, (COLUMN_X, 330), scale, WHITE, 10, SCRIPT, "center")
            confidence = f"{prediction.confidence:.2f}".replace(".", ",")
            chalk.text(confidence, (COLUMN_X, 395), 1.0, WHITE, 2, SANS, "center")
            chalk.text("seguridad", (COLUMN_X, 422), 0.5, DIM, 1, SANS, "center")

        chalk.text("lo último que has hecho", (COLUMN_X, 500), 0.9, DIM, 1, SCRIPT, "center")
        start_x = COLUMN_X - (HISTORY_LENGTH - 1) * HISTORY_STEP // 2
        for index, (letter, made_s) in enumerate(self._history):
            x = start_x + index * HISTORY_STEP
            newest = index == len(self._history) - 1
            color = YELLOW if newest else WHITE
            chalk.text(letter, (x, 585), letter_scale(letter, 1.6), color, 3, SCRIPT, "center")
            if newest:
                chalk.ring(
                    (x, 568), (30, 34), YELLOW, ring_progress(made_s, context.now), 2, tilt=8
                )
        return []
