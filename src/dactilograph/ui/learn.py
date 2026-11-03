import numpy as np

from dactilograph.dataset import DYNAMIC_LETTERS
from dactilograph.learning import LearnSession
from dactilograph.ui.base import (
    ALPHABET,
    BACKSPACE_KEYS,
    LEFT_KEYS,
    RIGHT_KEYS,
    SPACE_KEYS,
    Context,
    Screen,
    chalk_user_hand,
    chalk_word_button,
    letter_scale,
    no_hand_note,
    place_camera,
    ring_progress,
)
from dactilograph.ui.chalk import (
    BLUE,
    DIM,
    FAINT,
    SANS,
    SCRIPT,
    WHITE,
    YELLOW,
    Chalk,
    fit_hand,
    pin_photo,
    text_size,
)
from dactilograph.ui.widgets import Button

GUIDE_LEFT = 50
GUIDE_Y = 64
GUIDE_GAP = 15
CAMERA_TOP_LEFT = (420, 122)
CAMERA_SIZE = (640, 480)
COLUMN_X = 1160
LEFT_X = 215
REFERENCE_SIZE = 200


class LearnScreen(Screen):
    tray_active = "aprender"
    uses_camera = True

    def __init__(self) -> None:
        self._session = LearnSession(ALPHABET)
        self._hover_since: dict[str, float] = {}

    def enter(self, context: Context) -> None:
        pending = [letter for letter in ALPHABET if letter not in context.progress.learned]
        if pending and self._session.target in context.progress.learned:
            self._session.select(pending[0])

    def key_action(self, key: int) -> str | None:
        if key in SPACE_KEYS or key in RIGHT_KEYS:
            return "siguiente"
        if key in BACKSPACE_KEYS or key in LEFT_KEYS:
            return "anterior"
        return None

    def handle(self, action: str, context: Context) -> str | None:
        if action == "siguiente":
            self._session.next()
        elif action == "anterior":
            self._session.previous()
        elif action.startswith("letra:"):
            self._session.select(action.removeprefix("letra:"))
        else:
            return action
        return None

    def update(self, context: Context) -> None:
        if context.reading is None:
            return
        if self._session.check(context.reading.confirmed, context.now):
            context.progress.learn(self._session.target)
            context.progress.save()

    def draw(self, canvas: np.ndarray, chalk: Chalk, context: Context) -> list[Button]:
        buttons = self._draw_guide(chalk, context)
        self._draw_target(chalk, context)
        if context.frame is not None:
            pin_photo(canvas, place_camera(context.frame, CAMERA_SIZE), CAMERA_TOP_LEFT)
        chalk_user_hand(chalk, context.reading, CAMERA_TOP_LEFT, CAMERA_SIZE)
        no_hand_note(chalk, context.reading, (CAMERA_TOP_LEFT[0] + 320, CAMERA_TOP_LEFT[1] + 250))
        self._draw_verdict(chalk, context)
        succeeded = self._session.succeeded
        buttons.append(
            chalk_word_button(
                chalk,
                "siguiente",
                "espacio",
                (COLUMN_X, 560),
                WHITE,
                "siguiente",
                context.mouse,
                self._hover_since,
                context.now,
                emphasized=succeeded,
            )
        )
        buttons.append(
            chalk_word_button(
                chalk,
                "anterior",
                "retroceso",
                (COLUMN_X, 470),
                DIM,
                "anterior",
                context.mouse,
                self._hover_since,
                context.now,
            )
        )
        return buttons

    def _draw_guide(self, chalk: Chalk, context: Context) -> list[Button]:
        buttons = []
        x = GUIDE_LEFT
        for letter in ALPHABET:
            learned = letter in context.progress.learned
            is_target = letter == self._session.target
            width, height = text_size(letter, 0.7, 2)
            button = Button(
                x - 6, GUIDE_Y - height - 10, width + 12, height + 30, f"letra:{letter}"
            )
            color = (
                YELLOW if is_target else WHITE if learned or button.contains(context.mouse) else DIM
            )
            chalk.text(letter, (x, GUIDE_Y), 0.7, color, 2)
            dot = (x + width // 2, GUIDE_Y + 14)
            chalk.circle(dot, 3, color if learned else FAINT, -1 if learned else 1)
            if is_target:
                chalk.ring(
                    (x + width // 2, GUIDE_Y - height // 2 + 2),
                    (width // 2 + 12, 22),
                    YELLOW,
                    1.0,
                    2,
                )
            buttons.append(button)
            x += width + GUIDE_GAP
        learned_count = len(context.progress.learned & set(ALPHABET))
        position = f"{self._session.index + 1} de {len(ALPHABET)}  ·  {learned_count} aprendidas"
        chalk.text(position, (1240, GUIDE_Y), 0.55, DIM, 1, SANS, "right")
        chalk.line((40, GUIDE_Y + 28), (1240, GUIDE_Y + 32), FAINT, 1)
        return buttons

    def _draw_target(self, chalk: Chalk, context: Context) -> None:
        target = self._session.target
        chalk.text(target, (LEFT_X, 330), letter_scale(target, 8.5), WHITE, 14, SCRIPT, "center")
        dynamic = target in DYNAMIC_LETTERS
        chalk.text(
            "con movimiento" if dynamic else "estática",
            (LEFT_X, 385),
            1.3,
            BLUE,
            2,
            SCRIPT,
            "center",
        )
        if target in context.references:
            top_left = (LEFT_X - REFERENCE_SIZE // 2, 395)
            chalk.hand(fit_hand(context.references[target], top_left, REFERENCE_SIZE), WHITE, 3)
        note = "empieza así y haz el gesto" if dynamic else "así se hace"
        chalk.text(note, (LEFT_X, 625), 0.9, DIM, 1, SCRIPT, "center")

    def _draw_verdict(self, chalk: Chalk, context: Context) -> None:
        target = self._session.target
        if self._session.succeeded:
            chalk.text(
                target, (COLUMN_X, 250), letter_scale(target, 3.2), YELLOW, 5, SCRIPT, "center"
            )
            progress = ring_progress(self._session.success_s, context.now)
            chalk.ring((COLUMN_X, 220), (62, 66), YELLOW, progress, 3, tilt=8)
            chalk.text("¡Bien!", (COLUMN_X, 340), 1.5, YELLOW, 2, SCRIPT, "center")
            return
        chalk.text("veo", (COLUMN_X, 150), 0.6, DIM, 1, SANS, "center")
        prediction = context.reading.prediction if context.reading else None
        if prediction is None:
            chalk.text("-", (COLUMN_X, 250), 3.2, DIM, 4, SCRIPT, "center")
            return
        color = WHITE if prediction.letter == target else DIM
        chalk.text(
            prediction.letter,
            (COLUMN_X, 250),
            letter_scale(prediction.letter, 3.2),
            color,
            5,
            SCRIPT,
            "center",
        )
        confidence = f"{prediction.confidence:.2f}".replace(".", ",")
        chalk.text(confidence, (COLUMN_X, 315), 0.9, WHITE, 2, SANS, "center")
        chalk.text("seguridad", (COLUMN_X, 340), 0.5, DIM, 1, SANS, "center")
