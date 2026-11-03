import math

import numpy as np

from dactilograph.dataset import DYNAMIC_LETTERS
from dactilograph.game import (
    BUBBLE_RADIUS,
    MISS_DURATION_S,
    POP_DURATION_S,
    START_LIVES,
    BubbleGame,
)
from dactilograph.ui.base import (
    ALPHABET,
    Context,
    Screen,
    chalk_user_hand,
    chalk_word_button,
    letter_color,
    letter_scale,
    place_camera,
    ring_progress,
)
from dactilograph.ui.chalk import (
    BOARD,
    DIM,
    FAINT,
    PINK,
    SANS,
    SCRIPT,
    WHITE,
    YELLOW,
    Chalk,
    fit_hand,
    soft_mask,
)
from dactilograph.ui.widgets import Button

ARENA_TOP_LEFT = (240, 36)
ARENA_SIZE = (800, 600)
CAMERA_WEIGHT = 0.6
ARENA_FEATHER = 40
GAME_OVER_FADE = 0.65
LEFT_X = 60
RIGHT_X = 1148
HELP_SIZE = 170
HELP_X = 135
BURST_RAYS = 8
RESTART_KEY = ord("r")


class PlayScreen(Screen):
    tray_active = "jugar"
    uses_camera = True

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._game: BubbleGame | None = None
        self._new_record = False
        self._previous_record = 0
        self._over_since: float | None = None
        self._hover_since: dict[str, float] = {}

    def enter(self, context: Context) -> None:
        self._start(context)

    def key_action(self, key: int) -> str | None:
        return "otra" if key == RESTART_KEY else None

    def handle(self, action: str, context: Context) -> str | None:
        if action == "otra":
            self._start(context)
            return None
        return action

    def update(self, context: Context) -> None:
        game = self._game
        if game is None:
            return
        if context.reading is not None:
            for letter in context.reading.confirmed:
                game.try_pop(letter, context.now)
        game.update(context.now)
        if game.is_over and self._over_since is None:
            self._over_since = context.now
            self._new_record = context.progress.record_score(game.score)
            if self._new_record:
                context.progress.save()

    def draw(self, canvas: np.ndarray, chalk: Chalk, context: Context) -> list[Button]:
        game = self._game
        if game is None:
            return []
        self._draw_arena(canvas, context)
        x0, y0 = ARENA_TOP_LEFT
        chalk_user_hand(chalk, context.reading, ARENA_TOP_LEFT, ARENA_SIZE)
        self._draw_bubbles(chalk, game, context.now)
        self._draw_scores(chalk, game, context)
        self._draw_side(chalk, game, context)
        if not game.is_over:
            return []
        return self._draw_game_over(canvas, chalk, game, context, (x0, y0))

    def _start(self, context: Context) -> None:
        self._game = BubbleGame(ALPHABET, self._rng, context.now)
        self._previous_record = context.progress.high_score
        self._new_record = False
        self._over_since = None

    def _draw_arena(self, canvas: np.ndarray, context: Context) -> None:
        x, y = ARENA_TOP_LEFT
        width, height = ARENA_SIZE
        region = canvas[y : y + height, x : x + width]
        feather = soft_mask(
            width - 2 * ARENA_FEATHER, height - 2 * ARENA_FEATHER, ARENA_FEATHER, ARENA_FEATHER / 2
        )
        if context.frame is not None:
            camera = place_camera(context.frame, ARENA_SIZE)
            weight = feather * CAMERA_WEIGHT
            region[:] = (camera * weight + region * (1 - weight)).astype(np.uint8)
        if self._game is not None and self._game.is_over:
            weight = feather * GAME_OVER_FADE
            region[:] = (region * (1 - weight) + np.array(BOARD) * weight).astype(np.uint8)

    def _to_arena(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(ARENA_TOP_LEFT[0] + x * ARENA_SIZE[0]),
            int(ARENA_TOP_LEFT[1] + y * ARENA_SIZE[1]),
        )

    def _draw_bubbles(self, chalk: Chalk, game: BubbleGame, now_s: float) -> None:
        radius = int(BUBBLE_RADIUS * ARENA_SIZE[1])
        for bubble in game.bubbles:
            center = self._to_arena(bubble.x, bubble.y)
            color = letter_color(bubble.letter)
            chalk.circle(center, radius, color, 3)
            chalk.ring((center[0] - 16, center[1] - 18), (14, 9), color, 0.3, 2, tilt=-40)
            scale = letter_scale(bubble.letter, 1.7)
            chalk.text(
                bubble.letter, (center[0], center[1] + 20), scale, color, 3, SCRIPT, "center"
            )
        for pop in game.pops:
            center = self._to_arena(pop.x, pop.y)
            progress = (now_s - pop.popped_s) / POP_DURATION_S
            inner, outer = radius * (0.6 + progress), radius * (1.0 + 1.2 * progress)
            color = letter_color(pop.letter)
            for ray in range(BURST_RAYS):
                angle = 2 * math.pi * ray / BURST_RAYS
                start = (
                    int(center[0] + inner * math.cos(angle)),
                    int(center[1] + inner * math.sin(angle)),
                )
                end = (
                    int(center[0] + outer * math.cos(angle)),
                    int(center[1] + outer * math.sin(angle)),
                )
                chalk.line(start, end, color, 3)
        for miss in game.misses:
            center = self._to_arena(miss.x, 0)
            center = (center[0], center[1] + radius + 6)
            fade = 1 - (now_s - miss.popped_s) / MISS_DURATION_S
            chalk.circle(center, radius, tuple(int(c * fade) for c in FAINT), 2)
            chalk.cross(center, int(radius * 0.75), tuple(int(c * fade) for c in PINK), 3)

    def _draw_scores(self, chalk: Chalk, game: BubbleGame, context: Context) -> None:
        chalk.text(f"Puntos {game.score}", (LEFT_X, 100), 1.3, WHITE, 2, SCRIPT)
        chalk.text(f"Nivel {game.level}", (LEFT_X, 150), 1.3, WHITE, 2, SCRIPT)
        if self._new_record:
            label_width, _ = chalk.text("récord", (LEFT_X, 192), 0.65, DIM, 1)
            old_x = LEFT_X + label_width + 10
            width, height = chalk.text(str(self._previous_record), (old_x, 192), 0.65, DIM, 1)
            middle = 192 - height // 2
            chalk.line((old_x - 4, middle + 4), (old_x + width + 4, middle - 4), PINK, 2)
            new_x = old_x + width + 14
            chalk.text(str(context.progress.high_score), (new_x, 194), 0.9, YELLOW, 2, SCRIPT)
        else:
            chalk.text(f"récord {context.progress.high_score}", (LEFT_X, 192), 0.65, DIM, 1)
        self._draw_help(chalk, game, context)

    def _draw_side(self, chalk: Chalk, game: BubbleGame, context: Context) -> None:
        chalk.text("vidas", (RIGHT_X, 80), 0.6, DIM, 1, SANS, "center")
        for life in range(START_LIVES):
            center = (RIGHT_X - 44 + life * 44, 112)
            chalk.circle(center, 13, PINK, -1 if life < game.lives else 2)
        if game.is_over:
            return
        prediction = context.reading.prediction if context.reading else None
        chalk.text("haces", (RIGHT_X, 190), 0.6, DIM, 1, SANS, "center")
        if prediction is None:
            chalk.text("-", (RIGHT_X, 270), 2.6, DIM, 4, SCRIPT, "center")
        else:
            scale = letter_scale(prediction.letter, 2.6)
            chalk.text(prediction.letter, (RIGHT_X, 270), scale, YELLOW, 4, SCRIPT, "center")

    def _draw_help(self, chalk: Chalk, game: BubbleGame, context: Context) -> None:
        if game.is_over:
            return
        help_letter = game.help_letter(context.now)
        if help_letter is None or help_letter not in context.references:
            return
        chalk.text("así se hace la", (HELP_X, 300), 0.9, WHITE, 2, SCRIPT, "center")
        scale = letter_scale(help_letter, 1.8)
        chalk.text(help_letter, (HELP_X, 360), scale, YELLOW, 3, SCRIPT, "center")
        if help_letter in DYNAMIC_LETTERS:
            chalk.text("con movimiento", (HELP_X, 388), 0.55, DIM, 1, SANS, "center")
        top_left = (HELP_X - HELP_SIZE // 2, 395)
        chalk.hand(fit_hand(context.references[help_letter], top_left, HELP_SIZE), WHITE, 2)

    def _draw_game_over(
        self,
        canvas: np.ndarray,
        chalk: Chalk,
        game: BubbleGame,
        context: Context,
        origin: tuple[int, int],
    ) -> list[Button]:
        center_x = origin[0] + ARENA_SIZE[0] // 2
        chalk.text("Fin de la partida", (center_x, 210), 2.6, WHITE, 4, SCRIPT, "center")
        points = "1 punto" if game.score == 1 else f"{game.score} puntos"
        chalk.text(points, (center_x, 300), 1.6, YELLOW, 3, SCRIPT, "center")
        if self._new_record:
            chalk.text("¡nuevo récord!", (center_x, 360), 1.2, PINK, 2, SCRIPT, "center")
            chalk.ring(
                (center_x, 350), (130, 36), PINK, ring_progress(self._over_since, context.now), 2
            )
        buttons = [
            chalk_word_button(
                chalk,
                "otra partida",
                "r",
                (center_x - 150, 480),
                WHITE,
                "otra",
                context.mouse,
                self._hover_since,
                context.now,
                emphasized=True,
            ),
            chalk_word_button(
                chalk,
                "menú",
                "Esc",
                (center_x + 170, 480),
                WHITE,
                "menu",
                context.mouse,
                self._hover_since,
                context.now,
            ),
        ]
        return buttons
