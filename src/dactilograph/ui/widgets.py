from dataclasses import dataclass

import cv2
import numpy as np

from dactilograph.ui.chalk import (
    AA,
    BOARD_BOTTOM,
    HEIGHT,
    SANS,
    TAPE,
    WHITE,
    YELLOW,
    put_text,
    text_size,
)

TRAY_ITEMS = (
    ("Menú", "Esc", "menu"),
    ("Aprender", "1", "aprender"),
    ("Reconocer", "2", "reconocer"),
    ("Jugar", "3", "jugar"),
    ("Salir", "q", "salir"),
)
TRAY_LEFT = 60
TRAY_GAP = 52
STICK_WIDTH = 66
STICK_HEIGHT = 14
STICK_LIFT = 5
LABEL_COLOR = (235, 238, 240)


@dataclass(frozen=True)
class Button:
    x: int
    y: int
    width: int
    height: int
    action: str

    def contains(self, point: tuple[int, int] | None) -> bool:
        if point is None:
            return False
        px, py = point
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


QUIT_ONLY = (TRAY_ITEMS[-1],)


def draw_tray(
    canvas: np.ndarray,
    active: str,
    mouse: tuple[int, int] | None,
    items: tuple[tuple[str, str, str], ...] = TRAY_ITEMS,
) -> list[Button]:
    buttons = []
    x = TRAY_LEFT
    for label, key, action in items:
        label_width, _ = text_size(label, 0.75, 2)
        button = Button(x - 10, BOARD_BOTTOM + 4, STICK_WIDTH + label_width + 36, 62, action)
        lifted = button.contains(mouse)
        lift = STICK_LIFT if lifted else 0
        stick_top = HEIGHT - 50 - lift
        stick_color = YELLOW if action == active else (WHITE if lifted else TAPE)
        cv2.rectangle(
            canvas, (x, stick_top), (x + STICK_WIDTH, stick_top + STICK_HEIGHT), stick_color, -1, AA
        )
        cv2.circle(
            canvas,
            (x + STICK_WIDTH, stick_top + STICK_HEIGHT // 2),
            STICK_HEIGHT // 2,
            stick_color,
            -1,
            AA,
        )
        put_text(canvas, label, (x + STICK_WIDTH + 18, HEIGHT - 31), 0.75, LABEL_COLOR, 2)
        put_text(canvas, key, (x + STICK_WIDTH + 18, HEIGHT - 9), 0.6, LABEL_COLOR, 1, SANS)
        buttons.append(button)
        x += STICK_WIDTH + 18 + label_width + TRAY_GAP
    return buttons
