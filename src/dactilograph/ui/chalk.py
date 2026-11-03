from functools import lru_cache

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

WIDTH = 1280
HEIGHT = 720
FRAME = 22
TRAY_HEIGHT = 70
BOARD_BOTTOM = HEIGHT - TRAY_HEIGHT

BOARD = (45, 53, 33)
WHITE = (226, 235, 236)
YELLOW = (91, 211, 242)
PINK = (164, 142, 238)
BLUE = (230, 195, 140)
DIM = (165, 170, 160)
FAINT = (110, 120, 105)
WOOD = (50, 90, 138)
WOOD_DARK = (31, 59, 94)
TAPE = (205, 215, 220)

SCRIPT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
SANS = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

BONES = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]
SMEARS = 9
SMEAR_BLUR = 35
SMEAR_STRENGTH = 14
BOARD_NOISE = 3
GRAIN_MIN = 0.45
SHADOW_OFFSET = (6, 8)
SHADOW_MARGIN = 24
SHADOW_BLUR = 8
SHADOW_STRENGTH = 0.55

Point = tuple[int, int]
Color = tuple[int, int, int]


def board_background(rng: np.random.Generator) -> np.ndarray:
    board = np.zeros((HEIGHT, WIDTH, 3), np.float32) + np.array(BOARD, np.float32)
    for _ in range(SMEARS):
        smear = np.zeros((HEIGHT, WIDTH), np.float32)
        center = (int(rng.integers(100, WIDTH - 100)), int(rng.integers(80, BOARD_BOTTOM - 80)))
        axes = (int(rng.integers(120, 320)), int(rng.integers(30, 70)))
        cv2.ellipse(smear, center, axes, float(rng.uniform(-15, 15)), 0, 360, 1.0, -1)
        board += cv2.GaussianBlur(smear, (0, 0), SMEAR_BLUR)[..., None] * SMEAR_STRENGTH
    board += rng.normal(0, BOARD_NOISE, board.shape)
    return np.clip(board, 0, 255).astype(np.uint8)


def chalk_grain(rng: np.random.Generator) -> np.ndarray:
    grain = rng.uniform(GRAIN_MIN, 1.0, (HEIGHT, WIDTH)).astype(np.float32)
    return cv2.GaussianBlur(grain, (0, 0), 0.6)[..., None]


class Chalk:
    def __init__(self, grain: np.ndarray) -> None:
        self._grain = grain
        self.layer = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    def clear(self) -> None:
        self.layer[:] = 0

    def apply(self, canvas: np.ndarray) -> None:
        blue, green, red = cv2.split(self.layer)
        strength = cv2.max(cv2.max(blue, green), red)
        rows, cols = np.nonzero(strength)
        if rows.size == 0:
            return
        peak = strength[rows, cols].astype(np.float32)[:, None]
        color = self.layer[rows, cols].astype(np.float32) * (255 / peak)
        alpha = peak / 255 * self._grain[rows, cols]
        under = canvas[rows, cols].astype(np.float32)
        canvas[rows, cols] = (under + (color - under) * alpha).astype(np.uint8)

    def text(
        self,
        text: str,
        origin: Point,
        scale: float,
        color: Color,
        thickness: int = 2,
        font: int = SANS,
        align: str = "left",
    ) -> tuple[int, int]:
        return put_text(self.layer, text, origin, scale, color, thickness, font, align)

    def line(self, start: Point, end: Point, color: Color, thickness: int = 2) -> None:
        cv2.line(self.layer, start, end, color, thickness, AA)

    def circle(self, center: Point, radius: int, color: Color, thickness: int = 2) -> None:
        cv2.circle(self.layer, center, radius, color, thickness, AA)

    def ring(
        self,
        center: Point,
        axes: tuple[int, int],
        color: Color,
        progress: float = 1.0,
        thickness: int = 3,
        tilt: float = -4,
    ) -> None:
        sweep = 345 * min(max(progress, 0.0), 1.0)
        if sweep > 0:
            cv2.ellipse(self.layer, center, axes, tilt, -100, -100 + sweep, color, thickness, AA)

    def cross(self, center: Point, size: int, color: Color, thickness: int = 3) -> None:
        x, y = center
        self.line((x - size, y - size), (x + size, y + size), color, thickness)
        self.line((x + size, y - size), (x - size, y + size), color, thickness)

    def hand(self, points: np.ndarray, color: Color, thickness: int = 3) -> None:
        draw_skeleton(self.layer, points, color, thickness)


def text_size(text: str, scale: float, thickness: int = 2, font: int = SANS) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    return width, height


def put_text(
    image: np.ndarray,
    text: str,
    origin: Point,
    scale: float,
    color: Color,
    thickness: int = 2,
    font: int = SANS,
    align: str = "left",
) -> tuple[int, int]:
    width, height = text_size(text, scale, thickness, font)
    x, y = origin
    if align == "center":
        x -= width // 2
    elif align == "right":
        x -= width
    cv2.putText(image, text, (int(x), int(y)), font, scale, color, thickness, AA)
    return width, height


def draw_skeleton(image: np.ndarray, points: np.ndarray, color: Color, thickness: int) -> None:
    joint_radius = thickness + 1
    for start, end in BONES:
        cv2.line(image, tuple(points[start]), tuple(points[end]), color, thickness, AA)
    for point in points:
        cv2.circle(image, tuple(point), joint_radius, color, -1, AA)


def fit_hand(points: np.ndarray, top_left: Point, size: int, margin: float = 0.08) -> np.ndarray:
    xy = points[:, :2]
    low, high = xy.min(axis=0), xy.max(axis=0)
    usable = size * (1 - 2 * margin)
    scale = usable / max(float((high - low).max()), 1e-6)
    offset = np.array(top_left) + size * margin + (usable - (high - low) * scale) / 2
    return ((xy - low) * scale + offset).astype(int)


def hand_in_box(landmarks: np.ndarray, top_left: Point, size: tuple[int, int]) -> np.ndarray:
    scale = np.array(size, np.float64)
    return (landmarks[:, :2] * scale + np.array(top_left)).astype(int)


def _wood_grain(canvas: np.ndarray, rng: np.random.Generator, horizontal: bool) -> None:
    height, width = canvas.shape[:2]
    length, across = (width, height) if horizontal else (height, width)
    position = float(rng.uniform(0, 3))
    along = np.arange(0, length + 8, 8)
    while position < across:
        shade = tuple(int(c * rng.uniform(0.82, 1.08)) for c in WOOD)
        phase, amplitude = rng.uniform(0, 2 * np.pi), rng.uniform(0.4, 1.6)
        offsets = position + amplitude * np.sin(along / rng.uniform(60, 160) + phase)
        points = np.column_stack((along, offsets) if horizontal else (offsets, along))
        cv2.polylines(canvas, [points.astype(np.int32)], False, shade, 1, AA)
        position += float(rng.uniform(2.5, 6))


def draw_wood(canvas: np.ndarray, rng: np.random.Generator) -> None:
    inner = FRAME // 2
    canvas[:inner] = WOOD
    canvas[:, :inner] = WOOD
    canvas[:, WIDTH - inner :] = WOOD
    canvas[BOARD_BOTTOM:] = WOOD
    _wood_grain(canvas[:inner], rng, horizontal=True)
    _wood_grain(canvas[:BOARD_BOTTOM, :inner], rng, horizontal=False)
    _wood_grain(canvas[:BOARD_BOTTOM, WIDTH - inner :], rng, horizontal=False)
    _wood_grain(canvas[BOARD_BOTTOM + 4 :], rng, horizontal=True)
    cv2.rectangle(canvas, (inner, inner), (WIDTH - inner, BOARD_BOTTOM), WOOD_DARK, 2)
    cv2.line(canvas, (0, BOARD_BOTTOM), (WIDTH, BOARD_BOTTOM), WOOD_DARK, 3)


@lru_cache(maxsize=4)
def soft_mask(width: int, height: int, margin: int, sigma: float) -> np.ndarray:
    mask = np.zeros((height + 2 * margin, width + 2 * margin), np.float32)
    mask[margin : margin + height, margin : margin + width] = 1
    return cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]


def pin_photo(canvas: np.ndarray, photo: np.ndarray, top_left: Point) -> None:
    x, y = top_left
    height, width = photo.shape[:2]
    margin = SHADOW_MARGIN
    shadow = soft_mask(width, height, margin, SHADOW_BLUR)
    top, left = y + SHADOW_OFFSET[1] - margin, x + SHADOW_OFFSET[0] - margin
    region = canvas[top : top + shadow.shape[0], left : left + shadow.shape[1]]
    region[:] = (region * (1 - SHADOW_STRENGTH * shadow)).astype(np.uint8)
    canvas[y : y + height, x : x + width] = photo
    for center, angle in (((x + 18, y + 2), -22), ((x + width - 18, y + 2), 20)):
        corners = cv2.boxPoints((center, (84, 26), angle)).astype(np.int32)
        left, top = corners.min(axis=0)
        right, bottom = corners.max(axis=0) + 1
        region = canvas[top:bottom, left:right]
        tape = region.copy()
        cv2.fillPoly(tape, [corners - (left, top)], TAPE, AA)
        cv2.addWeighted(tape, 0.7, region, 0.3, 0, region)
