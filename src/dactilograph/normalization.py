from collections.abc import Callable, Sequence

import numpy as np

from dactilograph.hand_detector import Hand

WRIST = 0
MIDDLE_FINGER_MCP = 9
MIN_PALM_SIZE = 1e-6

# Un paso recibe los puntos (21, 3) tal como los dejó el paso anterior, y la mano original por si
# necesita más datos (lado de la mano, tamaño de la imagen...). Devuelve los puntos transformados.
NormalizationStep = Callable[[np.ndarray, Hand], np.ndarray]


def to_square_pixels(points: np.ndarray, hand: Hand) -> np.ndarray:
    # x va relativo al ancho e y al alto: sin esto la mano se deforma según la proporción de la
    # imagen. z sigue la escala de x.
    return points * [hand.image_width, hand.image_height, hand.image_width]


def center_on_wrist(points: np.ndarray, hand: Hand) -> np.ndarray:
    return points - points[WRIST]


def scale_by_palm_size(points: np.ndarray, hand: Hand) -> np.ndarray:
    # La palma mide lo mismo con el puño cerrado que con la mano abierta, así que no depende de la
    # letra que se esté haciendo.
    palm_size = np.linalg.norm(points[MIDDLE_FINGER_MCP] - points[WRIST])
    if palm_size < MIN_PALM_SIZE:
        raise ValueError("The hand has zero size and cannot be normalized")
    return points / palm_size


def mirror_left_hand(points: np.ndarray, hand: Hand) -> np.ndarray:
    if hand.handedness == "Left":
        return points * [-1, 1, 1]
    return points


DEFAULT_STEPS: tuple[NormalizationStep, ...] = (
    to_square_pixels,
    center_on_wrist,
    scale_by_palm_size,
    mirror_left_hand,
)


def normalize_hand(hand: Hand, steps: Sequence[NormalizationStep] = DEFAULT_STEPS) -> np.ndarray:
    points = hand.landmarks
    for step in steps:
        points = step(points, hand)
    return points.flatten()
