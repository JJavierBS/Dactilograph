from collections.abc import Iterable, Sequence
from dataclasses import replace

import cv2
import numpy as np
from sklearn.base import ClassifierMixin, clone

from dactilograph.dataset import NUM_LANDMARKS
from dactilograph.hand_detector import Hand
from dactilograph.normalization import WRIST, palm_size, to_square_pixels

MAX_ROTATION_DEGREES = 15.0
MAX_SCALE_CHANGE = 0.1
MAX_SHEAR = 0.1


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def scale_matrix(ex: float, ey: float) -> np.ndarray:
    return np.array([[ex, 0.0, 0.0], [0.0, ey, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrix(degrees: float) -> np.ndarray:
    cos, sin = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def shear_matrix(ix: float, iy: float) -> np.ndarray:
    return np.array([[1.0, ix, 0.0], [iy, 1.0, 0.0], [0.0, 0.0, 1.0]])


def around_center(matrix: np.ndarray, center: Sequence[float]) -> np.ndarray:
    cx, cy = center
    return translation_matrix(cx, cy) @ matrix @ translation_matrix(-cx, -cy)


def random_affine(rng: np.random.Generator, center: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    rotation = rotation_matrix(rng.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES))
    scale = scale_matrix(*(1 + rng.uniform(-MAX_SCALE_CHANGE, MAX_SCALE_CHANGE, 2)))
    shear = shear_matrix(*rng.uniform(-MAX_SHEAR, MAX_SHEAR, 2))
    return around_center(rotation @ scale @ shear, center)


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = points.astype(float)
    xy = transformed[:, :2].reshape(-1, 1, 2)
    transformed[:, :2] = cv2.transform(xy, matrix[:2]).reshape(-1, 2)
    return transformed


def augment_normalized_hand(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    points = apply_affine(features.reshape(NUM_LANDMARKS, 3), random_affine(rng))
    return (points / palm_size(points)).flatten()


def augment_sequence(hands: Sequence[Hand], rng: np.random.Generator) -> list[Hand]:
    first_hand = hands[0]
    center = to_square_pixels(first_hand.landmarks, first_hand)[WRIST, :2]
    matrix = random_affine(rng, center)
    augmented = []
    for hand in hands:
        size = [hand.image_width, hand.image_height, hand.image_width]
        pixels = apply_affine(to_square_pixels(hand.landmarks, hand), matrix)
        augmented.append(replace(hand, landmarks=pixels / size))
    return augmented


def cross_val_predict_both(
    model: ClassifierMixin,
    features: np.ndarray,
    labels: np.ndarray,
    transformed: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    copies: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    on_original = np.empty(len(labels), dtype=object)
    on_transformed = np.empty(len(labels), dtype=object)
    for train, test in splits:
        training = (features[train], labels[train])
        if copies is not None:
            training = with_copies(*training, copies[train])
        fitted = clone(model).fit(*training)
        on_original[test] = fitted.predict(features[test])
        on_transformed[test] = fitted.predict(transformed[test])
    return on_original.astype(str), on_transformed.astype(str)


def with_copies(
    features: np.ndarray, labels: np.ndarray, copies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    flat_copies = copies.reshape(-1, features.shape[1])
    all_labels = np.concatenate([labels, np.repeat(labels, copies.shape[1])])
    return np.concatenate([features, flat_copies]), all_labels
