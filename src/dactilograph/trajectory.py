from collections import Counter
from collections.abc import Sequence

import numpy as np

from dactilograph.hand_detector import Hand
from dactilograph.normalization import WRIST, palm_size, to_square_pixels

INDEX_FINGER_TIP = 8
TRACKED_POINTS = (WRIST, INDEX_FINGER_TIP)
DEFAULT_NUM_FRAMES = 20


# Devuelve (fotogramas, puntos seguidos, 2). A diferencia de normalize_hand, no centra cada
# fotograma en su muñeca porque eso borraría el movimiento: todo se mide desde la muñeca del primer
# fotograma y en tamaños de palma.
def hand_trajectory(hands: Sequence[Hand]) -> np.ndarray:
    if not hands:
        raise ValueError("The sequence has no hands")
    pixels = np.array([to_square_pixels(hand.landmarks, hand)[:, :2] for hand in hands])
    # La mediana evita que un fotograma mal detectado cambie la escala de toda la secuencia.
    scale = float(np.median([palm_size(points) for points in pixels]))
    trajectory = (pixels[:, TRACKED_POINTS] - pixels[0, WRIST]) / scale
    # MediaPipe a veces cambia de mano en un fotograma suelto; se usa la que más se repite.
    handedness = Counter(hand.handedness for hand in hands).most_common(1)[0][0]
    if handedness == "Left":
        trajectory = trajectory * [-1, 1]
    return trajectory


# Interpolar a instantes equiespaciados hace que no importen la velocidad del gesto, los FPS de la
# cámara ni los fotogramas en los que no se detectó la mano.
def resample(
    timestamps_ms: Sequence[float], values: np.ndarray, num_frames: int = DEFAULT_NUM_FRAMES
) -> np.ndarray:
    times = np.asarray(timestamps_ms, dtype=float)
    if len(times) != len(values):
        raise ValueError(f"Got {len(times)} timestamps for {len(values)} frames")
    if len(times) < 2:
        raise ValueError("At least 2 frames are needed to resample")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be strictly increasing")

    flat = np.asarray(values, dtype=float).reshape(len(times), -1)
    new_times = np.linspace(times[0], times[-1], num_frames)
    columns = [np.interp(new_times, times, column) for column in flat.T]
    return np.column_stack(columns).reshape(num_frames, *np.shape(values)[1:])


def trajectory_features(
    hands: Sequence[Hand], timestamps_ms: Sequence[float], num_frames: int = DEFAULT_NUM_FRAMES
) -> np.ndarray:
    return resample(timestamps_ms, hand_trajectory(hands), num_frames).flatten()
