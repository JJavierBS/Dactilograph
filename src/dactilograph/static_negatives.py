import numpy as np
import pandas as pd

from dactilograph.dataset import (
    FEATURE_COLUMNS,
    NO_DYNAMIC_LETTER,
    NUM_LANDMARKS,
    RecordedSequence,
    recording_ids,
)
from dactilograph.hand_detector import Hand

# Unos 1,5 s a los ~30 FPS a los que se graban las secuencias reales.
FRAMES_PER_WINDOW = 45
FRAME_INTERVAL_MS = 33
# Sin temblor la muñeca quedaría inmóvil del todo, algo que una mano real nunca hace, y el modelo
# podría aprender esa quietud perfecta como la señal de que no hay letra.
JITTER_PALMS = 0.02
SYNTHETIC_PALM_SIZE = 0.2
IMAGE_CENTER = np.array([0.5, 0.5, 0.0])


def _hand_from_normalized(points: np.ndarray, offset: np.ndarray) -> Hand:
    # Imagen de 1x1 para que las coordenadas ya estén en píxeles cuadrados.
    return Hand(
        landmarks=(points + offset) * SYNTHETIC_PALM_SIZE + IMAGE_CENTER,
        world_landmarks=np.zeros((NUM_LANDMARKS, 3)),
        handedness="Right",
        handedness_score=1.0,
        image_width=1,
        image_height=1,
    )


# El CSV estático guarda los puntos ya normalizados (centrados en la muñeca, en tamaños de palma y
# con la mano izquierda reflejada), así que se reconstruyen como una mano derecha quieta.
def negatives_from_static(
    dataset: pd.DataFrame, frames_per_window: int = FRAMES_PER_WINDOW, seed: int = 0
) -> list[RecordedSequence]:
    rng = np.random.default_rng(seed)
    points = dataset[FEATURE_COLUMNS].to_numpy().reshape(-1, NUM_LANDMARKS, 3)
    recordings = recording_ids(dataset)
    sequences = []
    for recording in np.unique(recordings):
        rows = np.flatnonzero(recordings == recording)
        if len(rows) < frames_per_window:
            continue
        start = (len(rows) - frames_per_window) // 2
        window = rows[start : start + frames_per_window]
        offsets = np.zeros((len(window), 3))
        offsets[:, :2] = rng.normal(0, JITTER_PALMS, (len(window), 2))
        sequences.append(
            RecordedSequence(
                person=dataset["persona"].iloc[window[0]],
                letter=NO_DYNAMIC_LETTER,
                # Negativo para no chocar con los números de las secuencias grabadas.
                sequence_id=-int(recording),
                timestamps_ms=[index * FRAME_INTERVAL_MS for index in range(len(window))],
                hands=[
                    _hand_from_normalized(points[row], offset)
                    for row, offset in zip(window, offsets, strict=True)
                ],
            )
        )
    return sequences
