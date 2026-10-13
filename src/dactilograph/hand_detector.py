import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from dactilograph.paths import HAND_LANDMARKER_PATH

# La imagen llega en espejo, así que MediaPipe etiqueta cada mano como la contraria.
MIRRORED_HANDEDNESS = {"Left": "Right", "Right": "Left"}


@dataclass(frozen=True)
class Hand:
    # landmarks: x e y relativos al tamaño de la imagen (0 a 1); z, profundidad respecto a la
    # muñeca en la misma escala que x. world_landmarks: en metros, centrados en la mano.
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    handedness: str
    handedness_score: float


class HandDetector:
    def __init__(self, model_path: Path = HAND_LANDMARKER_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}. Run: uv run scripts/download_model.py"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray) -> Hand | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())
        if not result.hand_landmarks:
            return None

        handedness = result.handedness[0][0]
        return Hand(
            landmarks=np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]]),
            world_landmarks=np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]]),
            handedness=MIRRORED_HANDEDNESS[handedness.category_name],
            handedness_score=handedness.score,
        )

    def _next_timestamp_ms(self) -> int:
        # El modo vídeo exige marcas de tiempo estrictamente crecientes, y dos fotogramas pueden
        # llegar en el mismo milisegundo.
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
