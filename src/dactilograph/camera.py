from typing import Self

import cv2
import numpy as np


class Camera:
    """Webcam that returns frames mirrored, as a front-facing camera shows them."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if the camera stopped delivering frames."""
        frame_read, frame = self._capture.read()
        if not frame_read:
            return None
        # MediaPipe labels left and right hands assuming a mirrored (selfie) image.
        return cv2.flip(frame, 1)

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
