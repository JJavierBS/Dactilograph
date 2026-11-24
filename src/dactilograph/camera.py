import threading
import time
from typing import Self

import cv2
import numpy as np

STOP_TIMEOUT_S = 1.0


class Camera:
    """Webcam that returns frames mirrored, as a front-facing camera shows them."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._latest: tuple[np.ndarray, float] | None = None
        self._stopped = False
        self._new_frame = threading.Condition()
        self._thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._thread.start()

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if the camera stopped delivering frames."""
        timed = self.read_timed()
        return None if timed is None else timed[0]

    def read_timed(self) -> tuple[np.ndarray, float] | None:
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._latest is not None or self._stopped)
            if self._latest is None:
                return None
            frame, captured_s = self._latest
            self._latest = None
        # MediaPipe labels left and right hands assuming a mirrored (selfie) image.
        return cv2.flip(frame, 1), captured_s

    def _capture_frames(self) -> None:
        try:
            while not self._stopped:
                frame_read, frame = self._capture.read()
                if not frame_read:
                    return
                with self._new_frame:
                    self._latest = (frame, time.monotonic())
                    self._new_frame.notify_all()
        finally:
            with self._new_frame:
                self._stopped = True
                self._new_frame.notify_all()

    def close(self) -> None:
        with self._new_frame:
            self._stopped = True
            self._new_frame.notify_all()
        self._thread.join(STOP_TIMEOUT_S)
        self._capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
