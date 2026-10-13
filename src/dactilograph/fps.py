import time


class FpsCounter:
    """Frames per second, smoothed with an exponential moving average to keep it readable."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._last_time: float | None = None
        self.fps = 0.0

    def tick(self, now: float | None = None) -> float:
        """Register a new frame and return the updated FPS."""
        now = time.perf_counter() if now is None else now
        if self._last_time is not None and now > self._last_time:
            instant_fps = 1.0 / (now - self._last_time)
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                self.fps = self._smoothing * self.fps + (1.0 - self._smoothing) * instant_fps
        self._last_time = now
        return self.fps
