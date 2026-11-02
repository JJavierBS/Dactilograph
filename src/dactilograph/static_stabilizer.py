from dactilograph.classifier import Prediction

HOLD_S = 0.4
MIN_CONFIDENCE = 0.6


class StaticLetterStabilizer:
    def __init__(self, hold_s: float = HOLD_S, min_confidence: float = MIN_CONFIDENCE) -> None:
        self._hold_s = hold_s
        self._min_confidence = min_confidence
        self._candidate: str | None = None
        self._since_s = 0.0
        self._emitted = False

    def update(self, now_s: float, prediction: Prediction | None) -> str | None:
        if prediction is None or prediction.confidence < self._min_confidence:
            self._candidate = None
            return None
        if prediction.letter != self._candidate:
            self._candidate = prediction.letter
            self._since_s = now_s
            self._emitted = False
            return None
        if self._emitted or now_s - self._since_s < self._hold_s:
            return None
        self._emitted = True
        return self._candidate
