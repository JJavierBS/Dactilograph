from collections import deque
from collections.abc import Callable, Sequence

from dactilograph.dataset import NO_DYNAMIC_LETTER
from dactilograph.hand_detector import Hand

# Misma duración que las secuencias de entrenamiento.
WINDOW_MS = 1500
# La ventana se clasifica solo si cubre casi toda la duración y tiene mano en al menos la mitad de
# los ~45 fotogramas, el mismo mínimo que se exige al grabar.
MIN_WINDOW_SPAN = 0.9
MIN_FRAMES = 22
# Al deslizarse, la ventana encaja con el gesto durante varios fotogramas seguidos; pedir varios
# evita que un fotograma suelto dispare una letra.
CONFIRMATIONS = 3
COOLDOWN_MS = 1000

SequencePredictor = Callable[[Sequence[Hand], Sequence[float]], str]


class DynamicRecognizer:
    def __init__(self, predict: SequencePredictor) -> None:
        self._predict = predict
        self._frames: deque[tuple[float, Hand]] = deque()
        self._candidate: str | None = None
        self._streak = 0
        self._cooldown_until_ms = float("-inf")

    def update(self, timestamp_ms: float, hand: Hand | None) -> str | None:
        if hand is not None:
            self._frames.append((timestamp_ms, hand))
        while self._frames and self._frames[0][0] < timestamp_ms - WINDOW_MS:
            self._frames.popleft()

        if timestamp_ms < self._cooldown_until_ms or not self._window_is_complete():
            self._reset_streak()
            return None

        timestamps, hands = zip(*self._frames, strict=True)
        letter = self._predict(hands, timestamps)
        if letter == NO_DYNAMIC_LETTER:
            self._reset_streak()
            return None
        self._streak = self._streak + 1 if letter == self._candidate else 1
        self._candidate = letter
        if self._streak < CONFIRMATIONS:
            return None

        # Se vacía la ventana para que el mismo gesto no se vuelva a reconocer al terminar la espera.
        self._frames.clear()
        self._reset_streak()
        self._cooldown_until_ms = timestamp_ms + COOLDOWN_MS
        return letter

    def _window_is_complete(self) -> bool:
        if len(self._frames) < MIN_FRAMES:
            return False
        span_ms = self._frames[-1][0] - self._frames[0][0]
        return span_ms >= WINDOW_MS * MIN_WINDOW_SPAN

    def _reset_streak(self) -> None:
        self._candidate = None
        self._streak = 0
