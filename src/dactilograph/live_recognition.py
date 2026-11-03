from collections.abc import Callable
from dataclasses import dataclass

from dactilograph.classifier import Prediction
from dactilograph.dynamic_recognizer import DynamicRecognizer, SequencePredictor
from dactilograph.hand_detector import Hand
from dactilograph.static_stabilizer import StaticLetterStabilizer

StaticPredictor = Callable[[Hand], Prediction]


@dataclass(frozen=True)
class Reading:
    hand: Hand | None
    prediction: Prediction | None
    confirmed: tuple[str, ...]


class LiveRecognizer:
    def __init__(
        self, predict_static: StaticPredictor, predict_sequence: SequencePredictor
    ) -> None:
        self._predict_static = predict_static
        self._predict_sequence = predict_sequence
        self.reset()

    def reset(self) -> None:
        self._stabilizer = StaticLetterStabilizer()
        self._dynamic = DynamicRecognizer(self._predict_sequence)

    def read(self, now_s: float, hand: Hand | None) -> Reading:
        prediction = self._predict_static(hand) if hand is not None else None
        made = (
            self._stabilizer.update(now_s, prediction),
            self._dynamic.update(now_s * 1000, hand),
        )
        confirmed = tuple(letter for letter in made if letter is not None)
        return Reading(hand, prediction, confirmed)
