from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from dactilograph.hand_detector import Hand
from dactilograph.normalization import normalize_hand
from dactilograph.paths import CLASSIFIER_PATH


@dataclass(frozen=True)
class Prediction:
    letter: str
    confidence: float


class LetterClassifier:
    def __init__(self, model_path: Path = CLASSIFIER_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {model_path}. Run: uv run scripts/train.py"
            )
        self._model = joblib.load(model_path)

    def predict(self, hand: Hand) -> Prediction:
        probabilities = self._model.predict_proba([normalize_hand(hand)])[0]
        best = int(np.argmax(probabilities))
        return Prediction(
            letter=str(self._model.classes_[best]), confidence=float(probabilities[best])
        )
