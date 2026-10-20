import joblib
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from dactilograph.classifier import LetterClassifier
from dactilograph.hand_detector import Hand
from dactilograph.normalization import normalize_hand


def make_hand(spread):
    landmarks = np.zeros((21, 3))
    landmarks[:, 0] = np.linspace(0.4, 0.4 + spread, 21)
    landmarks[:, 1] = np.linspace(0.6, 0.4, 21)
    return Hand(landmarks, np.zeros((21, 3)), "Right", 1.0, 640, 480)


def test_predicts_the_letter_of_the_closest_training_hand(tmp_path):
    open_hand, closed_hand = make_hand(0.2), make_hand(0.0)
    model = KNeighborsClassifier(n_neighbors=1).fit(
        [normalize_hand(open_hand), normalize_hand(closed_hand)], ["B", "A"]
    )
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    prediction = LetterClassifier(path).predict(open_hand)
    assert prediction.letter == "B"
    assert prediction.confidence == pytest.approx(1.0)


def test_missing_model_explains_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        LetterClassifier(tmp_path / "missing.joblib")
