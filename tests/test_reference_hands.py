import numpy as np
import pandas as pd
import pytest

from dactilograph.dataset import FEATURE_COLUMNS, NO_DYNAMIC_LETTER, RecordedSequence
from dactilograph.hand_detector import Hand
from dactilograph.normalization import MIDDLE_FINGER_MCP, WRIST, normalize_hand
from dactilograph.reference_hands import reference_hands


def static_dataset():
    rows = [["jose", "A", *np.full(63, value)] for value in (1.0, 3.0)]
    rows.append(["jose", "B", *np.full(63, 5.0)])
    return pd.DataFrame(rows, columns=["persona", "letra", *FEATURE_COLUMNS])


def make_hand(spread):
    landmarks = np.zeros((21, 3))
    landmarks[:, 0] = np.linspace(0.4, 0.4 + spread, 21)
    landmarks[:, 1] = np.linspace(0.6, 0.4, 21)
    landmarks[WRIST] = [0.5, 0.65, 0.0]
    landmarks[MIDDLE_FINGER_MCP] = [0.5, 0.5, 0.0]
    return Hand(landmarks, np.zeros((21, 3)), "Right", 1.0, 640, 480)


def sequence(letter, hands):
    return RecordedSequence("jose", letter, 1, list(range(len(hands))), hands)


def test_static_letters_use_their_mean_hand():
    references = reference_hands(static_dataset(), [])
    assert references["A"].shape == (21, 3)
    assert references["A"] == pytest.approx(np.full((21, 3), 2.0))
    assert references["B"] == pytest.approx(np.full((21, 3), 5.0))


def test_dynamic_letters_use_the_mean_of_their_first_frames():
    first, second = make_hand(0.1), make_hand(0.3)
    sequences = [sequence("Z", [first, second]), sequence("Z", [second, first])]
    references = reference_hands(static_dataset(), sequences)
    expected = np.mean([normalize_hand(first), normalize_hand(second)], axis=0)
    assert references["Z"] == pytest.approx(expected.reshape(21, 3))


def test_negative_sequences_have_no_reference():
    references = reference_hands(static_dataset(), [sequence(NO_DYNAMIC_LETTER, [make_hand(0.1)])])
    assert NO_DYNAMIC_LETTER not in references
