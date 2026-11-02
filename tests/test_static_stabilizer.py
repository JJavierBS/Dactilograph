from dactilograph.classifier import Prediction
from dactilograph.static_stabilizer import HOLD_S, MIN_CONFIDENCE, StaticLetterStabilizer

SURE = MIN_CONFIDENCE + 0.1
UNSURE = MIN_CONFIDENCE - 0.1


def test_letter_is_confirmed_after_being_held():
    stabilizer = StaticLetterStabilizer()
    assert stabilizer.update(0.0, Prediction("A", SURE)) is None
    assert stabilizer.update(HOLD_S / 2, Prediction("A", SURE)) is None
    assert stabilizer.update(HOLD_S, Prediction("A", SURE)) == "A"


def test_held_letter_is_confirmed_only_once():
    stabilizer = StaticLetterStabilizer()
    stabilizer.update(0.0, Prediction("A", SURE))
    stabilizer.update(HOLD_S, Prediction("A", SURE))
    assert stabilizer.update(HOLD_S * 5, Prediction("A", SURE)) is None


def test_changing_letter_restarts_the_hold():
    stabilizer = StaticLetterStabilizer()
    stabilizer.update(0.0, Prediction("A", SURE))
    assert stabilizer.update(HOLD_S, Prediction("B", SURE)) is None
    assert stabilizer.update(HOLD_S * 2, Prediction("B", SURE)) == "B"


def test_unsure_or_missing_prediction_restarts_the_hold():
    stabilizer = StaticLetterStabilizer()
    stabilizer.update(0.0, Prediction("A", SURE))
    stabilizer.update(HOLD_S / 2, Prediction("A", UNSURE))
    assert stabilizer.update(HOLD_S, Prediction("A", SURE)) is None
    stabilizer.update(HOLD_S * 1.5, None)
    assert stabilizer.update(HOLD_S * 2, Prediction("A", SURE)) is None


def test_same_letter_can_be_confirmed_again_after_letting_go():
    stabilizer = StaticLetterStabilizer()
    stabilizer.update(0.0, Prediction("A", SURE))
    stabilizer.update(HOLD_S, Prediction("A", SURE))
    stabilizer.update(5.0, None)
    stabilizer.update(10.0, Prediction("A", SURE))
    assert stabilizer.update(10.0 + HOLD_S, Prediction("A", SURE)) == "A"
