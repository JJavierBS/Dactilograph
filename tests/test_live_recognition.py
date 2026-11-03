import numpy as np

from dactilograph.classifier import Prediction
from dactilograph.hand_detector import Hand
from dactilograph.live_recognition import LiveRecognizer
from dactilograph.static_stabilizer import HOLD_S

HAND = Hand(np.zeros((21, 3)), np.zeros((21, 3)), "Right", 1.0, 640, 480)


def recognizer(letter="B", confidence=0.9):
    return LiveRecognizer(lambda hand: Prediction(letter, confidence), lambda hands, times: "-")


def test_without_hand_there_is_no_prediction():
    reading = recognizer().read(0.0, None)
    assert reading.hand is None
    assert reading.prediction is None
    assert reading.confirmed == ()


def test_held_static_letter_is_confirmed_once():
    live = recognizer()
    assert live.read(5.0, HAND).confirmed == ()
    assert live.read(5.0 + HOLD_S, HAND).confirmed == ("B",)
    assert live.read(10.0, HAND).confirmed == ()


def test_reset_forgets_the_letter_being_held():
    live = recognizer()
    live.read(5.0, HAND)
    live.reset()
    assert live.read(5.0 + HOLD_S, HAND).confirmed == ()
