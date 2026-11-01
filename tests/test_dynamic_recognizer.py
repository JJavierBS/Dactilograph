import numpy as np

from dactilograph.dataset import NO_DYNAMIC_LETTER
from dactilograph.dynamic_recognizer import (
    CONFIRMATIONS,
    COOLDOWN_MS,
    MIN_FRAMES,
    WINDOW_MS,
    DynamicRecognizer,
)
from dactilograph.hand_detector import Hand

FRAME_MS = 33
HAND = Hand(np.zeros((21, 3)), np.zeros((21, 3)), "Right", 1.0, 640, 480)


class FakePredictor:
    def __init__(self, letter):
        self.letter = letter
        self.calls = 0

    def __call__(self, hands, timestamps_ms):
        self.calls += 1
        return self.letter


def feed(recognizer, start_ms, num_frames, hand=HAND):
    results = []
    for frame in range(num_frames):
        results.append(recognizer.update(start_ms + frame * FRAME_MS, hand))
    return results


def frames_to_fill_window():
    return WINDOW_MS // FRAME_MS + 1


def test_nothing_is_predicted_until_the_window_is_full():
    predictor = FakePredictor("Z")
    recognizer = DynamicRecognizer(predictor)
    feed(recognizer, 0, frames_to_fill_window() - 5)
    assert predictor.calls == 0


def test_letter_is_returned_after_several_confirmations():
    recognizer = DynamicRecognizer(FakePredictor("Z"))
    results = feed(recognizer, 0, frames_to_fill_window() + CONFIRMATIONS)
    recognized = [result for result in results if result is not None]
    assert recognized == ["Z"]
    first = results.index("Z")
    assert results[first - CONFIRMATIONS + 1 : first] == [None] * (CONFIRMATIONS - 1)


def test_no_dynamic_letter_is_never_returned():
    recognizer = DynamicRecognizer(FakePredictor(NO_DYNAMIC_LETTER))
    assert set(feed(recognizer, 0, 200)) == {None}


def test_changing_prediction_restarts_the_confirmation():
    predictor = FakePredictor("Z")
    recognizer = DynamicRecognizer(predictor)
    results = []
    while predictor.calls < CONFIRMATIONS - 1:
        results += feed(recognizer, len(results) * FRAME_MS, 1)
    predictor.letter = "J"
    results += feed(recognizer, len(results) * FRAME_MS, CONFIRMATIONS - 1)
    assert set(results) == {None}
    assert feed(recognizer, len(results) * FRAME_MS, 1) == ["J"]


def test_same_gesture_is_not_recognized_twice():
    predictor = FakePredictor("Z")
    recognizer = DynamicRecognizer(predictor)
    results = feed(recognizer, 0, frames_to_fill_window() + CONFIRMATIONS)
    predictor.calls = 0
    cooldown_frames = COOLDOWN_MS // FRAME_MS
    results = feed(recognizer, len(results) * FRAME_MS, cooldown_frames)
    assert set(results) == {None}
    assert predictor.calls == 0


def test_window_without_enough_hands_is_not_classified():
    predictor = FakePredictor("Z")
    recognizer = DynamicRecognizer(predictor)
    for frame in range(200):
        hand = HAND if frame % 3 == 0 else None
        recognizer.update(frame * FRAME_MS, hand)
    assert frames_to_fill_window() // 3 < MIN_FRAMES
    assert predictor.calls == 0


def test_old_frames_leave_the_window():
    predictor = FakePredictor(NO_DYNAMIC_LETTER)
    seen_sizes = []

    def predict(hands, timestamps_ms):
        seen_sizes.append(len(hands))
        assert timestamps_ms[-1] - timestamps_ms[0] <= WINDOW_MS
        return predictor(hands, timestamps_ms)

    feed(DynamicRecognizer(predict), 0, 200)
    assert max(seen_sizes) <= frames_to_fill_window()
