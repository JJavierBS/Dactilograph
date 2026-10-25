import numpy as np
import pytest

from dactilograph.hand_detector import Hand
from dactilograph.normalization import MIDDLE_FINGER_MCP, WRIST
from dactilograph.trajectory import (
    INDEX_FINGER_TIP,
    hand_trajectory,
    resample,
    trajectory_features,
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
PALM_SIZE = 0.1


def make_hand(offset=(0.0, 0.0), scale=1.0, handedness="Right"):
    landmarks = np.full((21, 3), 0.0)
    landmarks[WRIST] = [0.0, 0.0, 0.0]
    landmarks[MIDDLE_FINGER_MCP] = [0.0, -PALM_SIZE, 0.0]
    landmarks[INDEX_FINGER_TIP] = [0.0, -2 * PALM_SIZE, 0.0]
    landmarks = landmarks * scale + [0.5 + offset[0], 0.6 + offset[1], 0.0]
    # Coordenadas relativas al ancho y al alto, como las da MediaPipe.
    landmarks = landmarks * [IMAGE_HEIGHT / IMAGE_WIDTH, 1, 1]
    return Hand(
        landmarks=landmarks,
        world_landmarks=np.zeros((21, 3)),
        handedness=handedness,
        handedness_score=1.0,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )


def moving_right(handedness="Right", scale=1.0, steps=5):
    return [
        make_hand(offset=(0.05 * step, 0.0), scale=scale, handedness=handedness)
        for step in range(steps)
    ]


def test_trajectory_starts_at_the_first_wrist():
    trajectory = hand_trajectory(moving_right())
    assert trajectory.shape == (5, 2, 2)
    assert trajectory[0, 0] == pytest.approx([0, 0])


def test_movement_is_measured_in_palm_sizes():
    trajectory = hand_trajectory(moving_right())
    step_in_palms = 0.05 / PALM_SIZE
    assert trajectory[1, 0] == pytest.approx([step_in_palms, 0])


def test_wrist_movement_is_kept():
    trajectory = hand_trajectory(moving_right())
    assert trajectory[-1, 0, 0] > trajectory[0, 0, 0]


def test_distance_to_the_camera_does_not_matter():
    near = hand_trajectory(moving_right(scale=2.0))
    far = hand_trajectory(moving_right(scale=1.0))
    assert near[:, 1] - near[:, 0] == pytest.approx(far[:, 1] - far[:, 0])


def test_left_hand_movement_is_mirrored():
    right = hand_trajectory(moving_right("Right"))
    left = hand_trajectory(moving_right("Left"))
    assert left == pytest.approx(right * [-1, 1])


def test_a_single_wrong_handedness_does_not_flip_the_sequence():
    hands = moving_right("Right")
    hands[2] = make_hand(offset=(0.1, 0.0), handedness="Left")
    assert hand_trajectory(hands) == pytest.approx(hand_trajectory(moving_right("Right")))


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError):
        hand_trajectory([])


def test_resample_interpolates_by_time():
    values = np.array([[0.0], [10.0], [30.0]])
    resampled = resample([0, 100, 200], values, num_frames=5)
    assert resampled.ravel() == pytest.approx([0, 5, 10, 20, 30])


def test_resample_keeps_the_shape_of_each_frame():
    values = np.zeros((7, 2, 2))
    assert resample(np.arange(7) * 33, values, num_frames=20).shape == (20, 2, 2)


def test_speed_of_the_gesture_does_not_matter():
    hands = moving_right()
    slow = trajectory_features(hands, [0, 400, 800, 1200, 1600])
    fast = trajectory_features(hands, [0, 100, 200, 300, 400])
    assert slow == pytest.approx(fast)


def test_missing_frames_are_filled_by_interpolation():
    hands = moving_right()
    timestamps = [0, 100, 200, 300, 400]
    complete = trajectory_features(hands, timestamps)
    with_gap = trajectory_features(hands[:2] + hands[3:], timestamps[:2] + timestamps[3:])
    assert with_gap == pytest.approx(complete)


def test_features_have_a_fixed_size():
    assert trajectory_features(moving_right(steps=4), [0, 30, 60, 90]).shape == (80,)
    assert trajectory_features(moving_right(steps=9), np.arange(9) * 30).shape == (80,)


@pytest.mark.parametrize("timestamps", [[0, 100, 100], [0, 200, 100]])
def test_timestamps_must_increase(timestamps):
    with pytest.raises(ValueError):
        resample(timestamps, np.zeros((3, 2)))


def test_resample_needs_two_frames():
    with pytest.raises(ValueError):
        resample([0], np.zeros((1, 2)))


def test_resample_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        resample([0, 100], np.zeros((3, 2)))
