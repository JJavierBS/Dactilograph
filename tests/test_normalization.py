import numpy as np
import pytest

from dactilograph.hand_detector import Hand
from dactilograph.normalization import MIDDLE_FINGER_MCP, WRIST, normalize_hand

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def make_hand(landmarks, handedness="Right", image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    return Hand(
        landmarks=np.asarray(landmarks, dtype=float),
        world_landmarks=np.zeros((21, 3)),
        handedness=handedness,
        handedness_score=1.0,
        image_width=image_width,
        image_height=image_height,
    )


@pytest.fixture
def landmarks():
    rng = np.random.default_rng(seed=0)
    points = np.column_stack(
        [rng.uniform(0.4, 0.6, 21), rng.uniform(0.4, 0.6, 21), rng.uniform(-0.05, 0.05, 21)]
    )
    points[WRIST] = [0.5, 0.65, 0.0]
    points[MIDDLE_FINGER_MCP] = [0.5, 0.5, -0.01]
    return points


def test_returns_63_values(landmarks):
    assert normalize_hand(make_hand(landmarks)).shape == (63,)


def test_wrist_is_at_origin_and_palm_measures_one(landmarks):
    points = normalize_hand(make_hand(landmarks)).reshape(21, 3)
    assert points[WRIST] == pytest.approx([0, 0, 0])
    assert np.linalg.norm(points[MIDDLE_FINGER_MCP]) == pytest.approx(1.0)


def test_position_in_the_image_does_not_matter(landmarks):
    moved = landmarks + [0.2, -0.1, 0.0]
    assert normalize_hand(make_hand(moved)) == pytest.approx(normalize_hand(make_hand(landmarks)))


def test_distance_to_the_camera_does_not_matter(landmarks):
    center = landmarks[WRIST]
    closer = center + (landmarks - center) * 1.8
    assert normalize_hand(make_hand(closer)) == pytest.approx(normalize_hand(make_hand(landmarks)))


def test_left_hand_matches_its_mirrored_right_hand(landmarks):
    mirrored = landmarks * [-1, 1, 1] + [1, 0, 0]
    left = normalize_hand(make_hand(mirrored, handedness="Left"))
    right = normalize_hand(make_hand(landmarks, handedness="Right"))
    assert left == pytest.approx(right)


def test_image_aspect_ratio_does_not_matter(landmarks):
    pixels = landmarks * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH]
    wide_width, wide_height = 1280, 480
    same_hand_in_wide_image = pixels / [wide_width, wide_height, wide_width]
    wide = normalize_hand(make_hand(same_hand_in_wide_image, image_width=1280, image_height=480))
    assert wide == pytest.approx(normalize_hand(make_hand(landmarks)))


def test_zero_size_hand_is_rejected():
    with pytest.raises(ValueError):
        normalize_hand(make_hand(np.full((21, 3), 0.5)))


def test_custom_steps_replace_the_default_ones(landmarks):
    def drop_depth(points, hand):
        return points[:, :2]

    assert normalize_hand(make_hand(landmarks), steps=[drop_depth]).shape == (42,)
