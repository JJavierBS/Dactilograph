import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from dactilograph.augmentation import (
    MAX_ROTATION_DEGREES,
    apply_affine,
    around_center,
    augment_normalized_hand,
    augment_sequence,
    cross_val_predict_both,
    random_affine,
    rotation_matrix,
    scale_matrix,
    shear_matrix,
    translation_matrix,
    with_copies,
)
from dactilograph.hand_detector import Hand
from dactilograph.normalization import MIDDLE_FINGER_MCP, WRIST, normalize_hand
from dactilograph.trajectory import hand_trajectory

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def make_hand(offset=(0.0, 0.0), handedness="Right"):
    rng = np.random.default_rng(0)
    landmarks = np.column_stack(
        [rng.uniform(0.4, 0.6, 21), rng.uniform(0.4, 0.6, 21), rng.uniform(-0.05, 0.05, 21)]
    )
    landmarks[WRIST] = [0.5, 0.65, 0.0]
    landmarks[MIDDLE_FINGER_MCP] = [0.5, 0.5, -0.01]
    return Hand(
        landmarks + [*offset, 0.0],
        np.zeros((21, 3)),
        handedness,
        1.0,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )


def transform_point(matrix, x, y):
    return apply_affine(np.array([[x, y, 0.0]]), matrix)[0, :2]


def test_each_basic_transformation_moves_a_point_as_expected():
    assert transform_point(translation_matrix(2, 3), 1, 1) == pytest.approx([3, 4])
    assert transform_point(scale_matrix(2, 0.5), 2, 2) == pytest.approx([4, 1])
    assert transform_point(rotation_matrix(90), 1, 0) == pytest.approx([0, 1])
    assert transform_point(shear_matrix(0.5, 0), 1, 2) == pytest.approx([2, 2])


def test_composing_matrices_equals_applying_them_in_order():
    scale, rotation = scale_matrix(2, 3), rotation_matrix(30)
    one_by_one = transform_point(rotation, *transform_point(scale, 1, 1))
    assert transform_point(rotation @ scale, 1, 1) == pytest.approx(one_by_one)


def test_center_of_a_transformation_does_not_move():
    matrix = around_center(rotation_matrix(40) @ scale_matrix(1.2, 0.8), (5, 7))
    assert transform_point(matrix, 5, 7) == pytest.approx([5, 7])
    assert transform_point(random_affine(np.random.default_rng(0), (5, 7)), 5, 7) == pytest.approx(
        [5, 7]
    )


def test_depth_does_not_change():
    points = np.array([[1.0, 2.0, 0.3], [4.0, 5.0, -0.2]])
    assert apply_affine(points, rotation_matrix(30))[:, 2] == pytest.approx(points[:, 2])


def test_random_affine_is_reproducible_and_varies():
    first = random_affine(np.random.default_rng(1))
    assert first == pytest.approx(random_affine(np.random.default_rng(1)))
    assert first != pytest.approx(random_affine(np.random.default_rng(2)))


def test_random_affine_keeps_the_rotation_small():
    rng = np.random.default_rng(0)
    limit = MAX_ROTATION_DEGREES + 15
    for _ in range(100):
        x, y = random_affine(rng)[:2, 0]
        assert abs(np.degrees(np.arctan2(y, x))) < limit


def test_normalization_already_cancels_translation_and_uniform_scale():
    hand = make_hand()
    pixels = hand.landmarks * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH]
    matrix = translation_matrix(30, -20) @ scale_matrix(1.5, 1.5)
    moved = apply_affine(pixels, matrix)
    moved[:, 2] *= 1.5
    moved_hand = Hand(
        moved / [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH],
        hand.world_landmarks,
        "Right",
        1.0,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    assert normalize_hand(moved_hand) == pytest.approx(normalize_hand(hand))


def test_augmented_hand_stays_normalized_but_changes():
    features = normalize_hand(make_hand())
    augmented = augment_normalized_hand(features, np.random.default_rng(0))
    points = augmented.reshape(21, 3)
    assert points[WRIST] == pytest.approx([0, 0, 0])
    assert np.linalg.norm(points[MIDDLE_FINGER_MCP]) == pytest.approx(1.0)
    assert augmented != pytest.approx(features)


def test_still_sequence_stays_still_because_all_frames_share_the_matrix():
    hands = [make_hand()] * 5
    wrist = hand_trajectory(augment_sequence(hands, np.random.default_rng(0)))[:, 0]
    assert wrist == pytest.approx(np.zeros_like(wrist))


def test_augmenting_a_sequence_changes_the_hand_but_keeps_its_data():
    hands = [make_hand(), make_hand(offset=(0.05, 0.0))]
    original_landmarks = hands[1].landmarks.copy()
    augmented = augment_sequence(hands, np.random.default_rng(0))
    assert augmented[0].landmarks[WRIST] == pytest.approx(hands[0].landmarks[WRIST])
    assert augmented[1].landmarks != pytest.approx(hands[1].landmarks)
    assert augmented[1].handedness == hands[1].handedness
    assert hands[1].landmarks == pytest.approx(original_landmarks)


def test_with_copies_repeats_the_label_of_each_sample():
    features = np.array([[0.0], [1.0]])
    copies = np.array([[[0.1], [0.2]], [[1.1], [1.2]]])
    all_features, labels = with_copies(features, np.array(["A", "B"]), copies)
    assert all_features.ravel() == pytest.approx([0, 1, 0.1, 0.2, 1.1, 1.2])
    assert labels.tolist() == ["A", "B", "A", "A", "B", "B"]


def test_copies_of_a_test_sample_are_not_used_for_training():
    features = np.array([[0.0], [1.0]])
    labels = np.array(["A", "B"])
    copies = features[:, np.newaxis].copy()
    splits = [(np.array([0]), np.array([1]))]
    model = KNeighborsClassifier(n_neighbors=1)
    transformed = features.copy()
    on_original, on_transformed = cross_val_predict_both(
        model, features, labels, transformed, splits, copies
    )
    assert on_original[1] == "A"
    assert on_transformed[1] == "A"


def test_each_test_sample_is_also_classified_transformed():
    features = np.array([[0.0], [10.0], [1.0], [9.0]])
    labels = np.array(["A", "B", "A", "B"])
    transformed = np.array([[0.0], [10.0], [9.0], [1.0]])
    splits = [(np.array([0, 1]), np.array([2, 3]))]
    model = KNeighborsClassifier(n_neighbors=1)
    on_original, on_transformed = cross_val_predict_both(
        model, features, labels, transformed, splits
    )
    assert on_original[2:].tolist() == ["A", "B"]
    assert on_transformed[2:].tolist() == ["B", "A"]
