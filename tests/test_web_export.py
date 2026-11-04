import json

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dactilograph.dataset import RecordedSequence
from dactilograph.hand_detector import Hand
from dactilograph.normalization import normalize_hand
from dactilograph.web_export import (
    mirrored,
    nearest_neighbors_payload,
    normalization_fixture,
    reference_hands_payload,
    sequence_fixture,
    static_classifier_fixture,
    static_classifier_labels,
    static_classifier_onnx,
)


def random_hand(rng, handedness="Right"):
    landmarks = rng.uniform(0.2, 0.8, (21, 3))
    return Hand(landmarks, np.zeros((21, 3)), handedness, 0.9, 640, 480)


def predict_like_exported_knn(payload, features):
    distances = np.linalg.norm(np.array(payload["features"]) - features, axis=1)
    return payload["labels"][int(np.argmin(distances))]


def test_exported_nearest_neighbors_predict_like_the_model():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(30, 5))
    labels = np.array(["J", "Z", "-"] * 10)
    model = KNeighborsClassifier(n_neighbors=1).fit(features, labels)
    payload = json.loads(json.dumps(nearest_neighbors_payload(model)))
    queries = rng.normal(size=(20, 5))
    expected = model.predict(queries)
    assert [predict_like_exported_knn(payload, q) for q in queries] == list(expected)


def test_only_uniform_euclidean_nearest_neighbors_are_exported():
    model = KNeighborsClassifier(n_neighbors=1, weights="distance").fit([[0.0], [1.0]], ["A", "B"])
    with pytest.raises(ValueError):
        nearest_neighbors_payload(model)


def test_static_classifier_converts_to_onnx():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 63))
    labels = np.array(["A", "B", "C"] * 20)
    model = CalibratedClassifierCV(SVC(), ensemble=False).fit(features, labels)
    assert static_classifier_onnx(model).startswith(b"\x08")
    assert static_classifier_labels(model) == {"labels": ["A", "B", "C"]}

    fixture = static_classifier_fixture(model, features[:3])
    assert fixture["labels"] == ["A", "B", "C"]
    for case in fixture["cases"]:
        assert sum(case["probabilities"]) == pytest.approx(1.0)
        assert case["letter"] == fixture["labels"][int(np.argmax(case["probabilities"]))]


def test_reference_hands_are_plain_lists():
    payload = reference_hands_payload({"A": np.ones((21, 3)) / 3})
    assert json.loads(json.dumps(payload))["A"][0] == [0.333333] * 3


def test_normalization_fixture_matches_normalize_hand():
    hand = random_hand(np.random.default_rng(1))
    [case] = normalization_fixture([hand])
    assert case["handedness"] == "Right"
    assert case["normalized"] == normalize_hand(hand).tolist()


def test_mirrored_only_swaps_the_side():
    hand = random_hand(np.random.default_rng(2))
    flipped = mirrored(hand)
    assert flipped.handedness == "Left"
    assert np.array_equal(flipped.landmarks, hand.landmarks)


def test_sequence_fixture_keeps_inputs_and_expected_letter():
    rng = np.random.default_rng(3)
    hands = [random_hand(rng) for _ in range(5)]
    sequence = RecordedSequence("p", "Z", 1, [0.0, 30.0, 60.0, 90.0, 120.0], hands)
    model = KNeighborsClassifier(n_neighbors=1).fit(rng.normal(size=(2, 143)), ["Z", "-"])
    [case] = sequence_fixture(model, [sequence])
    assert len(case["hands"]) == 5
    assert len(case["features"]) == 143
    assert case["letter"] in {"Z", "-"}
