import warnings
from collections.abc import Iterable, Sequence

import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from dactilograph.dataset import NUM_FEATURES, RecordedSequence
from dactilograph.dynamic_features import sequence_features
from dactilograph.hand_detector import Hand
from dactilograph.normalization import normalize_hand

ONNX_OPSET = 17
MODEL_DECIMALS = 6


def static_classifier_onnx(model: ClassifierMixin) -> bytes:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        onnx_model = to_onnx(
            model,
            initial_types=[("hand", FloatTensorType([None, NUM_FEATURES]))],
            options={id(model): {"zipmap": False}},
            target_opset=ONNX_OPSET,
        )
    return onnx_model.SerializeToString()


def static_classifier_labels(model: ClassifierMixin) -> dict:
    return {"labels": [str(label) for label in model.classes_]}


def nearest_neighbors_payload(model: KNeighborsClassifier) -> dict:
    if model.weights != "uniform" or model.effective_metric_ != "euclidean":
        raise ValueError("Only uniform euclidean k-NN can be exported")
    return {
        "k": int(model.n_neighbors),
        "labels": [str(model.classes_[index]) for index in model._y],
        "features": np.round(model._fit_X, MODEL_DECIMALS).tolist(),
    }


def reference_hands_payload(references: dict[str, np.ndarray]) -> dict:
    return {
        letter: np.round(points, MODEL_DECIMALS).tolist() for letter, points in references.items()
    }


def hand_payload(hand: Hand) -> dict:
    return {
        "landmarks": hand.landmarks.tolist(),
        "handedness": hand.handedness,
        "width": hand.image_width,
        "height": hand.image_height,
    }


def mirrored(hand: Hand) -> Hand:
    other = "Left" if hand.handedness == "Right" else "Right"
    return Hand(
        hand.landmarks,
        hand.world_landmarks,
        other,
        hand.handedness_score,
        hand.image_width,
        hand.image_height,
    )


def normalization_fixture(hands: Iterable[Hand]) -> list[dict]:
    return [{**hand_payload(hand), "normalized": normalize_hand(hand).tolist()} for hand in hands]


def static_classifier_fixture(model: ClassifierMixin, inputs: np.ndarray) -> dict:
    probabilities = model.predict_proba(inputs)
    return {
        "labels": [str(label) for label in model.classes_],
        "cases": [
            {
                "input": row.tolist(),
                "probabilities": probs.tolist(),
                "letter": str(model.classes_[int(np.argmax(probs))]),
            }
            for row, probs in zip(inputs, probabilities, strict=True)
        ],
    }


def sequence_fixture(
    model: KNeighborsClassifier, sequences: Sequence[RecordedSequence]
) -> list[dict]:
    cases = []
    for sequence in sequences:
        features = sequence_features(sequence.hands, sequence.timestamps_ms)
        cases.append(
            {
                "timestamps_ms": list(sequence.timestamps_ms),
                "hands": [hand_payload(hand) for hand in sequence.hands],
                "features": features.tolist(),
                "letter": str(model.predict([features])[0]),
            }
        )
    return cases
