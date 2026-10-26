import numpy as np
import pandas as pd
import pytest

from dactilograph.dataset import FEATURE_COLUMNS, NO_DYNAMIC_LETTER
from dactilograph.dynamic_features import sequence_features
from dactilograph.normalization import MIDDLE_FINGER_MCP, WRIST, normalize_hand
from dactilograph.static_negatives import FRAMES_PER_WINDOW, negatives_from_static
from dactilograph.trajectory import hand_trajectory


def normalized_hand(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, (21, 3))
    points[WRIST] = 0
    points[MIDDLE_FINGER_MCP] = [0, -1, 0]
    return points.flatten()


def static_dataset(recordings):
    rows = []
    for person, letter, num_frames in recordings:
        for frame in range(num_frames):
            rows.append([person, letter, *normalized_hand(frame)])
    return pd.DataFrame(rows, columns=["persona", "letra", *FEATURE_COLUMNS])


def test_one_negative_per_static_recording():
    dataset = static_dataset([("jose", "A", 100), ("jose", "B", 100), ("ana", "A", 100)])
    negatives = negatives_from_static(dataset)
    assert len(negatives) == 3
    assert {sequence.letter for sequence in negatives} == {NO_DYNAMIC_LETTER}
    assert [sequence.person for sequence in negatives] == ["jose", "jose", "ana"]
    assert all(len(sequence.hands) == FRAMES_PER_WINDOW for sequence in negatives)


def test_ids_do_not_clash_with_recorded_sequences():
    negatives = negatives_from_static(static_dataset([("jose", "A", 50), ("jose", "B", 50)]))
    ids = [sequence.sequence_id for sequence in negatives]
    assert len(set(ids)) == 2
    assert all(sequence_id < 0 for sequence_id in ids)


def test_short_recordings_are_skipped():
    dataset = static_dataset([("jose", "A", FRAMES_PER_WINDOW - 1), ("jose", "B", 100)])
    assert len(negatives_from_static(dataset)) == 1


def test_handshape_is_kept():
    dataset = static_dataset([("jose", "A", FRAMES_PER_WINDOW)])
    [negative] = negatives_from_static(dataset)
    for frame, hand in enumerate(negative.hands):
        assert normalize_hand(hand) == pytest.approx(normalized_hand(frame))


def test_hand_trembles_but_stays_in_place():
    [negative] = negatives_from_static(static_dataset([("jose", "A", 100)]))
    wrist = hand_trajectory(negative.hands)[:, 0]
    assert np.abs(wrist).max() < 0.2
    assert np.abs(wrist).max() > 0


def test_negatives_are_reproducible():
    dataset = static_dataset([("jose", "A", 100)])
    [first] = negatives_from_static(dataset, seed=1)
    [second] = negatives_from_static(dataset, seed=1)
    assert first.hands[10].landmarks == pytest.approx(second.hands[10].landmarks)


def test_sequence_features_combine_movement_and_handshape():
    [negative] = negatives_from_static(static_dataset([("jose", "A", 100)]))
    features = sequence_features(negative.hands, negative.timestamps_ms)
    expected_handshape = np.mean([normalize_hand(hand) for hand in negative.hands], axis=0)
    assert features.shape == (80 + 63,)
    assert features[80:] == pytest.approx(expected_handshape)
