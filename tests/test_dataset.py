import csv

import numpy as np
import pandas as pd
import pytest

from dactilograph.dataset import (
    COLUMNS,
    FEATURE_COLUMNS,
    NO_DYNAMIC_LETTER,
    SEQUENCE_COLUMNS,
    append_samples,
    append_sequence,
    count_samples,
    count_sequences,
    iter_sequences,
    load_dataset,
    load_sequences,
    recording_folds,
    recording_ids,
)
from dactilograph.hand_detector import Hand


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def test_new_file_gets_header_and_rows(tmp_path):
    path = tmp_path / "data" / "landmarks.csv"
    written = append_samples(path, "jose", "A", [np.zeros(63), np.ones(63)])
    rows = read_rows(path)
    assert written == 2
    assert rows[0] == COLUMNS
    assert rows[1][:3] == ["jose", "A", "0.000000"]
    assert len(rows) == 3


def test_appending_keeps_a_single_header(tmp_path):
    path = tmp_path / "landmarks.csv"
    append_samples(path, "jose", "A", [np.zeros(63)])
    append_samples(path, "jose", "B", [np.zeros(63)])
    rows = read_rows(path)
    assert rows.count(COLUMNS) == 1
    assert len(rows) == 3


def test_counts_only_the_given_person(tmp_path):
    path = tmp_path / "landmarks.csv"
    append_samples(path, "jose", "A", [np.zeros(63)] * 3)
    append_samples(path, "jose", "CH", [np.zeros(63)])
    append_samples(path, "ana", "A", [np.zeros(63)] * 5)
    assert count_samples(path, "jose") == {"A": 3, "CH": 1}


def test_missing_file_has_no_samples(tmp_path):
    assert count_samples(tmp_path / "landmarks.csv", "jose") == {}


def test_dynamic_letter_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        append_samples(tmp_path / "landmarks.csv", "jose", "J", [np.zeros(63)])


def test_sample_with_wrong_size_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        append_samples(tmp_path / "landmarks.csv", "jose", "A", [np.zeros(42)])


def test_load_dataset_reads_samples_back(tmp_path):
    path = tmp_path / "landmarks.csv"
    append_samples(path, "jose", "N", [np.arange(63)])
    dataset = load_dataset(path)
    assert dataset["letra"].tolist() == ["N"]
    assert dataset[FEATURE_COLUMNS].to_numpy()[0] == pytest.approx(np.arange(63))


def test_each_block_of_rows_is_a_recording():
    dataset = pd.DataFrame(
        {
            "persona": ["jose", "jose", "jose", "jose", "ana", "jose"],
            "letra": ["A", "A", "B", "A", "A", "A"],
        }
    )
    assert recording_ids(dataset).tolist() == [1, 1, 2, 3, 4, 5]


def test_recordings_of_each_letter_are_spread_across_folds():
    dataset = pd.DataFrame(
        {
            "persona": ["jose"] * 6,
            "letra": ["A", "A", "B", "A", "B", "B"],
        }
    )
    assert recording_folds(dataset, n_folds=2).tolist() == [0, 0, 0, 1, 1, 1]


def make_hand(seed=0, handedness="Right"):
    rng = np.random.default_rng(seed)
    return Hand(
        landmarks=rng.uniform(0, 1, (21, 3)),
        world_landmarks=rng.uniform(-0.1, 0.1, (21, 3)),
        handedness=handedness,
        handedness_score=0.9,
        image_width=640,
        image_height=480,
    )


def test_sequence_file_gets_header_and_one_row_per_frame(tmp_path):
    path = tmp_path / "data" / "sequences.csv"
    append_sequence(path, "jose", "Z", [1000, 1033, 1066], [make_hand(i) for i in range(3)])
    rows = read_rows(path)
    assert rows[0] == SEQUENCE_COLUMNS
    assert len(rows) == 4


def test_each_sequence_gets_a_new_id(tmp_path):
    path = tmp_path / "sequences.csv"
    hands = [make_hand(0), make_hand(1)]
    assert append_sequence(path, "jose", "Z", [0, 33], hands) == 1
    assert append_sequence(path, "ana", "J", [0, 33], hands) == 2


def test_sequence_is_read_back_as_recorded(tmp_path):
    path = tmp_path / "sequences.csv"
    hands = [make_hand(0, "Left"), make_hand(1, "Left"), make_hand(2, "Left")]
    append_sequence(path, "jose", "LL", [5000, 5040, 5075], hands)
    [sequence] = iter_sequences(load_sequences(path))
    assert (sequence.person, sequence.letter, sequence.sequence_id) == ("jose", "LL", 1)
    assert sequence.timestamps_ms == [0, 40, 75]
    for read_hand, hand in zip(sequence.hands, hands, strict=True):
        assert read_hand.landmarks == pytest.approx(hand.landmarks, abs=1e-6)
        assert read_hand.world_landmarks == pytest.approx(hand.world_landmarks, abs=1e-6)
        assert read_hand.handedness == "Left"
        assert read_hand.handedness_score == pytest.approx(0.9)
        assert (read_hand.image_width, read_hand.image_height) == (640, 480)


def test_sequences_are_kept_apart(tmp_path):
    path = tmp_path / "sequences.csv"
    append_sequence(path, "jose", "Z", [0, 33, 66], [make_hand(i) for i in range(3)])
    append_sequence(path, "jose", "Z", [0, 33], [make_hand(i) for i in range(2)])
    sequences = list(iter_sequences(load_sequences(path)))
    assert [len(sequence.hands) for sequence in sequences] == [3, 2]


def test_labels_that_look_empty_are_read_as_text(tmp_path):
    path = tmp_path / "sequences.csv"
    append_sequence(path, "NA", NO_DYNAMIC_LETTER, [0, 33], [make_hand(0), make_hand(1)])
    [sequence] = iter_sequences(load_sequences(path))
    assert (sequence.person, sequence.letter) == ("NA", NO_DYNAMIC_LETTER)


def test_counts_sequences_not_frames(tmp_path):
    path = tmp_path / "sequences.csv"
    hands = [make_hand(i) for i in range(4)]
    append_sequence(path, "jose", "Z", [0, 33, 66, 99], hands)
    append_sequence(path, "jose", "Z", [0, 33], hands[:2])
    append_sequence(path, "jose", NO_DYNAMIC_LETTER, [0, 33], hands[:2])
    append_sequence(path, "ana", "Z", [0, 33], hands[:2])
    assert count_sequences(path, "jose") == {"Z": 2, NO_DYNAMIC_LETTER: 1}
    assert count_sequences(tmp_path / "missing.csv", "jose") == {}


def test_static_letter_is_rejected_as_sequence(tmp_path):
    with pytest.raises(ValueError):
        append_sequence(tmp_path / "sequences.csv", "jose", "A", [0, 33], [make_hand()] * 2)


@pytest.mark.parametrize(
    ("timestamps", "num_hands"), [([0, 33, 66], 2), ([0], 1)], ids=["mismatch", "one frame"]
)
def test_invalid_sequence_is_rejected(tmp_path, timestamps, num_hands):
    with pytest.raises(ValueError):
        append_sequence(
            tmp_path / "sequences.csv", "jose", "Z", timestamps, [make_hand()] * num_hands
        )
