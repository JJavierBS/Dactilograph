import csv

import numpy as np
import pandas as pd
import pytest

from dactilograph.dataset import (
    COLUMNS,
    FEATURE_COLUMNS,
    append_samples,
    count_samples,
    load_dataset,
    recording_folds,
    recording_ids,
)


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
