import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from dactilograph.paths import DATA_DIR

STATIC_LETTERS = (
    "A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
) 
NUM_FEATURES = 63
FEATURE_COLUMNS = [f"{axis}{point}" for point in range(NUM_FEATURES // 3) for axis in "xyz"]
COLUMNS = ["persona", "letra", *FEATURE_COLUMNS]
DEFAULT_DATASET_PATH = DATA_DIR / "landmarks.csv"


def append_samples(path: Path, person: str, letter: str, samples: Iterable[np.ndarray]) -> int:
    if letter not in STATIC_LETTERS:
        raise ValueError(f"Unknown letter: {letter}")
    rows = []
    for sample in samples:
        if len(sample) != NUM_FEATURES:
            raise ValueError(f"Expected {NUM_FEATURES} values per sample, got {len(sample)}")
        rows.append([person, letter, *(f"{value:.6f}" for value in sample)])

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if needs_header:
            writer.writerow(COLUMNS)
        writer.writerows(rows)
    return len(rows)


def count_samples(path: Path, person: str) -> Counter[str]:
    if not path.exists():
        return Counter()
    with path.open(newline="", encoding="utf-8") as file:
        return Counter(row["letra"] for row in csv.DictReader(file) if row["persona"] == person)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def recording_ids(dataset: pd.DataFrame) -> np.ndarray:
    labels = dataset["persona"] + "\0" + dataset["letra"]
    return (labels != labels.shift()).cumsum().to_numpy()


def recording_folds(dataset: pd.DataFrame, n_folds: int) -> np.ndarray:
    recordings = recording_ids(dataset)
    first_rows = pd.DataFrame({"letra": dataset["letra"].to_numpy(), "recording": recordings})
    first_rows = first_rows.drop_duplicates("recording")
    fold_of = dict(
        zip(
            first_rows["recording"],
            first_rows.groupby("letra").cumcount() % n_folds,
            strict=True,
        )
    )
    return np.array([fold_of[recording] for recording in recordings])
