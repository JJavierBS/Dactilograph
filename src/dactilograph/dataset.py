import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from dactilograph.paths import DATA_DIR

STATIC_LETTERS = (
    "A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
)  # fmt: skip
NUM_FEATURES = 63
# Mismo orden que normalize_hand: x0, y0, z0, x1, y1, z1, ...
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
