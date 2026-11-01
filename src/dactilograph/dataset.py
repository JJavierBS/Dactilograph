import csv
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dactilograph.hand_detector import Hand
from dactilograph.paths import DATA_DIR

STATIC_LETTERS = (
    "A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
) 
NUM_FEATURES = 63
NUM_LANDMARKS = NUM_FEATURES // 3
FEATURE_COLUMNS = [f"{axis}{point}" for point in range(NUM_LANDMARKS) for axis in "xyz"]
COLUMNS = ["persona", "letra", *FEATURE_COLUMNS]
DEFAULT_DATASET_PATH = DATA_DIR / "landmarks.csv"

DYNAMIC_LETTERS = ("J", "LL", "Ñ", "RR", "V", "W", "X", "Y", "Z")
# Clase negativa: mano quieta, transiciones entre letras o movimientos que no son ninguna letra.
NO_DYNAMIC_LETTER = "-"
SEQUENCE_LABELS = (*DYNAMIC_LETTERS, NO_DYNAMIC_LETTER)
WORLD_COLUMNS = [f"w{column}" for column in FEATURE_COLUMNS]
SEQUENCE_COLUMNS = [
    "persona", "letra", "secuencia", "t_ms", "mano", "confianza_mano", "ancho", "alto",
    *FEATURE_COLUMNS, *WORLD_COLUMNS,
]  # fmt: skip
DEFAULT_SEQUENCES_PATH = DATA_DIR / "sequences.csv"


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


@dataclass(frozen=True)
class RecordedSequence:
    person: str
    letter: str
    sequence_id: int
    timestamps_ms: list[float]
    hands: list[Hand]


def _format(values: np.ndarray) -> list[str]:
    return [f"{value:.6f}" for value in values.flatten()]


# Se guardan los puntos tal como los da MediaPipe, sin normalizar, para poder recalcular las
# características de otra forma sin tener que volver a grabar.
def append_sequence(
    path: Path, person: str, letter: str, timestamps_ms: Sequence[float], hands: Sequence[Hand]
) -> int:
    if letter not in SEQUENCE_LABELS:
        raise ValueError(f"Unknown sequence label: {letter}")
    if len(timestamps_ms) != len(hands):
        raise ValueError(f"Got {len(timestamps_ms)} timestamps for {len(hands)} hands")
    if len(hands) < 2:
        raise ValueError("A sequence needs at least 2 frames")

    sequence_id = _next_sequence_id(path)
    start_ms = timestamps_ms[0]
    rows = [
        [
            person,
            letter,
            sequence_id,
            round(timestamp_ms - start_ms),
            hand.handedness,
            f"{hand.handedness_score:.6f}",
            hand.image_width,
            hand.image_height,
            *_format(hand.landmarks),
            *_format(hand.world_landmarks),
        ]
        for timestamp_ms, hand in zip(timestamps_ms, hands, strict=True)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if needs_header:
            writer.writerow(SEQUENCE_COLUMNS)
        writer.writerows(rows)
    return sequence_id


def _next_sequence_id(path: Path) -> int:
    if not path.exists():
        return 1
    with path.open(newline="", encoding="utf-8") as file:
        return max((int(row["secuencia"]) for row in csv.DictReader(file)), default=0) + 1


def count_sequences(path: Path, person: str) -> Counter[str]:
    if not path.exists():
        return Counter()
    with path.open(newline="", encoding="utf-8") as file:
        sequences = {
            (row["letra"], row["secuencia"])
            for row in csv.DictReader(file)
            if row["persona"] == person
        }
    return Counter(letter for letter, _ in sequences)


def load_sequences(path: Path) -> pd.DataFrame:
    # Sin keep_default_na=False, una persona llamada "NA" o "None" se leería como vacía.
    return pd.read_csv(path, keep_default_na=False)


def iter_sequences(dataset: pd.DataFrame) -> Iterator[RecordedSequence]:
    for sequence_id, rows in dataset.groupby("secuencia", sort=True):
        landmarks = rows[FEATURE_COLUMNS].to_numpy().reshape(-1, NUM_LANDMARKS, 3)
        world_landmarks = rows[WORLD_COLUMNS].to_numpy().reshape(-1, NUM_LANDMARKS, 3)
        hands = [
            Hand(
                landmarks=points,
                world_landmarks=world_points,
                handedness=handedness,
                handedness_score=float(score),
                image_width=int(width),
                image_height=int(height),
            )
            for points, world_points, handedness, score, width, height in zip(
                landmarks,
                world_landmarks,
                rows["mano"],
                rows["confianza_mano"],
                rows["ancho"],
                rows["alto"],
                strict=True,
            )
        ]
        yield RecordedSequence(
            person=rows["persona"].iloc[0],
            letter=rows["letra"].iloc[0],
            sequence_id=int(sequence_id),
            timestamps_ms=rows["t_ms"].astype(float).tolist(),
            hands=hands,
        )
