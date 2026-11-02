from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from dactilograph.dataset import (
    FEATURE_COLUMNS,
    NO_DYNAMIC_LETTER,
    NUM_LANDMARKS,
    RecordedSequence,
)
from dactilograph.normalization import normalize_hand


def reference_hands(
    static_dataset: pd.DataFrame, sequences: Iterable[RecordedSequence]
) -> dict[str, np.ndarray]:
    means = static_dataset.groupby("letra")[FEATURE_COLUMNS].mean()
    references = {
        str(letter): row.to_numpy().reshape(NUM_LANDMARKS, 3) for letter, row in means.iterrows()
    }

    first_frames: dict[str, list[np.ndarray]] = defaultdict(list)
    for sequence in sequences:
        if sequence.letter not in references and sequence.letter != NO_DYNAMIC_LETTER:
            first_frames[sequence.letter].append(normalize_hand(sequence.hands[0]))
    for letter, frames in first_frames.items():
        references[letter] = np.mean(frames, axis=0).reshape(NUM_LANDMARKS, 3)
    return references
