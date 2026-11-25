import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SEQUENCES_PATH,
    FEATURE_COLUMNS,
    iter_sequences,
    load_dataset,
    load_sequences,
)
from dactilograph.paths import CLASSIFIER_PATH, DYNAMIC_CLASSIFIER_PATH, PROJECT_ROOT
from dactilograph.reference_hands import reference_hands
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

DESCRIPTION = "Exporta los modelos entrenados y los datos de referencia para la versión web."
WEB_DIR = PROJECT_ROOT / "web"
SEED = 0
STATIC_CASES = 60
NORMALIZATION_CASES = 40
SEQUENCES_PER_LETTER = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--web", type=Path, default=WEB_DIR, help="carpeta de la web")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"{path}  ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    args = parse_args()
    models_dir = args.web / "public" / "models"
    fixtures_dir = args.web / "tests" / "fixtures"
    rng = np.random.default_rng(SEED)

    static_model = joblib.load(CLASSIFIER_PATH)
    dynamic_model = joblib.load(DYNAMIC_CLASSIFIER_PATH)
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    sequences = list(iter_sequences(load_sequences(DEFAULT_SEQUENCES_PATH)))

    onnx_path = models_dir / "letter_classifier.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(static_classifier_onnx(static_model))
    print(f"{onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} KB)")
    write_json(models_dir / "letter_classifier.json", static_classifier_labels(static_model))
    write_json(models_dir / "dynamic_classifier.json", nearest_neighbors_payload(dynamic_model))
    references = reference_hands(dataset, sequences)
    write_json(models_dir / "reference_hands.json", reference_hands_payload(references))

    inputs = dataset[FEATURE_COLUMNS].to_numpy()
    chosen = rng.choice(len(inputs), STATIC_CASES, replace=False)
    write_json(
        fixtures_dir / "static_classifier.json",
        static_classifier_fixture(static_model, inputs[np.sort(chosen)]),
    )

    frames = [hand for sequence in sequences for hand in sequence.hands]
    picked = [frames[i] for i in np.sort(rng.choice(len(frames), NORMALIZATION_CASES, False))]
    hands = picked + [mirrored(hand) for hand in picked[: NORMALIZATION_CASES // 4]]
    write_json(fixtures_dir / "normalization.json", normalization_fixture(hands))

    per_letter: dict[str, list] = {}
    for sequence in sequences:
        per_letter.setdefault(sequence.letter, []).append(sequence)
    sample = [seq for group in per_letter.values() for seq in group[:SEQUENCES_PER_LETTER]]
    write_json(fixtures_dir / "sequences.json", sequence_fixture(dynamic_model, sample))


if __name__ == "__main__":
    main()
