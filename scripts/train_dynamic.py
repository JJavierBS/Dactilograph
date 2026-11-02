"""Entrena el clasificador de las letras con movimiento y lo guarda en models/.

Uso: uv run scripts/train_dynamic.py
Los negativos se generan a partir de las grabaciones de letras estáticas, además de los que haya
grabados con la etiqueta "-".
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dactilograph.augmentation import (
    augment_sequence,
    cross_val_predict_both,
    with_copies,
)
from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SEQUENCES_PATH,
    NO_DYNAMIC_LETTER,
    SEQUENCE_LABELS,
    iter_sequences,
    load_dataset,
    load_sequences,
)
from dactilograph.dynamic_features import sequence_features
from dactilograph.paths import DYNAMIC_CLASSIFIER_PATH
from dactilograph.static_negatives import negatives_from_static

MAX_FOLDS = 5
TOP_CONFUSIONS = 10
SEED = 0
DEFAULT_COPIES = 5
AUGMENTED_SUFFIX = " + aumento afín"
MIN_NEGATIVE_RECALL = 0.9


def candidates() -> dict[str, ClassifierMixin]:
    # Hay muchos más negativos que ejemplos de cada letra: balanced compensa el desequilibrio y
    # k=1 evita que los vecinos negativos se impongan por número.
    return {
        "k-NN (k=1)": KNeighborsClassifier(n_neighbors=1),
        "SVM (RBF)": SVC(class_weight="balanced", random_state=SEED),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--secuencias", type=Path, default=DEFAULT_SEQUENCES_PATH, help="CSV de secuencias"
    )
    parser.add_argument(
        "--estaticas",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="CSV de letras estáticas del que se sacan los negativos",
    )
    parser.add_argument(
        "--salida", type=Path, default=DYNAMIC_CLASSIFIER_PATH, help="modelo a guardar"
    )
    parser.add_argument(
        "--copias",
        type=int,
        default=DEFAULT_COPIES,
        help="copias por secuencia con transformaciones afines (0 para no comparar con aumento)",
    )
    return parser.parse_args()


def negative_recall(labels: np.ndarray, predicted: np.ndarray) -> float:
    return recall_score(labels, predicted, labels=[NO_DYNAMIC_LETTER], average="macro")


def macro_f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    return f1_score(labels, predicted, average="macro", zero_division=0)


def print_top_confusions(true_labels: np.ndarray, predicted: np.ndarray) -> None:
    labels = [label for label in SEQUENCE_LABELS if label in set(true_labels)]
    counts = confusion_matrix(true_labels, predicted, labels=labels)
    np.fill_diagonal(counts, 0)
    pairs = pd.DataFrame(counts, index=labels, columns=labels).stack()
    pairs = pairs[pairs > 0].sort_values(ascending=False).head(TOP_CONFUSIONS)
    print("\nConfusiones más frecuentes (real -> predicha):")
    for (real, predicted_label), count in pairs.items():
        print(f"  {real:>2} -> {predicted_label:<2} {count}")


def main() -> None:
    args = parse_args()
    if args.copias < 0:
        raise SystemExit("--copias cannot be negative")
    recorded = list(iter_sequences(load_sequences(args.secuencias)))
    negatives = negatives_from_static(load_dataset(args.estaticas))
    sequences = recorded + negatives

    features = np.array([sequence_features(seq.hands, seq.timestamps_ms) for seq in sequences])
    labels = np.array([seq.letter for seq in sequences])
    groups = np.array([seq.sequence_id for seq in sequences])
    counts = pd.Series(labels).value_counts()
    print(
        f"{len(recorded)} secuencias grabadas + {len(negatives)} negativos sacados de "
        f"letras estáticas. Por etiqueta: {counts.to_dict()}"
    )

    n_folds = min(int(counts.min()), MAX_FOLDS)
    if n_folds < 2:
        raise SystemExit("Each label needs at least 2 sequences to evaluate the model")
    # Cada secuencia va entera a un solo pliegue: se prueba siempre con secuencias no vistas.
    by_sequence = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    rng = np.random.default_rng(SEED)
    copies = np.array(
        [
            [
                sequence_features(augment_sequence(seq.hands, rng), seq.timestamps_ms)
                for _ in range(args.copias)
            ]
            for seq in sequences
        ]
    )

    transformed = np.array(
        [
            sequence_features(augment_sequence(seq.hands, rng), seq.timestamps_ms)
            for seq in sequences
        ]
    )

    splits = list(by_sequence.split(features, labels, groups))
    results = {}
    for name, model in candidates().items():
        results[name] = cross_val_predict_both(model, features, labels, transformed, splits)
        if args.copias:
            results[name + AUGMENTED_SUFFIX] = cross_val_predict_both(
                model, features, labels, transformed, splits, copies
            )

    print("\nCon secuencias no vistas (tal cual / con transformaciones afines):")
    for name, (on_original, on_transformed) in results.items():
        print(
            f"  {name}: F1 medio {macro_f1(labels, on_original):.3f} / "
            f"{macro_f1(labels, on_transformed):.3f}, negativos bien "
            f"{negative_recall(labels, on_original):.1%} / "
            f"{negative_recall(labels, on_transformed):.1%}"
        )

    usable = [
        name
        for name in results
        if all(negative_recall(labels, result) >= MIN_NEGATIVE_RECALL for result in results[name])
    ]
    if not usable:
        print(f"\nNingún modelo llega al {MIN_NEGATIVE_RECALL:.0%} de negativos bien")
    best_name = max(
        usable or results,
        key=lambda name: np.mean([macro_f1(labels, result) for result in results[name]]),
    )
    print(f"\nMejor modelo: {best_name}. Acierto por etiqueta con secuencias no vistas:")
    print(classification_report(labels, results[best_name][0], digits=3, zero_division=0))
    print_top_confusions(labels, results[best_name][0])

    final_model = candidates()[best_name.removesuffix(AUGMENTED_SUFFIX)]
    if best_name.endswith(AUGMENTED_SUFFIX):
        final_model.fit(*with_copies(features, labels, copies))
    else:
        final_model.fit(features, labels)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.salida)
    print(f"\nModelo entrenado con todas las secuencias y guardado en {args.salida}")


if __name__ == "__main__":
    main()
