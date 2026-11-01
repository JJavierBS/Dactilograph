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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SEQUENCES_PATH,
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
    return parser.parse_args()


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

    results = {}
    for name, model in candidates().items():
        results[name] = cross_val_predict(model, features, labels, groups=groups, cv=by_sequence)
        print(f"\n{name}: {accuracy_score(labels, results[name]):.1%} con secuencias no vistas")

    best_name = max(results, key=lambda name: accuracy_score(labels, results[name]))
    print(f"\nMejor modelo: {best_name}. Acierto por etiqueta con secuencias no vistas:")
    print(classification_report(labels, results[best_name], digits=3, zero_division=0))
    print_top_confusions(labels, results[best_name])

    final_model = candidates()[best_name].fit(features, labels)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.salida)
    print(f"\nModelo entrenado con todas las secuencias y guardado en {args.salida}")


if __name__ == "__main__":
    main()
