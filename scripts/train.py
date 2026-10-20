"""Entrena los clasificadores candidatos, compara su acierto y guarda el mejor en models/.

Uso: uv run scripts/train.py
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    STATIC_LETTERS,
    load_dataset,
    recording_folds,
    recording_ids,
)
from dactilograph.paths import CLASSIFIER_PATH

TOP_CONFUSIONS = 10
SEED = 0


def candidates() -> dict[str, ClassifierMixin]:
    #Sin StandardScaler porque la normalización ya deja todo en la misma unidad (tamaños de palma)
    return {
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, default=DEFAULT_DATASET_PATH, help="CSV del dataset")
    parser.add_argument("--salida", type=Path, default=CLASSIFIER_PATH, help="modelo a guardar")
    return parser.parse_args()


def print_top_confusions(true_letters: pd.Series, predicted: np.ndarray) -> None:
    letters = [letter for letter in STATIC_LETTERS if letter in set(true_letters)]
    counts = confusion_matrix(true_letters, predicted, labels=letters)
    np.fill_diagonal(counts, 0)
    matrix = pd.DataFrame(counts, index=letters, columns=letters)
    pairs = matrix.stack().sort_values(ascending=False).head(TOP_CONFUSIONS)
    print(
        f"\nConfusiones más frecuentes (letra real -> predicha, de {len(true_letters)} muestras):"
    )
    for (real, predicted_letter), count in pairs.items():
        if count > 0:
            print(f"  {real:>2} -> {predicted_letter:<2} {count}")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.datos)
    features = dataset[FEATURE_COLUMNS].to_numpy()
    letters = dataset["letra"]
    recordings = recording_ids(dataset)
    print(
        f"{len(dataset)} muestras, {letters.nunique()} letras, {len(set(recordings))} grabaciones"
    )

    #Evaluación honesta: se prueba siempre con grabaciones que el modelo no ha visto.
    n_folds = pd.Series(recordings).groupby(letters.to_numpy()).nunique().min()
    if n_folds < 2:
        raise SystemExit("Each letter needs at least 2 recordings to evaluate the model")
    by_recording = PredefinedSplit(recording_folds(dataset, n_folds))
    by_frame = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    results = {}
    for name, model in candidates().items():
        predicted = cross_val_predict(model, features, letters, cv=by_recording)
        optimistic = cross_val_predict(model, features, letters, cv=by_frame)
        results[name] = predicted
        print(
            f"\n{name}: {accuracy_score(letters, predicted):.1%} con grabaciones no vistas "
            f"({accuracy_score(letters, optimistic):.1%} mezclando fotogramas, optimista)"
        )

    best_name = max(results, key=lambda name: accuracy_score(letters, results[name]))
    print(f"\nMejor modelo: {best_name}. Acierto por letra con grabaciones no vistas:")
    print(classification_report(letters, results[best_name], digits=3, zero_division=0))
    print_top_confusions(letters, results[best_name])

    final_model = candidates()[best_name].fit(features, letters)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.salida)
    print(f"\nModelo entrenado con todas las muestras y guardado en {args.salida}")


if __name__ == "__main__":
    main()
