"""Dibuja una mano del dataset junto a sus transformaciones afines y las matrices usadas.

Uso: uv run scripts/plot_augmentation.py [--letra L] [--salida figuras/aumento_afin.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

from dactilograph.augmentation import (
    MAX_ROTATION_DEGREES,
    MAX_SCALE_CHANGE,
    MAX_SHEAR,
    apply_affine,
    rotation_matrix,
    scale_matrix,
    shear_matrix,
)
from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    NUM_LANDMARKS,
    STATIC_LETTERS,
    load_dataset,
)
from dactilograph.paths import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "figuras" / "aumento_afin.png"
ORIGINAL_COLOR = "#b0b0b0"
TRANSFORMED_COLOR = "#1f77b4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--letra", default="L", choices=STATIC_LETTERS, help="letra a dibujar")
    parser.add_argument("--datos", type=Path, default=DEFAULT_DATASET_PATH, help="CSV del dataset")
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT, help="imagen a guardar")
    return parser.parse_args()


def draw_skeleton(axis: plt.Axes, points: np.ndarray, color: str) -> None:
    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start, end = points[connection.start], points[connection.end]
        axis.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2)
    axis.scatter(points[:, 0], points[:, 1], color=color, s=12, zorder=3)


def matrix_text(matrix: np.ndarray) -> str:
    rows = [" ".join(f"{value:6.2f}" for value in row) for row in matrix[:2]]
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.datos)
    rows = dataset.loc[dataset["letra"] == args.letra, FEATURE_COLUMNS].to_numpy()
    if len(rows) == 0:
        raise SystemExit(f"No samples of letter {args.letra} in {args.datos}")
    points = rows[0].reshape(NUM_LANDMARKS, 3)

    rotation = rotation_matrix(MAX_ROTATION_DEGREES)
    scale = scale_matrix(1 + MAX_SCALE_CHANGE, 1 - MAX_SCALE_CHANGE)
    shear = shear_matrix(MAX_SHEAR, MAX_SHEAR)
    panels = {
        "Original": np.eye(3),
        f"Rotación {MAX_ROTATION_DEGREES:.0f}°": rotation,
        f"Escala X +{MAX_SCALE_CHANGE:.0%}, Y -{MAX_SCALE_CHANGE:.0%}": scale,
        f"Inclinación {MAX_SHEAR}": shear,
        "Composición": rotation @ scale @ shear,
    }

    figure, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.2))
    for axis, (title, matrix) in zip(axes, panels.items(), strict=True):
        draw_skeleton(axis, points, ORIGINAL_COLOR)
        draw_skeleton(axis, apply_affine(points, matrix), TRANSFORMED_COLOR)
        axis.set_title(title)
        axis.set_xlabel(matrix_text(matrix), family="monospace")
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Transformaciones afines sobre la letra {args.letra}")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.salida, dpi=150, bbox_inches="tight")
    print(f"Figura guardada en {args.salida}")


if __name__ == "__main__":
    main()
