import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dactilograph.paths import (
    CLASSIFIER_PATH,
    DYNAMIC_CLASSIFIER_PATH,
    HAND_LANDMARKER_PATH,
    PROJECT_ROOT,
)
from dactilograph.ui.base import Context, Screen
from dactilograph.ui.chalk import DIM, FAINT, PINK, SCRIPT, WHITE, Chalk
from dactilograph.ui.widgets import Button

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOTS_S = 0.4
STEP_Y = 300
STEP_GAP = 70
ERROR_LINES = 3


@dataclass(frozen=True)
class Step:
    label: str
    output: Path
    script: str


STEPS = (
    Step("Descargar el modelo de la mano", HAND_LANDMARKER_PATH, "download_model.py"),
    Step("Entrenar las letras estáticas", CLASSIFIER_PATH, "train.py"),
    Step("Entrenar las letras con movimiento", DYNAMIC_CLASSIFIER_PATH, "train_dynamic.py"),
)


def missing_steps() -> list[Step]:
    return [step for step in STEPS if not step.output.exists()]


class Preparation:
    def __init__(self, steps: list[Step]) -> None:
        self.steps = steps
        self.current = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self.error is None and self.current >= len(self.steps)

    def _run(self) -> None:
        for step in self.steps:
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / step.script)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not step.output.exists():
                lines = (result.stderr or result.stdout).strip().splitlines()
                self.error = "\n".join(lines[-ERROR_LINES:]) or "falló sin dar detalles"
                return
            self.current += 1


class PreparingScreen(Screen):
    def __init__(self, preparation: Preparation) -> None:
        self._preparation = preparation

    def draw(self, canvas: np.ndarray, chalk: Chalk, context: Context) -> list[Button]:
        preparation = self._preparation
        chalk.text("Preparando Dactilograph", (640, 150), 2.6, WHITE, 4, SCRIPT, "center")
        chalk.text(
            "Solo la primera vez: descargo y entreno los modelos con los datos grabados.",
            (640, 210),
            0.7,
            DIM,
            1,
            align="center",
        )
        for index, step in enumerate(preparation.steps):
            y = STEP_Y + index * STEP_GAP
            finished = index < preparation.current
            running = index == preparation.current and preparation.error is None
            failed = index == preparation.current and preparation.error is not None
            color = WHITE if finished or running else PINK if failed else FAINT
            chalk.circle((330, y - 10), 10, color, -1 if finished else 2)
            label = step.label
            if running:
                label += "." * (1 + int(context.now / DOTS_S) % 3)
            chalk.text(label, (370, y), 1.3, color, 2, SCRIPT)
            if failed:
                chalk.cross((330, y - 10), 14, PINK, 3)
        if preparation.error is not None:
            y = STEP_Y + len(preparation.steps) * STEP_GAP + 10
            chalk.text("No he podido preparar los modelos:", (330, y), 0.7, PINK, 1)
            lines = preparation.error.splitlines()
            for offset, line in enumerate(lines):
                chalk.text(line[:100], (330, y + 30 + offset * 26), 0.6, WHITE, 1)
            hint = "Corrige el problema y vuelve a abrir Dactilograph."
            chalk.text(hint, (330, y + 40 + len(lines) * 26), 0.6, DIM, 1)
        return []


class NoCameraScreen(Screen):
    tray_items = (("Reintentar", "r", "reintentar"), ("Salir", "q", "salir"))

    def key_action(self, key: int) -> str | None:
        return "reintentar" if key == ord("r") else None

    def draw(self, canvas: np.ndarray, chalk: Chalk, context: Context) -> list[Button]:
        chalk.text("No encuentro la cámara", (640, 280), 3.0, WHITE, 4, SCRIPT, "center")
        chalk.cross((640, 400), 34, PINK, 4)
        chalk.text(
            "Conecta una webcam, cierra las apps que la estén usando y pulsa r.",
            (640, 500),
            0.7,
            DIM,
            1,
            align="center",
        )
        return []
