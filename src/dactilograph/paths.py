from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
