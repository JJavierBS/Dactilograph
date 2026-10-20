"""Graba muestras de las letras estáticas en el CSV del dataset.

Uso: uv run scripts/collect_data.py --persona <nombre>
Teclas: a/d cambian de letra, espacio empieza a grabar (para sola al llegar a --muestras, o antes
con otro espacio), q o Esc para salir.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from dactilograph.camera import Camera
from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    STATIC_LETTERS,
    append_samples,
    count_samples,
)
from dactilograph.drawing import draw_hand
from dactilograph.hand_detector import HandDetector
from dactilograph.normalization import normalize_hand

WINDOW_NAME = "Dactilograph - recogida de datos"
QUIT_KEYS = {ord("q"), 27}
PREVIOUS_LETTER_KEY = ord("a")
NEXT_LETTER_KEY = ord("d")
RECORD_KEY = ord(" ")
DEFAULT_SAMPLES_PER_RECORDING = 100
TEXT_COLOR = (0, 255, 0)
RECORDING_COLOR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona", required=True, help="nombre de quien hace los signos")
    parser.add_argument("--salida", type=Path, default=DEFAULT_DATASET_PATH, help="CSV de destino")
    parser.add_argument(
        "--muestras",
        type=int,
        default=DEFAULT_SAMPLES_PER_RECORDING,
        help="muestras por grabación",
    )
    return parser.parse_args()


def put_text(
    frame: np.ndarray,
    text: str,
    row: int,
    color: tuple[int, int, int] = TEXT_COLOR,
    scale: float = 0.7,
) -> None:
    cv2.putText(frame, text, (10, 30 * row), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main() -> None:
    args = parse_args()
    person = args.persona.strip()
    if not person:
        raise SystemExit("--persona cannot be empty")
    if args.muestras < 1:
        raise SystemExit("--muestras must be at least 1")
    counts = count_samples(args.salida, person)
    letter_index = 0
    recording = False
    samples: list[np.ndarray] = []

    with Camera() as camera, HandDetector() as detector:
        while True:
            frame = camera.read()
            if frame is None:
                print("Could not read a frame from the camera")
                break

            hand = detector.detect(frame)
            letter = STATIC_LETTERS[letter_index]
            if hand is not None:
                draw_hand(frame, hand)
                if recording:
                    samples.append(normalize_hand(hand))
            if recording and len(samples) >= args.muestras:
                counts[letter] += append_samples(args.salida, person, letter, samples)
                samples, recording = [], False

            put_text(frame, f"Letra: {letter}", 1, scale=1.0)
            put_text(frame, f"Muestras de {letter}: {counts[letter] + len(samples)}", 2)
            if recording:
                progress = f"{len(samples)}/{args.muestras}"
                put_text(frame, f"GRABANDO ({progress})  espacio: parar", 3, RECORDING_COLOR)
            else:
                put_text(frame, "a/d: letra  espacio: grabar  q: salir", 3, scale=0.5)
            if hand is None:
                put_text(frame, "Sin mano", 4, RECORDING_COLOR)
            cv2.imshow(WINDOW_NAME, frame)

            pressed_key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if pressed_key in QUIT_KEYS or window_closed:
                break
            if pressed_key == RECORD_KEY:
                if recording:
                    counts[letter] += append_samples(args.salida, person, letter, samples)
                    samples = []
                recording = not recording
            elif not recording and pressed_key == NEXT_LETTER_KEY:
                letter_index = (letter_index + 1) % len(STATIC_LETTERS)
            elif not recording and pressed_key == PREVIOUS_LETTER_KEY:
                letter_index = (letter_index - 1) % len(STATIC_LETTERS)

    if samples:
        append_samples(args.salida, person, STATIC_LETTERS[letter_index], samples)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
