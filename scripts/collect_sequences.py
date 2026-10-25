"""Graba secuencias de las letras con movimiento en el CSV de secuencias.

Uso: uv run scripts/collect_sequences.py --persona <nombre>
Teclas: a/d cambian de letra, espacio inicia una cuenta atrás y después graba --duracion segundos
(otro espacio la cancela), q o Esc para salir. La etiqueta "-" es para grabar negativos: mano
quieta, transiciones entre letras o movimientos que no son ninguna letra.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from dactilograph.camera import Camera
from dactilograph.dataset import (
    DEFAULT_SEQUENCES_PATH,
    NO_DYNAMIC_LETTER,
    SEQUENCE_LABELS,
    append_sequence,
    count_sequences,
)
from dactilograph.drawing import draw_hand
from dactilograph.hand_detector import Hand, HandDetector

WINDOW_NAME = "Dactilograph - recogida de secuencias"
QUIT_KEYS = {ord("q"), 27}
PREVIOUS_LETTER_KEY = ord("a")
NEXT_LETTER_KEY = ord("d")
RECORD_KEY = ord(" ")
DEFAULT_DURATION_S = 1.5
COUNTDOWN_S = 1.0
MIN_DETECTED_FRACTION = 0.5
TEXT_COLOR = (0, 255, 0)
COUNTDOWN_COLOR = (0, 165, 255)
RECORDING_COLOR = (0, 0, 255)
# cv2.putText solo dibuja ASCII.
DISPLAY_NAMES = {"Ñ": "N~", NO_DYNAMIC_LETTER: "- (ninguna)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona", required=True, help="nombre de quien hace los signos")
    parser.add_argument(
        "--salida", type=Path, default=DEFAULT_SEQUENCES_PATH, help="CSV de destino"
    )
    parser.add_argument(
        "--duracion",
        type=float,
        default=DEFAULT_DURATION_S,
        help="segundos que dura cada grabación",
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


def label_name(letter: str) -> str:
    return DISPLAY_NAMES.get(letter, letter)


def main() -> None:
    args = parse_args()
    person = args.persona.strip()
    if not person:
        raise SystemExit("--persona cannot be empty")
    if args.duracion <= 0:
        raise SystemExit("--duracion must be positive")
    counts = count_sequences(args.salida, person)
    letter_index = 0
    countdown_start: float | None = None
    recording_start: float | None = None
    total_frames = 0
    timestamps_ms: list[float] = []
    hands: list[Hand] = []
    last_result = ""

    with Camera() as camera, HandDetector() as detector:
        while True:
            frame = camera.read()
            if frame is None:
                print("Could not read a frame from the camera")
                break
            now = time.monotonic()

            hand = detector.detect(frame)
            letter = SEQUENCE_LABELS[letter_index]
            if hand is not None:
                draw_hand(frame, hand)

            if countdown_start is not None and now - countdown_start >= COUNTDOWN_S:
                countdown_start, recording_start = None, now
            if recording_start is not None:
                total_frames += 1
                if hand is not None:
                    timestamps_ms.append(now * 1000)
                    hands.append(hand)
                if now - recording_start >= args.duracion:
                    detected = len(hands) / total_frames
                    if detected >= MIN_DETECTED_FRACTION and len(hands) >= 2:
                        append_sequence(args.salida, person, letter, timestamps_ms, hands)
                        counts[letter] += 1
                        last_result = f"Guardada ({len(hands)} fotogramas)"
                    else:
                        last_result = f"Descartada: mano solo en el {detected:.0%}"
                    recording_start, total_frames, timestamps_ms, hands = None, 0, [], []

            put_text(frame, f"Letra: {label_name(letter)}", 1, scale=1.0)
            put_text(frame, f"Secuencias de {label_name(letter)}: {counts[letter]}", 2)
            if countdown_start is not None:
                remaining = COUNTDOWN_S - (now - countdown_start)
                put_text(frame, f"PREPARADO... {remaining:.1f}", 3, COUNTDOWN_COLOR)
            elif recording_start is not None:
                elapsed = now - recording_start
                put_text(frame, f"GRABANDO {elapsed:.1f}/{args.duracion:.1f} s", 3, RECORDING_COLOR)
            else:
                put_text(frame, "a/d: letra  espacio: grabar  q: salir", 3, scale=0.5)
                if last_result:
                    put_text(frame, last_result, 4, scale=0.5)
            if hand is None:
                put_text(frame, "Sin mano", 5, RECORDING_COLOR)
            cv2.imshow(WINDOW_NAME, frame)

            pressed_key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if pressed_key in QUIT_KEYS or window_closed:
                break
            busy = countdown_start is not None or recording_start is not None
            if pressed_key == RECORD_KEY:
                if busy:
                    last_result = "Cancelada"
                    countdown_start, recording_start = None, None
                    total_frames, timestamps_ms, hands = 0, [], []
                else:
                    countdown_start = now
            elif not busy and pressed_key == NEXT_LETTER_KEY:
                letter_index = (letter_index + 1) % len(SEQUENCE_LABELS)
            elif not busy and pressed_key == PREVIOUS_LETTER_KEY:
                letter_index = (letter_index - 1) % len(SEQUENCE_LABELS)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
