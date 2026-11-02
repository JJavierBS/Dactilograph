"""Juego para practicar el alfabeto: explota las burbujas haciendo la letra que llevan dentro.

Uso: uv run scripts/play.py
Teclas: r empieza una partida nueva, q o Esc para salir.
"""

import time

import cv2
import numpy as np

from dactilograph.camera import Camera
from dactilograph.classifier import DynamicLetterClassifier, LetterClassifier
from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SEQUENCES_PATH,
    DYNAMIC_LETTERS,
    STATIC_LETTERS,
    iter_sequences,
    load_dataset,
    load_sequences,
)
from dactilograph.drawing import draw_hand, draw_reference_hand
from dactilograph.dynamic_recognizer import DynamicRecognizer
from dactilograph.game import BUBBLE_RADIUS, POP_DURATION_S, BubbleGame
from dactilograph.hand_detector import HandDetector
from dactilograph.reference_hands import reference_hands
from dactilograph.static_stabilizer import StaticLetterStabilizer

WINDOW_NAME = "Dactilograph - burbujas"
QUIT_KEYS = {ord("q"), 27}
RESTART_KEY = ord("r")
ALL_LETTERS = (*STATIC_LETTERS, *DYNAMIC_LETTERS)
BUBBLE_COLOR = (255, 200, 80)
BUBBLE_OPACITY = 0.35
LETTER_COLOR = (255, 255, 255)
POP_COLOR = (80, 255, 255)
TEXT_COLOR = (255, 255, 255)
WARNING_COLOR = (0, 0, 255)
HELP_COLOR = (0, 255, 255)
HELP_BOX_SIZE = 160
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_centered(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    (width, height), _ = cv2.getTextSize(text, FONT, scale, thickness)
    origin = (center[0] - width // 2, center[1] + height // 2)
    cv2.putText(frame, text, origin, FONT, scale, color, thickness)


def draw_bubbles(frame: np.ndarray, game: BubbleGame, now_s: float) -> None:
    height, width = frame.shape[:2]
    radius = int(BUBBLE_RADIUS * height)
    overlay = frame.copy()
    centers = [(int(bubble.x * width), int(bubble.y * height)) for bubble in game.bubbles]
    for center in centers:
        cv2.circle(overlay, center, radius, BUBBLE_COLOR, -1)
    cv2.addWeighted(overlay, BUBBLE_OPACITY, frame, 1 - BUBBLE_OPACITY, 0, frame)
    for bubble, center in zip(game.bubbles, centers, strict=True):
        cv2.circle(frame, center, radius, BUBBLE_COLOR, 2)
        put_centered(frame, bubble.letter, center, 1.1, LETTER_COLOR, 3)

    for pop in game.pops:
        progress = (now_s - pop.popped_s) / POP_DURATION_S
        center = (int(pop.x * width), int(pop.y * height))
        cv2.circle(frame, center, int(radius * (1 + progress)), POP_COLOR, 3)


def draw_hud(frame: np.ndarray, game: BubbleGame, detected: str) -> None:
    width = frame.shape[1]
    cv2.putText(frame, f"Puntos: {game.score}", (10, 30), FONT, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, f"Nivel: {game.level}", (10, 60), FONT, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, f"Vidas: {game.lives}", (width - 140, 30), FONT, 0.8, TEXT_COLOR, 2)
    if detected:
        cv2.putText(frame, f"Haces: {detected}", (width - 180, 60), FONT, 0.7, TEXT_COLOR, 2)


def draw_help(frame: np.ndarray, letter: str, reference: np.ndarray) -> None:
    height = frame.shape[0]
    top_left = (10, height - HELP_BOX_SIZE - 10)
    bottom_right = (10 + HELP_BOX_SIZE, height - 10)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
    draw_reference_hand(frame, reference, top_left, HELP_BOX_SIZE, HELP_COLOR)
    label = f"Así se hace la {letter}"
    if letter in DYNAMIC_LETTERS:
        label += " (con movimiento)"
    cv2.putText(frame, label, (10, top_left[1] - 10), FONT, 0.6, HELP_COLOR, 2)


def draw_game_over(frame: np.ndarray, game: BubbleGame) -> None:
    height, width = frame.shape[:2]
    put_centered(frame, "FIN DE LA PARTIDA", (width // 2, height // 2 - 30), 1.4, WARNING_COLOR, 3)
    put_centered(
        frame,
        f"Puntos: {game.score}  -  r: jugar otra vez",
        (width // 2, height // 2 + 25),
        0.8,
        TEXT_COLOR,
        2,
    )


def main() -> None:
    classifier = LetterClassifier()
    dynamic_classifier = DynamicLetterClassifier()
    references = reference_hands(
        load_dataset(DEFAULT_DATASET_PATH),
        iter_sequences(load_sequences(DEFAULT_SEQUENCES_PATH)),
    )
    rng = np.random.default_rng()
    game = BubbleGame(ALL_LETTERS, rng, time.monotonic())
    stabilizer = StaticLetterStabilizer()
    dynamic_recognizer = DynamicRecognizer(dynamic_classifier.predict)

    with Camera() as camera, HandDetector() as detector:
        while True:
            frame = camera.read()
            if frame is None:
                print("Could not read a frame from the camera")
                break
            now = time.monotonic()

            hand = detector.detect(frame)
            prediction = classifier.predict(hand) if hand is not None else None
            made = [
                stabilizer.update(now, prediction),
                dynamic_recognizer.update(now * 1000, hand),
            ]
            for letter in made:
                if letter is not None:
                    game.try_pop(letter, now)
            game.update(now)

            if hand is not None:
                draw_hand(frame, hand)
            draw_bubbles(frame, game, now)
            detected = prediction.letter if prediction is not None else ""
            draw_hud(frame, game, detected)
            help_letter = game.help_letter(now)
            if help_letter is not None and help_letter in references:
                draw_help(frame, help_letter, references[help_letter])
            if game.is_over:
                draw_game_over(frame, game)
            cv2.imshow(WINDOW_NAME, frame)

            pressed_key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if pressed_key in QUIT_KEYS or window_closed:
                break
            if pressed_key == RESTART_KEY:
                game = BubbleGame(ALL_LETTERS, rng, now)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
