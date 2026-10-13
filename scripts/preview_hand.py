"""Muestra la webcam con el esqueleto de la mano y la punta del índice normalizada.

Salir con q, Esc o cerrando la ventana.
"""

import cv2

from dactilograph.camera import Camera
from dactilograph.drawing import draw_hand
from dactilograph.fps import FpsCounter
from dactilograph.hand_detector import HandDetector
from dactilograph.normalization import normalize_hand

WINDOW_NAME = "Dactilograph"
QUIT_KEYS = {ord("q"), 27}
TEXT_COLOR = (0, 255, 0)
INDEX_FINGER_TIP = 8


def main() -> None:
    fps_counter = FpsCounter()
    with Camera() as camera, HandDetector() as detector:
        while True:
            frame = camera.read()
            if frame is None:
                print("Could not read a frame from the camera")
                break

            hand = detector.detect(frame)
            if hand is not None:
                draw_hand(frame, hand)
                x, y, z = normalize_hand(hand).reshape(21, 3)[INDEX_FINGER_TIP]
                cv2.putText(
                    frame,
                    f"Indice: x={x:+.2f} y={y:+.2f} z={z:+.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    TEXT_COLOR,
                    2,
                )

            fps = fps_counter.tick()
            cv2.putText(
                frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2
            )
            cv2.imshow(WINDOW_NAME, frame)

            pressed_key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if pressed_key in QUIT_KEYS or window_closed:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
