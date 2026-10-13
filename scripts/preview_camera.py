"""Show the webcam feed with the current FPS. Quit with q, Esc or by closing the window."""

import cv2

from dactilograph.camera import Camera
from dactilograph.fps import FpsCounter

WINDOW_NAME = "Dactilograph"
QUIT_KEYS = {ord("q"), 27}
FPS_TEXT_COLOR = (0, 255, 0)


def main() -> None:
    fps_counter = FpsCounter()
    with Camera() as camera:
        while True:
            frame = camera.read()
            if frame is None:
                print("Could not read a frame from the camera")
                break

            fps = fps_counter.tick()
            cv2.putText(
                frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FPS_TEXT_COLOR, 2
            )
            cv2.imshow(WINDOW_NAME, frame)

            pressed_key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if pressed_key in QUIT_KEYS or window_closed:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
