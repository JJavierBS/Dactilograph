"""Check that the environment works: libraries import and the hand model runs."""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from dactilograph.paths import HAND_LANDMARKER_PATH


def main() -> None:
    if not HAND_LANDMARKER_PATH.exists():
        raise SystemExit("Hand model not found. Run: uv run scripts/download_model.py")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank_frame)
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect_for_video(image, timestamp_ms=0)

    print(f"OpenCV {cv2.__version__}, MediaPipe {mp.__version__}")
    print(f"Hand model OK (hands detected on a blank frame: {len(result.hand_landmarks)})")

    camera = cv2.VideoCapture(0)
    camera_ok, _ = camera.read()
    camera.release()
    print("Webcam OK" if camera_ok else "Webcam not available")


if __name__ == "__main__":
    main()
