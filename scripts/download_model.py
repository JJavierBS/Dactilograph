"""Download the MediaPipe hand landmarker model into models/."""

import urllib.request

from dactilograph.paths import HAND_LANDMARKER_PATH

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> None:
    if HAND_LANDMARKER_PATH.exists():
        print(f"Model already present: {HAND_LANDMARKER_PATH}")
        return
    HAND_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, HAND_LANDMARKER_PATH)
    print(f"Saved to {HAND_LANDMARKER_PATH}")


if __name__ == "__main__":
    main()
