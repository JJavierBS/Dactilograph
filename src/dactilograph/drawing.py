import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

from dactilograph.hand_detector import Hand

BONE_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 0, 255)
LABEL_COLOR = (0, 255, 0)
REFERENCE_MARGIN = 0.1


def draw_hand(frame: np.ndarray, hand: Hand) -> None:
    height, width = frame.shape[:2]
    points = (hand.landmarks[:, :2] * [width, height]).astype(int)

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start, end = points[connection.start], points[connection.end]
        cv2.line(frame, tuple(start), tuple(end), BONE_COLOR, 2)
    for point in points:
        cv2.circle(frame, tuple(point), 4, JOINT_COLOR, -1)

    wrist = points[0]
    label = f"{hand.handedness} ({hand.handedness_score:.2f})"
    cv2.putText(
        frame, label, (wrist[0] - 40, wrist[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 2
    )


def draw_reference_hand(
    frame: np.ndarray,
    points: np.ndarray,
    top_left: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    xy = points[:, :2]
    low, high = xy.min(axis=0), xy.max(axis=0)
    usable = size * (1 - 2 * REFERENCE_MARGIN)
    scale = usable / max(float((high - low).max()), 1e-6)
    offset = np.array(top_left) + size * REFERENCE_MARGIN + (usable - (high - low) * scale) / 2
    pixels = ((xy - low) * scale + offset).astype(int)

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start, end = pixels[connection.start], pixels[connection.end]
        cv2.line(frame, tuple(start), tuple(end), color, 2)
    for point in pixels:
        cv2.circle(frame, tuple(point), 3, color, -1)
