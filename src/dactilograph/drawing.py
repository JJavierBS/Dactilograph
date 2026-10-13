import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

from dactilograph.hand_detector import Hand

BONE_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 0, 255)
LABEL_COLOR = (0, 255, 0)


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
