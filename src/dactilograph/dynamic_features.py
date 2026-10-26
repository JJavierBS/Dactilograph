from collections.abc import Sequence

import numpy as np

from dactilograph.hand_detector import Hand
from dactilograph.normalization import normalize_hand
from dactilograph.trajectory import trajectory_features


# El movimiento lo da la trayectoria; la forma media distingue letras con recorridos parecidos.
def sequence_features(hands: Sequence[Hand], timestamps_ms: Sequence[float]) -> np.ndarray:
    handshape = np.mean([normalize_hand(hand) for hand in hands], axis=0)
    return np.concatenate([trajectory_features(hands, timestamps_ms), handshape])
