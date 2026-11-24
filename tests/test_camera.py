import time

import numpy as np
import pytest

from dactilograph import camera as camera_module
from dactilograph.camera import Camera

FRAME_INTERVAL_S = 0.005


class FakeCapture:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.delivered = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        time.sleep(FRAME_INTERVAL_S)
        if self.delivered == self.num_frames:
            return False, None
        self.delivered += 1
        frame = np.zeros((2, 3, 3), np.uint8)
        frame[0, 0] = self.delivered
        return True, frame

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    capture = FakeCapture(num_frames=20)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda index: capture)
    return capture


def test_frames_are_mirrored(fake_capture):
    with Camera() as camera:
        frame = camera.read()
    assert frame[0, 2, 0] > 0
    assert frame[0, 0, 0] == 0


def test_each_frame_is_returned_once_with_its_capture_time(fake_capture):
    with Camera() as camera:
        readings = [camera.read_timed() for _ in range(5)]
    numbers = [int(frame[0, 2, 0]) for frame, _ in readings]
    times = [captured_s for _, captured_s in readings]
    assert numbers == sorted(set(numbers))
    assert times == sorted(set(times))


def test_slow_reader_gets_the_newest_frame(fake_capture):
    with Camera() as camera:
        camera.read()
        time.sleep(FRAME_INTERVAL_S * 6)
        frame = camera.read()
        assert int(frame[0, 2, 0]) >= fake_capture.delivered - 1


def test_read_returns_none_when_the_camera_stops(fake_capture):
    with Camera() as camera:
        frames = [camera.read() for _ in range(fake_capture.num_frames + 1)]
    assert frames[-1] is None


def test_close_releases_the_camera(fake_capture):
    Camera().close()
    assert fake_capture.released
