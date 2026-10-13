import pytest

from dactilograph.fps import FpsCounter


def test_first_frame_has_no_fps_yet():
    assert FpsCounter().tick(now=0.0) == 0.0


def test_constant_frame_rate_is_measured():
    counter = FpsCounter()
    for frame in range(20):
        fps = counter.tick(now=frame * 0.1)
    assert fps == pytest.approx(10.0)


def test_single_slow_frame_is_smoothed():
    counter = FpsCounter(smoothing=0.9)
    for frame in range(10):
        counter.tick(now=frame * 0.1)
    fps = counter.tick(now=0.9 + 1.0)
    assert fps == pytest.approx(0.9 * 10.0 + 0.1 * 1.0)
