"""No-webcam check for the capture setup and timestamp."""

import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
sys.modules.setdefault("cv2", types.SimpleNamespace(
    CAP_PROP_FRAME_WIDTH=3,
    CAP_PROP_FRAME_HEIGHT=4,
    CAP_PROP_FPS=5,
    VideoCapture=None,
))
from tracker import camera


class FakeCapture:
    def __init__(self):
        self.settings = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, key, value):
        self.settings.append((key, value))

    def read(self):
        return True, "frame"

    def release(self):
        self.released = True


class Week1Test(unittest.TestCase):
    def test_camera_sets_defaults_and_yields_timestamp(self):
        capture = FakeCapture()
        original = camera.cv2.VideoCapture
        camera.cv2.VideoCapture = lambda source: capture
        try:
            stream = camera.frames(3)
            frame, timestamp = next(stream)
            stream.close()
        finally:
            camera.cv2.VideoCapture = original
        self.assertEqual(frame, "frame")
        self.assertIsInstance(timestamp, float)
        self.assertEqual(capture.settings, [(3, 1280), (4, 720), (5, 30)])
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
