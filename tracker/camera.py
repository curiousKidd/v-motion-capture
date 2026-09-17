"""Small OpenCV capture iterator for the pose viewer."""

import time

import cv2


def frames(source=0, width=1280, height=720, fps=30):
    """Yield ``(frame, capture_timestamp_ms)`` from a camera or media file."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open source: {source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame, time.time() * 1000
    finally:
        capture.release()
