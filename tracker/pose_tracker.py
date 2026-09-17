"""Week 1 MediaPipe pose viewer (mp.tasks API — solutions API removed in mediapipe>=0.10.30)."""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

try:
    from .camera import frames
except ImportError:  # Supports: python tracker/pose_tracker.py
    from camera import frames

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# 33-point BlazePose topology — fixed by the model, not exposed by mp.tasks.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
    (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def source(value):
    return int(value) if value.isdigit() else value


def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", type=source,
                        help="camera index or image/video path")
    parser.add_argument("--save-frame", help="write first annotated frame and exit")
    args = parser.parse_args()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=RunningMode.VIDEO,
    )
    first = True
    last_time = time.perf_counter()
    # ponytail: full-body landmarks drawn; upper-body subset filtering lands in week 2 normalizer.
    with PoseLandmarker.create_from_options(options) as landmarker:
        for frame, capture_timestamp_ms in frames(args.source):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, int(capture_timestamp_ms))
            found = bool(result.pose_landmarks)
            if found:
                draw_landmarks(frame, result.pose_landmarks[0])
            now = time.perf_counter()
            cv2.putText(frame, f"FPS: {1 / max(now - last_time, 1e-6):.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            last_time = now
            if first:
                height, width = frame.shape[:2]
                print(f"Resolution: {width}x{height}")
                print(f"Landmarks found: {'yes' if found else 'no'}")
                first = False
            if args.save_frame:
                if not cv2.imwrite(args.save_frame, frame):
                    raise RuntimeError(f"Could not save frame: {args.save_frame}")
                print(f"Saved annotated frame: {args.save_frame}")
                return
            cv2.imshow("MediaPipe Pose", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
