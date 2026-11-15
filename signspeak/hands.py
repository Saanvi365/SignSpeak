"""Hand detection and landmark extraction built on MediaPipe.

Uses the MediaPipe Tasks HandLandmarker API (mediapipe >= 0.10.14, including
1.x, where the legacy mp.solutions API was removed). The hand_landmarker.task
model file is downloaded automatically on first run.

A single frame goes in, a normalized 21-point landmark list (or None) comes
out, plus the annotated frame for display.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

NUM_LANDMARKS = 21

MODEL_PATH = os.path.join("models", "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# The standard 21-point hand skeleton, for drawing. The Tasks API does not
# ship the old mp.solutions drawing helpers, so we draw with OpenCV.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def ensure_model(path=MODEL_PATH):
    """Download the hand landmarker model if it is not already on disk."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading hand landmarker model to {path} (one time, ~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def draw_hand(frame, landmarks, color=(80, 220, 100)):
    """Draw the 21 landmarks and bone connections onto the frame (in place)."""
    points = [(int(x), int(y)) for x, y in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (60, 60, 60), 1)
    for i, (x, y) in enumerate(points):
        radius = 5 if i in (4, 8, 12, 16, 20) else 3  # fingertips stand out
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame


class HandTracker:
    """Detect one hand per frame and return its 21 landmarks."""

    def __init__(self, max_hands=1, detection_confidence=0.7,
                 tracking_confidence=0.6, model_path=MODEL_PATH):
        ensure_model(model_path)
        # tracking_confidence has no Tasks-API equivalent; the landmarker
        # smooths across frames internally in VIDEO mode.
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame, draw=True):
        """Run detection on a BGR frame.

        Returns (landmarks, annotated_frame) where landmarks is a list of
        21 (x, y) pixel tuples for the first detected hand, or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = None
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in hand]
            if draw:
                draw_hand(frame, landmarks)
        return landmarks, frame

    def close(self):
        self._landmarker.close()
