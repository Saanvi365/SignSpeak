"""Hand detection and landmark extraction built on MediaPipe.

A single frame goes in, a normalized 21-point landmark list (or None) comes
out, plus the annotated frame for display.
"""

import cv2
import mediapipe as mp

NUM_LANDMARKS = 21


class HandTracker:
    """Detect one hand per frame and return its 21 landmarks."""

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.6):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def process(self, frame, draw=True):
        """Run detection on a BGR frame.

        Returns (landmarks, annotated_frame) where landmarks is a list of
        21 (x, y) pixel tuples for the first detected hand, or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        landmarks = None
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in hand.landmark]
            if draw:
                self._drawing.draw_landmarks(
                    frame,
                    hand,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self._styles.get_default_hand_landmarks_style(),
                    self._styles.get_default_hand_connections_style(),
                )
        return landmarks, frame

    def close(self):
        self._hands.close()
