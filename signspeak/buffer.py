"""Turn a noisy stream of per-frame letter predictions into words.

A single frame's prediction jitters, so a letter only counts after the same
prediction holds for `hold_frames` consecutive frames above the confidence
threshold. Special gestures: open palm adds a space, closed fist deletes the
last character.
"""

from collections import deque

# Landmark indices used by the gesture heuristics.
FINGERTIPS = (8, 12, 16, 20)   # index, middle, ring, pinky
FINGER_PIPS = (6, 10, 14, 18)  # joints just below each fingertip
WRIST = 0


def fingers_extended(landmarks):
    """Count how many of the four fingers are extended (tip above its PIP joint)."""
    return sum(
        1
        for tip, pip in zip(FINGERTIPS, FINGER_PIPS)
        if landmarks[tip][1] < landmarks[pip][1]
    )


def gesture(landmarks):
    """Classify special control gestures. Returns 'space', 'delete', or None."""
    extended = fingers_extended(landmarks)
    if extended == 4:
        return "space"
    if extended == 0:
        return "delete"
    return None


class WordBuilder:
    """Accumulates stable predictions into the current word and sentence."""

    def __init__(self, hold_frames=12, min_confidence=0.75, history=5):
        self.hold_frames = hold_frames
        self.min_confidence = min_confidence
        self._stable = None
        self._stable_count = 0
        self._recent_gesture = deque(maxlen=history)
        self.current_word = ""
        self.sentence = ""

    def _reset_stability(self):
        self._stable = None
        self._stable_count = 0

    def update(self, prediction):
        """Feed one frame's prediction.

        prediction is (letter, confidence) or None when no hand is visible.
        Returns an event string when something changed: 'letter', 'space',
        'delete', or None.
        """
        if prediction is None:
            self._reset_stability()
            self._recent_gesture.clear()
            return None

        letter, confidence = prediction

        if confidence < self.min_confidence:
            self._reset_stability()
            return None

        if letter == self._stable:
            self._stable_count += 1
        else:
            self._stable = letter
            self._stable_count = 1

        if self._stable_count < self.hold_frames:
            return None

        # Same stable letter already locked in; ignore repeats until the sign changes.
        self._reset_stability()
        self.current_word += letter.lower()
        return "letter"

    def add_space(self):
        if self.current_word:
            self.sentence = (self.sentence + " " + self.current_word).strip()
            self.current_word = ""
            return "space"
        return None

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence = self.sentence[:-1].rstrip()
        else:
            return None
        return "delete"

    def clear(self):
        self._reset_stability()
        self.current_word = ""
        self.sentence = ""

    def display_text(self):
        parts = [p for p in (self.sentence, self.current_word) if p]
        return " ".join(parts)
