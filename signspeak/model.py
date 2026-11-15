"""Landmark normalization and the letter classifier.

Raw landmark pixel coordinates depend on where the hand sits in the frame
and how far it is from the camera, so every sample is normalized before it
reaches the classifier: translate the wrist to the origin, then scale by the
wrist-to-middle-finger-MCP distance. That makes the features invariant to
position and hand size, which is what lets a model trained on your samples
generalize across the frame.
"""

import csv
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

WRIST = 0
MIDDLE_MCP = 9
FEATURE_DIM = 21 * 2  # 21 landmarks, (x, y) each


def normalize_landmarks(landmarks):
    """Turn 21 (x, y) landmarks into a scale/position-invariant feature vector."""
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (21, 2):
        raise ValueError(f"expected 21 (x, y) landmarks, got shape {pts.shape}")

    wrist = pts[WRIST].copy()
    pts -= wrist

    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        raise ValueError("degenerate hand landmarks (zero scale)")
    pts /= scale

    return pts.flatten()


def load_dataset(csv_path):
    """Load labeled samples saved by collect_data.py. Returns (X, y).

    The CSV starts with a header row, and a stray keypress during collection
    can leave a malformed line behind - bad rows are skipped and counted,
    never fatal.
    """
    X, y = [], []
    skipped = 0
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) != FEATURE_DIM + 1:
                if row:  # truly blank lines are ignored silently
                    skipped += 1
                continue
            *features, label = row
            try:
                X.append([float(v) for v in features])
            except ValueError:
                skipped += 1  # header row or a malformed sample
                continue
            y.append(label)
    if skipped:
        print(f"skipped {skipped} malformed rows")
    if not X:
        raise ValueError(f"no samples found in {csv_path} - run collect_data.py first")
    return np.asarray(X), np.asarray(y)


def train(csv_path, model_path, test_fraction=0.2, random_state=42):
    """Train the classifier and save it. Returns the classification report."""
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=random_state, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    report = classification_report(y_test, clf.predict(X_test))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    return report


class LetterClassifier:
    """Load a saved model and predict letters from raw landmarks."""

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found - train a model first with train.py"
            )
        self._clf = joblib.load(model_path)

    def predict(self, landmarks):
        """Return (letter, confidence) for one frame's landmarks."""
        features = normalize_landmarks(landmarks).reshape(1, -1)
        letter = self._clf.predict(features)[0]
        confidence = float(np.max(self._clf.predict_proba(features)))
        return letter, confidence
