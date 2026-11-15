"""Record labeled hand-landmark samples from your webcam.

Hold a sign, press the matching letter key to save a sample. Samples are
appended as CSV rows to data/landmarks.csv: 42 normalized features + label.
Press q to quit.
"""

import csv
import os

import cv2

from signspeak.hands import HandTracker
from signspeak.model import normalize_landmarks

DATA_PATH = os.path.join("data", "landmarks.csv")


def main():
    os.makedirs("data", exist_ok=True)
    new_file = not os.path.exists(DATA_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")

    tracker = HandTracker()
    counts = {}
    print("Hold a sign and press its letter key to record. Press q to quit.")

    try:
        with open(DATA_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow([f"f{i}" for i in range(42)] + ["label"])

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror, so it feels natural

                landmarks, frame = tracker.process(frame)
                status = "hand detected - press a letter to save"
                if landmarks is None:
                    status = "no hand"

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                total = sum(counts.values())
                cv2.putText(frame, f"samples: {total}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("SignSpeak - collect data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                letter = chr(key).lower()
                if "a" <= letter <= "z" and landmarks is not None:
                    features = normalize_landmarks(landmarks)
                    writer.writerow([f"{v:.6f}" for v in features] + [letter])
                    f.flush()
                    counts[letter] = counts.get(letter, 0) + 1
                    print(f"saved '{letter}' ({counts[letter]} so far)")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone. {sum(counts.values())} new samples in {DATA_PATH}")
    for letter in sorted(counts):
        print(f"  {letter}: {counts[letter]}")


if __name__ == "__main__":
    main()
