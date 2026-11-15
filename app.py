"""SignSpeak live demo: webcam ASL fingerspelling to text (and speech).

Controls while running:
  hold a letter sign steady  -> letter is added to the current word
  hold an open palm          -> space (finishes the current word)
  hold a closed fist         -> backspace
  s                          -> speak the sentence (needs pyttsx3)
  c                          -> clear everything
  q                          -> quit
"""

import os

import cv2

from signspeak.buffer import WordBuilder, gesture
from signspeak.hands import HandTracker
from signspeak.model import LetterClassifier

MODEL_PATH = os.path.join("models", "asl_landmark_model.joblib")

# A control gesture must hold this many consecutive frames before it fires.
GESTURE_HOLD_FRAMES = 15


def speak(text):
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 not installed - pip install pyttsx3 to enable speech")
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"{MODEL_PATH} not found - run collect_data.py, then train.py first"
        )

    classifier = LetterClassifier(MODEL_PATH)
    tracker = HandTracker()
    builder = WordBuilder()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")

    gesture_streak = None
    gesture_count = 0
    print("SignSpeak running. q = quit, s = speak, c = clear.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            landmarks, frame = tracker.process(frame)

            prediction = None
            if landmarks is not None:
                control = gesture(landmarks)
                if control is not None:
                    if control == gesture_streak:
                        gesture_count += 1
                    else:
                        gesture_streak, gesture_count = control, 1
                    if gesture_count == GESTURE_HOLD_FRAMES:
                        if control == "space":
                            builder.add_space()
                        else:
                            builder.backspace()
                        gesture_streak, gesture_count = None, 0
                else:
                    gesture_streak, gesture_count = None, 0
                    prediction = classifier.predict(landmarks)

            builder.update(prediction)

            if prediction is not None:
                letter, confidence = prediction
                cv2.putText(
                    frame, f"{letter.upper()} ({confidence:.0%})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )

            text = builder.display_text()
            cv2.rectangle(frame, (0, frame.shape[0] - 50),
                          (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, text[-60:], (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("SignSpeak", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                spoken = builder.display_text()
                if spoken:
                    print("speaking:", spoken)
                    speak(spoken)
            if key == ord("c"):
                builder.clear()
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    final = builder.display_text()
    if final:
        print("final transcript:", final)


if __name__ == "__main__":
    main()
