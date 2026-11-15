"""Train the ASL letter classifier on samples from collect_data.py."""

import os

from signspeak.model import train

DATA_PATH = os.path.join("data", "landmarks.csv")
MODEL_PATH = os.path.join("models", "asl_landmark_model.joblib")


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"{DATA_PATH} not found - run collect_data.py and record some letters first"
        )

    print(f"Training on {DATA_PATH} ...")
    report = train(DATA_PATH, MODEL_PATH)
    print(report)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
