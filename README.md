# SignSpeak

Real-time American Sign Language (ASL) fingerspelling translator: it reads
hand signs from your webcam, recognizes letters, builds them into words and
sentences, and can speak the result out loud.

Originally prototyped for Technovation as a Colab notebook, rebuilt here as a
clean, runnable Python project.

## How it works

1. **Hand tracking** - MediaPipe extracts 21 hand landmarks from each webcam
   frame (`signspeak/hands.py`).
2. **Letter classification** - the landmarks are normalized (position and
   scale invariant) and classified by a small scikit-learn model
   (`signspeak/model.py`).
3. **Word building** - a stability buffer turns a stream of letters into
   words: hold a sign to lock it in, open palm to add a space, fist to
   backspace (`signspeak/buffer.py`).
4. **Speech output** - finished sentences can be spoken with a local
   text-to-speech engine (optional).

## Quick start

```bash
git clone https://github.com/Saanvi365/SignSpeak.git
cd SignSpeak
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires mediapipe >= 0.10.14 (works on mediapipe 1.x, which removed the old
`mp.solutions` API - this repo uses the newer Tasks HandLandmarker API).
On the first run the hand-tracking model (`models/hand_landmarker.task`,
~8 MB) downloads itself automatically; you just need to be online once.

### 1. Collect training samples (a few minutes)

The demo ships without a pretrained model - you teach it your own hand, which
also makes it robust to your webcam and lighting. For each letter, hold the
sign and press that letter's key to record samples.

```bash
python collect_data.py
```

- Hold a sign, then press the matching letter key (`a`-`z`) to save a sample.
- Collect ~50 samples per letter you want to recognize (a handful of letters
  is enough for a first demo - start with `a b c l o v y`).
- Press `q` to quit. Samples are appended to `data/landmarks.csv`.

### 2. Train the classifier

```bash
python train.py
```

Trains on `data/landmarks.csv`, prints per-letter accuracy, and saves
`models/asl_landmark_model.joblib`.

### 3. Run the live demo

```bash
python app.py
```

- Hold a letter sign steady to lock it into the current word.
- Open palm held steady = space, closed fist held steady = backspace.
- Press `s` to speak the sentence (needs `pyttsx3`), `c` to clear, `q` to quit.

## Optional: CNN on a full ASL dataset

`notebooks/train_cnn_colab.ipynb` trains a CNN on the Kaggle ASL Alphabet
image dataset (87k images, 29 classes) in Google Colab - the approach from the
original prototype. The landmark model above trains in seconds on your own
machine and is what the live demo uses by default.

## Project layout

```
signspeak/
  hands.py    MediaPipe hand detection and landmark extraction
  model.py    landmark normalization + classifier train/predict wrapper
  buffer.py   stability buffer that turns letters into words
collect_data.py   record your own labeled landmark samples
train.py          train and evaluate the letter classifier
app.py            real-time webcam demo
notebooks/train_cnn_colab.ipynb   CNN training on the full dataset (Colab)
```

## Tech stack

Python, OpenCV, MediaPipe, scikit-learn, NumPy, pyttsx3 (optional TTS).
