# Handwritten Digit Classifier

A convolutional neural network for recognizing handwritten digits (0–9),
implemented entirely from scratch in NumPy — no TensorFlow, PyTorch, or
any other ML framework — with a Streamlit interface for trying it live.

## What's in this project

| File | Purpose |
|---|---|
| `streamlit_app.py` | The web app. Run this to use the classifier. |
| `digit_utils.py` | Shared image preprocessing (crop, scale, center, test-time augmentation). Used by **both** the app and the training script, so training and inference always agree on how an image is read. |
| `train_model.py` | Trains the CNN from scratch — forward pass, backpropagation, and an Adam optimizer, all hand-implemented — on heavily augmented data. Produces `cnn_digits_model.pkl`. |
| `cnn_digits_model.pkl` | The trained model weights. Required for the app to run. |

## Requirements

```bash
pip install streamlit numpy pandas pillow altair scikit-learn
```

Optional, for drawing digits with your mouse/finger instead of clicking a
pixel grid:

```bash
pip install "streamlit-drawable-canvas[image]"
```

## Running the app

Make sure all four files above are in the **same folder**, then:

```bash
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

### Three ways to try it

1. **Browse Sample Digits** — a gallery of real, labeled digits from the
   training dataset. Pick one and predict.
2. **Draw Your Own** — draw a digit with your mouse or touchscreen (needs
   `streamlit-drawable-canvas`; falls back to a click-to-paint pixel grid
   if that package isn't installed).
3. **Upload a Photo** — upload a picture of a handwritten digit. Toggle
   "Invert colors" depending on whether your photo is a dark digit on a
   light background or the reverse.

Every prediction shows the predicted digit, a confidence level, the top-3
guesses, and — under "See the full probability breakdown" — the score for
all 10 digits.

## How it works

```
Input (drawing / photo)
        │
        ▼
  Crop to the ink, scale to fill the frame, center by mass   ← digit_utils.py
        │
        ▼
  8×8 grayscale grid, 16 gray levels
        │
        ▼
  Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Dense → ReLU → Dense → Softmax
        │
        ▼
  Test-time augmentation: average predictions over small pixel
  shifts + thickened/thinned versions of the input
        │
        ▼
  Final prediction + confidence
```

The preprocessing step matters as much as the model itself: an
uncropped, off-center, arbitrarily-scaled drawing looks nothing like the
clean, normalized digits the model was trained on. `digit_utils.py`
closes that gap by cropping to the drawn ink, scaling each axis
independently (so a tall, narrow shape like a slim figure-eight doesn't
get crushed into an unrecognizable sliver), and centering the result —
the same normalization the training data itself uses.

## Retraining the model

The shipped `cnn_digits_model.pkl` was trained on the classic
[UCI ML handwritten digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset)
(1,797 images, via `sklearn.datasets.load_digits`), heavily augmented with
random rotation, independent-axis scaling (including occasional extreme
stretches/squishes), translation, elastic warping, stroke
thickening/thinning, and noise — so it has seen shapes much closer to
real freehand drawing than the pristine originals.

To retrain (e.g. after tweaking augmentation in `train_model.py`):

```bash
python train_model.py
```

This will:

1. Run a numerical gradient check to confirm the hand-written
   backpropagation is mathematically correct (aborts if not).
2. Generate an augmented training set (~1–2 minutes).
3. Train for 24 epochs, printing accuracy on both clean and
   artificially-messy validation digits each epoch.
4. Save the best-scoring weights to `cnn_digits_model.pkl`, tagged with a
   content fingerprint and its measured accuracy.

**After retraining, fully restart the Streamlit app — don't just refresh
the browser.** Streamlit caches the loaded model in memory
(`@st.cache_resource`), so it won't notice the file on disk changed until
the process restarts. The sidebar's **Model info** panel shows the
fingerprint of whatever model is actually loaded, so you can confirm the
swap took effect before assuming something else is wrong.

## Known limitations

- **Training set size and resolution.** With ~1,800 training images at
  8×8 resolution, this is a small, simple model — not a production-grade
  recognizer. Expect it to do very well on clean, reasonably-proportioned
  digits and to struggle occasionally on unusual handwriting styles.
- **Extreme proportions.** A digit drawn far taller than it is wide (or
  vice versa) can become genuinely ambiguous once compressed to 8×8 — for
  example, a "4" with an exaggerated tail can visually collapse into
  something that looks like a "9" at that resolution. This is an
  information-loss limit of the format, not a bug.
- **Stylistic mismatches.** The training data uses plain, single-stroke
  digits. Ornate serifs, a "7" with a crossbar through the stem, or
  heavily cursive strokes can look unlike anything the model has seen.

For best results: draw digits thick, centered, filling most of the
frame, with roughly natural proportions and no extra flourishes.

## Credits

CNN architecture, forward pass, backpropagation, and training loop are
all custom NumPy implementations — no autograd or deep learning framework
is used anywhere in this project.
