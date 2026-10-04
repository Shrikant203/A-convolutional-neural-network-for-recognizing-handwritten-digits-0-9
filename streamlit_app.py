"""
Task 6: Creating a Streamlit User Interface
==============================================
A friendly, guided web interface for the Task 1 CNN digit-classification
model (the from-scratch NumPy CNN, saved as cnn_digits_model.pkl).

Three ways to try the model:
  1. Browse a visual gallery of real sample digits and predict on them
  2. Draw a digit by hand with your mouse / finger
  3. Upload a photo or scan of a handwritten digit

REQUIRED FILES in the same folder as this script:
  - cnn_digits_model.pkl   the trained model weights
  - digit_utils.py         shared preprocessing (crop/scale/center + TTA),
                            also used by train_model.py so training and
                            inference always agree on how images are read

Run with:  streamlit run streamlit_app.py

Optional (for the hand-drawing tab):
  pip install "streamlit-drawable-canvas[image]"

To retrain the model on more heavily augmented data (recommended if
accuracy on hand-drawn digits still feels weak):
  python train_model.py
This regenerates cnn_digits_model.pkl in place - just re-run the app after.
"""

import pickle
import hashlib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image, ImageOps

# Optional dependency - the app still works without it (falls back to a
# clickable pixel-brush grid), but the real mouse/touch canvas is much
# nicer, so we try to import it. We deliberately catch *any* exception
# here (not just ImportError): recent streamlit-drawable-canvas releases
# are built on Streamlit's newer "Components v2" system, and a version
# mismatch between that package and the installed Streamlit raises a
# StreamlitAPIException at import time rather than an ImportError. Either
# way, we don't want a packaging issue to crash the whole app.
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
    CANVAS_IMPORT_ERROR = None
except Exception as _canvas_err:
    CANVAS_AVAILABLE = False
    CANVAS_IMPORT_ERROR = str(_canvas_err)

# ----------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "cnn_digits_model.pkl"
CLASS_NAMES = [str(i) for i in range(10)]
ACCENT = "#FF4B4B"          # Streamlit's default red, used consistently
GOOD = "#21C55D"
MID = "#F59E0B"
BAD = "#EF4444"

# ----------------------------------------------------------------------
# Global styling - makes the whole app feel considered, not default
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Professional, technical typeface for headings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    /* Tighten up the default top padding, and let the page breathe wider */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

    /* Bump the overall base font size so everything reads bigger and the
       wide layout doesn't feel empty */
    html, body, [class*="css"] { font-size: 17px; font-family: 'Inter', sans-serif; }
    p, li, label, .stMarkdown { font-size: 1.05rem; }
    h1 { font-size: 2.6rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.6rem !important; }
    h4 { font-size: 1.3rem !important; }

    /* Hero banner - restrained, technical, no playful gradients */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, #17181c 0%, #101114 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-top: 3px solid #FF4B4B;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.6rem;
    }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.14em;
        text-transform: uppercase; color: #9CA3AF;
        margin-bottom: 0.9rem;
    }
    .hero-eyebrow::before { content: ''; width: 8px; height: 8px; border-radius: 2px; background: #FF4B4B; display: inline-block; }
    .hero h1 {
        margin: 0 !important;
        font-family: 'Inter', sans-serif;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
        color: #F5F5F7;
    }
    .hero p.hero-sub { margin: 0 0 1.4rem 0; opacity: 0.72; font-size: 1.12rem; line-height: 1.6; max-width: 760px; }
    .hero-stats { display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1.1rem; }
    .hero-stat { padding: 0 1.6rem 0 0; margin-right: 1.6rem; border-right: 1px solid rgba(255,255,255,0.08); }
    .hero-stat:last-child { border-right: none; }
    .hero-stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: #F5F5F7; }
    .hero-stat-label { font-size: 0.82rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.15rem; }

    /* Big section-navigation buttons (replaces default tiny tabs) */
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        height: auto;
    }
    div.stButton > button {
        font-size: 1.08rem;
        font-weight: 600;
        padding: 0.7rem 1.1rem;
        border-radius: 10px;
    }

    /* Step badges used inside each section */
    .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 34px; height: 34px; border-radius: 50%;
        background: #FF4B4B; color: white; font-weight: 800; font-size: 1.1rem;
        margin-right: 12px; flex-shrink: 0;
    }
    .step-row { display: flex; align-items: flex-start; gap: 0.2rem; margin-bottom: 0.9rem; }
    .step-text { padding-top: 5px; font-size: 1.2rem; }

    /* Prediction result card */
    .result-card {
        border-radius: 18px; padding: 2rem 1.8rem; text-align: center;
        border: 2px solid var(--card-border); background: var(--card-bg);
    }
    .result-digit { font-size: 6rem; font-weight: 800; line-height: 1; margin: 0.3rem 0; }
    .result-label { font-size: 1.1rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em; }
    .confidence-pill {
        display: inline-block; padding: 0.4rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 1.15rem; margin-top: 0.6rem;
    }

    /* Sample gallery thumbnail button look */
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }

    /* Footer */
    .app-footer { text-align: center; opacity: 0.55; font-size: 0.9rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Model definition (matches the architecture trained and saved in Task 1)
# and inference-only forward pass - identical logic to the Task 4 API.
# ----------------------------------------------------------------------


def im2col(x, kh, kw, stride=1, pad=0):
    N, H, W, C = x.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    cols = np.zeros((N, out_h, out_w, kh, kw, C), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, :, i, j, :] = x[:, i:i_max:stride, j:j_max:stride, :]
    return cols.reshape(N, out_h, out_w, kh * kw * C), out_h, out_w


def conv_forward(x, W, b, stride=1, pad=1):
    N, H, Wd, C = x.shape
    k = W.shape[0]
    out_ch = W.shape[-1]
    cols, out_h, out_w = im2col(x, k, k, stride, pad)
    W_col = W.reshape(-1, out_ch)
    out = cols.reshape(N * out_h * out_w, -1) @ W_col + b
    return out.reshape(N, out_h, out_w, out_ch)


def relu(x):
    return np.maximum(0, x)


def maxpool_forward(x, size=2, stride=2):
    N, H, W, C = x.shape
    out_h, out_w = H // stride, W // stride
    x = x[:, :out_h * stride, :out_w * stride, :]
    out = np.zeros((N, out_h, out_w, C), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            window = x[:, i*stride:i*stride+size, j*stride:j*stride+size, :]
            out[:, i, j, :] = window.max(axis=(1, 2))
    return out


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class DigitCNN:
    def __init__(self, weights_path):
        with open(weights_path, "rb") as f:
            raw = pickle.load(f)
        # Newer weight files (from train_model.py) carry a "_meta" entry
        # with training info and a content fingerprint, so the sidebar can
        # prove exactly which model is loaded. Older files won't have it.
        self.w = {k: v for k, v in raw.items() if k != "_meta"}
        self.meta = raw.get("_meta")
        if self.meta is None:
            hasher = hashlib.sha256()
            for key in sorted(self.w):
                hasher.update(self.w[key].tobytes())
            self.meta = {
                "augmented_training": False,
                "weights_hash": hasher.hexdigest()[:10],
            }

    def predict_proba(self, x):
        """x: numpy array of shape (1, 8, 8, 1), pixel values in [0, 1]."""
        w = self.w
        x = conv_forward(x, w["conv1_W"], w["conv1_b"], stride=1, pad=1)
        x = relu(x)
        x = maxpool_forward(x, 2, 2)
        x = conv_forward(x, w["conv2_W"], w["conv2_b"], stride=1, pad=1)
        x = relu(x)
        x = maxpool_forward(x, 2, 2)
        x = x.reshape(x.shape[0], -1)
        x = x @ w["fc1_W"] + w["fc1_b"]
        x = relu(x)
        logits = x @ w["fc2_W"] + w["fc2_b"]
        return softmax(logits)


@st.cache_resource
def load_model():
    return DigitCNN(MODEL_PATH)


@st.cache_data
def load_sample_digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.images, d.target


try:
    model = load_model()
    MODEL_LOAD_ERROR = None
except Exception as e:  # pragma: no cover - defensive UI path
    model = None
    MODEL_LOAD_ERROR = str(e)

sample_images, sample_labels = load_sample_digits()

# Single source of truth for the accuracy shown anywhere in the UI, so the
# sidebar and the hero banner can never disagree with each other or with
# whatever model is actually loaded.
if model is not None and getattr(model, "meta", None) and model.meta.get("clean_val_accuracy"):
    DISPLAY_ACCURACY = f"{model.meta['clean_val_accuracy'] * 100:.1f}%"
else:
    DISPLAY_ACCURACY = "N/A"


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
# Cropping, scaling, centering and test-time-augmentation logic lives in
# digit_utils.py so the training script (train_model.py) can use the exact
# same pipeline when generating training data. Keeping one source of truth
# here is what guarantees training and inference never drift apart.
from digit_utils import (
    pixels_to_display_image,
    crop_to_ink,
    shift_2d,
    center_by_mass,
    dilate3,
    erode3,
    smart_preprocess_to_8x8,
    predict_with_tta as _predict_with_tta,
)


def confidence_style(confidence):
    if confidence >= 0.80:
        return GOOD, "High confidence"
    elif confidence >= 0.50:
        return MID, "Moderate confidence"
    else:
        return BAD, "Low confidence"


def predict_with_tta(pixels_8x8):
    return _predict_with_tta(model, pixels_8x8)


def render_prediction(pixels_8x8, key_prefix=""):
    """Runs the model on an 8x8 array (values 0-16) and renders a full,
    easy-to-read result: input preview, headline result card, top-3
    ranked guesses, and the full probability chart."""
    if model is None:
        st.error(
            f"The model file couldn't be loaded ({MODEL_LOAD_ERROR}). "
            f"Make sure **{MODEL_PATH}** is in the same folder as this app."
        )
        return

    with st.spinner("Running the digit through the CNN (with test-time augmentation)..."):
        probs = predict_with_tta(pixels_8x8)
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    color, confidence_word = confidence_style(confidence)

    st.markdown("#### Result")
    col1, col2, col3 = st.columns([1, 1.1, 1.3])

    with col1:
        st.image(
            pixels_to_display_image(pixels_8x8, size=260),
            caption="What the model actually saw (8×8 pixels)",
            width="stretch",
        )

    with col2:
        st.markdown(
            f"""
            <div class="result-card" style="--card-border:{color}55; --card-bg:{color}14;">
                <div class="result-label">Predicted digit</div>
                <div class="result-digit" style="color:{color};">{pred}</div>
                <span class="confidence-pill" style="background:{color}22; color:{color};">
                    {confidence*100:.1f}% · {confidence_word}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if confidence < 0.5:
            st.caption(
                "⚠️ The model isn't very sure here — try making the digit bolder, "
                "more centered, and filling more of the frame."
            )

    with col3:
        st.markdown("**Top 3 guesses**")
        top3 = np.argsort(probs)[::-1][:3]
        for rank, digit in enumerate(top3, start=1):
            p = float(probs[digit])
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}[rank]
            st.markdown(f"{medal} **{digit}** — {p*100:.1f}%")
            st.progress(min(max(p, 0.0), 1.0))

    with st.expander("See the full probability breakdown (all 10 digits)"):
        df = pd.DataFrame({
            "Digit": CLASS_NAMES,
            "Probability": probs,
            "Highlighted": [d == pred for d in range(10)],
        })
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Digit:N", sort=None, title="Digit"),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                color=alt.condition(
                    alt.datum.Highlighted, alt.value(ACCENT), alt.value("#5B7FDE55")
                ),
                tooltip=[alt.Tooltip("Digit:N"), alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")


def guide_step(number, text):
    st.markdown(
        f"""
        <div class="step-row">
            <div class="step-badge">{number}</div>
            <div class="step-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


CANVAS_FIX_INSTRUCTIONS = (
    "**Try this in your terminal, then restart the app:**\n"
    "```bash\n"
    "pip uninstall -y streamlit-drawable-canvas\n"
    "pip install --upgrade streamlit\n"
    'pip install "streamlit-drawable-canvas[image]"\n'
    "```\n"
    "This installs a version of the canvas built for your Streamlit version, "
    "along with the Pillow/numpy extras it needs to hand image data back to Python."
)


def render_pixel_grid_fallback():
    """A dependency-free way to 'draw' a digit: click cells to brighten them.
    Used whenever the real mouse-drawing canvas isn't available."""
    guide_step(1, "Click grid cells to \"paint\" them brighter — think of it as a very chunky brush.")
    guide_step(2, "Use the brush strength control to decide how bright each click makes a cell.")
    guide_step(3, "Press <b>Predict</b> once your digit looks recognizable in the live preview.")

    if "grid" not in st.session_state:
        st.session_state.grid = np.zeros((8, 8), dtype=np.int32)

    brush_strength = st.slider("Brush strength (brightness added per click)", 2, 16, 8, key="fallback_brush")

    top_l, top_r = st.columns([3, 1])
    with top_r:
        st.write("")
        if st.button("Load example '0'", width="stretch", key="fallback_load"):
            st.session_state.grid = sample_images[
                np.where(sample_labels == 0)[0][0]
            ].astype(np.int32).copy()
        if st.button("Clear grid", width="stretch", key="fallback_clear"):
            st.session_state.grid = np.zeros((8, 8), dtype=np.int32)
        st.write("")
        st.markdown("**Live preview**")
        st.image(pixels_to_display_image(st.session_state.grid), width="stretch")

    with top_l:
        for r in range(8):
            row_cols = st.columns(8)
            for c in range(8):
                val = int(st.session_state.grid[r, c])
                label = "⬛" if val == 0 else ("⬜" if val >= 12 else "◾")
                if row_cols[c].button(
                    label, key=f"cell_{r}_{c}",
                    help=f"Brightness {val}/16 — click to brighten",
                ):
                    st.session_state.grid[r, c] = min(16, val + brush_strength)
                    st.rerun()

    st.write("")
    if st.button("🔮 Predict", key="predict_grid", type="primary"):
        if st.session_state.grid.sum() == 0:
            st.warning("The grid looks empty — click some cells to draw a digit first!")
        else:
            st.divider()
            render_prediction(st.session_state.grid.astype(np.float64))


# ----------------------------------------------------------------------
# Section navigation state (used by both the sidebar and the main nav bar)
# ----------------------------------------------------------------------
if "active_section" not in st.session_state:
    st.session_state.active_section = "samples"

NAV_OPTIONS = [
    ("samples", "🖼️  Browse Sample Digits"),
    ("draw", "✏️  Draw Your Own"),
    ("upload", "📤  Upload a Photo"),
]


def render_nav_buttons(container, orientation="horizontal"):
    """Renders the section-switcher buttons into `container` (st or
    st.sidebar), either as a row or stacked vertically."""
    if orientation == "horizontal":
        slots = container.columns(3, gap="medium")
    else:
        slots = [container] * len(NAV_OPTIONS)

    for slot, (nav_key, nav_label) in zip(slots, NAV_OPTIONS):
        is_active = st.session_state.active_section == nav_key
        if slot.button(
            nav_label,
            key=f"nav_{orientation}_{nav_key}",
            type="primary" if is_active else "secondary",
            width="stretch",
        ):
            st.session_state.active_section = nav_key
            st.rerun()


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Digit Classifier")
    st.caption("A convolutional neural network, built from scratch in NumPy — no TensorFlow or PyTorch involved.")

    st.markdown("#### 🔍 Model info")
    if model is not None and getattr(model, "meta", None):
        meta = model.meta
        if meta.get("augmented_training"):
            st.success(
                f"**Robust model loaded** ✓\n\n"
                f"Fingerprint: `{meta.get('weights_hash', '?')}`\n\n"
                f"Clean-digit accuracy: {meta.get('clean_val_accuracy', 0) * 100:.1f}% · "
                f"Messy/rotated-digit accuracy: {meta.get('augmented_val_accuracy', 0) * 100:.1f}%"
            )
        else:
            st.warning(
                f"**Original (non-augmented) model** — fingerprint `{meta.get('weights_hash', '?')}`.\n\n"
                "This is the legacy model, trained only on pristine digits. "
                "Run `python train_model.py` and restart the app to switch to "
                "the robustness-trained version."
            )
        with st.popover("What is this fingerprint for?"):
            st.markdown(
                "Streamlit caches the loaded model in memory. If predictions "
                "don't improve after you swap in a new `cnn_digits_model.pkl`, "
                "check the fingerprint here first — if it hasn't changed, the "
                "app is still serving the old model from cache. **Fully stop "
                "and restart** `streamlit run streamlit_app.py` (a browser "
                "refresh alone is not enough), or use the app menu's "
                "**Clear cache**, then reload this page."
            )
    else:
        st.info("Model metadata unavailable.")

    m1, m2 = st.columns(2)
    m1.metric("Test accuracy", DISPLAY_ACCURACY)
    m2.metric("Input size", "8×8 px")

    st.divider()
    st.markdown("#### 🧭 Jump to a mode")
    render_nav_buttons(st.sidebar, orientation="vertical")
    st.caption(f"Currently open: **{dict(NAV_OPTIONS)[st.session_state.active_section]}**")

    st.divider()

    with st.expander("🧠 How does this actually work?", expanded=False):
        st.markdown(
            """
            1. Your digit is converted into an **8×8 grayscale grid** —
               64 numbers, each from 0 (black) to 16 (white).
            2. It passes through two **convolution + pooling** layers,
               which learn to detect strokes, curves and edges.
            3. A final **dense (fully-connected) layer** turns those
               features into a probability for each digit, 0–9.
            4. The digit with the highest probability is the prediction.

            **Architecture:** `Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Dense → Softmax`
            """
        )

    with st.expander("🎯 Why predictions can miss", expanded=False):
        st.markdown(
            """
            Before scoring anything, the app **crops your input to just the
            ink, scales each axis to fill the frame without collapsing thin
            shapes, and centers it by mass** — the same normalization used
            to build the training images. Loopy digits like an **8** or **6**
            specifically get extra care so they don't get crushed into a
            thin line that looks like a **1**.

            A few things can still trip the model up:
            - It was trained on **plain, single-stroke digits** — ornate
              serifs or a crossed **7** (a bar through the stem) can look
              unlike anything it has seen
            - Extremely thin or wobbly strokes lose detail once reduced
              to 8×8
            - A digit drawn at a steep angle is harder than one drawn
              upright
            """
        )

    with st.expander("💡 Tips for the best results", expanded=False):
        st.markdown(
            """
            - Keep the digit **thick, bold, and roughly centered**
            - Fill most of the drawing area — tiny digits lose detail
            - Draw a **plain 7** (no horizontal crossbar) and a **plain 1**
              (no serif foot) for best accuracy
            - Avoid cursive or stylized handwriting — simple block shapes
              work best
            - This model only understands **single digits (0–9)**, not
              letters or multi-digit numbers
            """
        )

    with st.expander("📊 About the training data", expanded=False):
        st.markdown(
            """
            Trained on the classic **UCI ML handwritten digits** dataset
            (via `sklearn.datasets.load_digits`) — 1,797 real handwritten
            digits, each normalized to **8×8 pixels with 16 gray levels**.

            The model itself is retrained (not just the preprocessing) on
            a heavily **augmented** version of that data — each digit is
            randomly rotated, rescaled, shifted, thickened/thinned, and
            slightly warped, then run through the exact same crop/center
            pipeline this app uses — so it has actually seen shapes much
            closer to real freehand drawing than the pristine originals.
            Run `python train_model.py` any time to regenerate the model
            with fresh augmented data.
            """
        )

    st.divider()
    st.markdown("#### 🚦 Confidence key")
    st.markdown(
        f"""
        <div style="line-height:2.1;">
        <span style="color:{GOOD};">●</span> High confidence (≥ 80%)<br>
        <span style="color:{MID};">●</span> Moderate confidence (50–80%)<br>
        <span style="color:{BAD};">●</span> Low confidence (&lt; 50%)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("🔄 Reset app", width="stretch", help="Clear any drawing, uploaded image, and results"):
        for k in ["grid", "gallery_seed", "selected_sample", "show_sample_result"]:
            st.session_state.pop(k, None)
        st.rerun()

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <span class="hero-eyebrow">Convolutional Neural Network · Implemented from scratch</span>
        <h1>Handwritten Digit Recognition</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "A small CNN, implemented in raw NumPy with no deep learning framework, classifying "
    "handwritten digits in real time. Browse labeled training examples, draw your own digit, "
    "or upload a photo to see it in action."
)


if MODEL_LOAD_ERROR:
    st.error(
        f"⚠️ Couldn't load the model file **{MODEL_PATH}**. Please make sure it's in the "
        f"same folder as this script, then refresh the page.\n\nDetails: {MODEL_LOAD_ERROR}"
    )

st.write("")
render_nav_buttons(st, orientation="horizontal")
section = st.session_state.active_section
st.write("")

# ------------------------------------------------------------------
# Section 1: visual gallery of real dataset samples
# ------------------------------------------------------------------
if section == "samples":
    guide_step(1, "Filter by digit (optional), then click <b>Use this</b> under any thumbnail below.")
    guide_step(2, "Press <b>Predict</b> to see what the model thinks.")

    filter_col, count_col = st.columns([2, 3])
    with filter_col:
        digit_filter = st.selectbox(
            "Filter gallery by true digit",
            ["All digits"] + [str(i) for i in range(10)],
            help="The gallery is pulled from the real test dataset, so each thumbnail's true label is known.",
        )

    if digit_filter == "All digits":
        candidate_idxs = np.arange(len(sample_images))
    else:
        candidate_idxs = np.where(sample_labels == int(digit_filter))[0]

    if "gallery_seed" not in st.session_state:
        st.session_state.gallery_seed = 0
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = int(candidate_idxs[0])

    with count_col:
        st.write("")
        if st.button("🔀 Shuffle gallery", help="Show a different random set of samples"):
            st.session_state.gallery_seed += 1

    rng = np.random.RandomState(st.session_state.gallery_seed)
    show_idxs = rng.choice(candidate_idxs, size=min(8, len(candidate_idxs)), replace=False)

    st.write("")
    cols = st.columns(4)
    for i, idx in enumerate(show_idxs):
        with cols[i % 4]:
            with st.container(border=True):
                st.image(
                    pixels_to_display_image(sample_images[idx], size=170),
                    width="stretch",
                )
                st.caption(f"True label: **{sample_labels[idx]}**")
                if st.button("Use this", key=f"use_{idx}", width="stretch"):
                    st.session_state.selected_sample = int(idx)

    st.divider()
    sel = st.session_state.selected_sample
    st.markdown(f"**Selected sample** — index `{sel}`, true label **{sample_labels[sel]}**")

    left, right = st.columns([1, 3])
    with left:
        st.image(pixels_to_display_image(sample_images[sel], size=220), width="stretch")

    with right:
        st.write("")
        if st.button("🔮 Predict", key="predict_sample", type="primary", width="stretch"):
            st.session_state.show_sample_result = True

    if st.session_state.get("show_sample_result"):
        st.divider()
        render_prediction(sample_images[sel].copy())

# ------------------------------------------------------------------
# Section 2: draw a digit (real canvas if available, pixel-brush fallback otherwise)
# ------------------------------------------------------------------
elif section == "draw":
    if CANVAS_AVAILABLE:
        guide_step(1, "Draw a single digit (0–9) in the black canvas below using your mouse, trackpad or touchscreen.")
        guide_step(2, "Make it thick and centered — use the brush size slider if it's too thin.")
        guide_step(3, "Press <b>Predict</b> when you're happy with it.")

        draw_col, preview_col = st.columns([3, 2])
        canvas_result = None
        canvas_error = None

        with draw_col:
            brush_size = st.slider("Brush size", 8, 40, 22, help="Thicker strokes are usually easier for the model to read.")
            try:
                canvas_result = st_canvas(
                    fill_color="white",
                    stroke_width=brush_size,
                    stroke_color="white",
                    background_color="black",
                    height=440,
                    width=440,
                    drawing_mode="freedraw",
                    return_image_data=True,
                    key="canvas",
                )
            except Exception as e:
                canvas_error = str(e)

            if canvas_result is not None:
                btn_col1, btn_col2 = st.columns(2)
                predict_drawn = btn_col1.button("🔮 Predict", key="predict_canvas", type="primary", width="stretch")
                if btn_col2.button("🧹 Clear canvas", width="stretch"):
                    st.rerun()

        if canvas_error is not None:
            st.warning(
                "⚠️ The drawing canvas failed to load in this environment — this usually means the "
                "installed `streamlit-drawable-canvas` package doesn't match your Streamlit version.\n\n"
                + CANVAS_FIX_INSTRUCTIONS
                + f"\n\nTechnical detail: `{canvas_error}`"
            )
            st.info("Meanwhile, here's a manual pixel-brush grid you can use instead:")
            render_pixel_grid_fallback()
        else:
            drawn_pixels = None
            if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
                gray = np.array(
                    Image.fromarray(canvas_result.image_data.astype(np.uint8), mode="RGBA").convert("L")
                )
                drawn_pixels = smart_preprocess_to_8x8(gray)

            with preview_col:
                st.markdown("#### Live model input preview")
                if drawn_pixels is not None and drawn_pixels.sum() > 0:
                    st.image(pixels_to_display_image(drawn_pixels, size=320), width="stretch",
                              caption="Auto-cropped, scaled and centered — this is what the model sees")
                else:
                    st.info("Start drawing to see a live preview of what the model will receive.")

            if predict_drawn:
                if drawn_pixels is None or drawn_pixels.sum() == 0:
                    st.warning("The canvas looks empty — draw a digit first!")
                else:
                    st.divider()
                    render_prediction(drawn_pixels)

    else:
        st.warning(
            "✏️ For the best experience (drawing with your mouse/finger), install the optional "
            "package:\n\n`pip install \"streamlit-drawable-canvas[image]\"`\n\n"
            + (f"Technical detail: `{CANVAS_IMPORT_ERROR}`\n\n" if CANVAS_IMPORT_ERROR else "")
            + "Until then, here's a manual pixel-brush fallback you can use instead."
        )
        render_pixel_grid_fallback()

# ------------------------------------------------------------------
# Section 3: upload an image
# ------------------------------------------------------------------
elif section == "upload":
    guide_step(1, "Upload a clear photo or scan with a single handwritten digit.")
    guide_step(2, "Check the processed preview below — toggle <b>Invert colors</b> if it looks wrong (white background vs. black).")
    guide_step(3, "Press <b>Predict</b>.")

    up_col, opt_col = st.columns([2, 1])
    with opt_col:
        st.markdown("**Options**")
        invert = st.checkbox(
            "Invert colors",
            value=True,
            help="Turn this ON for a dark digit on a light/white background (most photos). "
                 "Turn it OFF if your image already has a bright digit on a dark background.",
        )
        st.caption("Best results: good lighting, digit fills most of the frame, plain background.")

    with up_col:
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("L")
        gray_arr = np.array(img).astype(np.float64)
        if invert:
            gray_arr = 255.0 - gray_arr

        cropped_preview = crop_to_ink(gray_arr)
        arr = smart_preprocess_to_8x8(gray_arr)

        st.divider()
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**Original upload**")
            st.image(uploaded, width="stretch")
        with p2:
            st.markdown("**Auto-cropped to the digit**")
            if cropped_preview is not None:
                st.image(
                    Image.fromarray(cropped_preview.astype(np.uint8)).resize((260, 260), Image.NEAREST),
                    width="stretch",
                )
            else:
                st.warning("Couldn't find a clear digit in this photo — try the invert toggle, better lighting, or more contrast against the background.")
        with p3:
            st.markdown("**What the model will see**")
            st.image(pixels_to_display_image(arr, size=260), width="stretch")
            st.caption("If this doesn't look like a recognizable digit, try the invert toggle or a clearer photo.")

        st.write("")
        if st.button("🔮 Predict", key="predict_upload", type="primary"):
            if arr.sum() == 0:
                st.warning("No digit was detected in this image — please try another photo.")
            else:
                st.divider()
                render_prediction(arr)
    else:
        st.info("👆 Upload a PNG or JPG photo of a handwritten digit to get started.")

st.divider()
st.markdown(
    "<div class='app-footer'>Task 6 — Streamlit User Interface for Deep Learning Models · "
    "Model trained from scratch in NumPy in Task 1</div>",
    unsafe_allow_html=True,
)
