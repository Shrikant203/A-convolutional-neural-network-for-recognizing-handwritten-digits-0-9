"""
digit_utils.py
==============
Shared, framework-free image preprocessing used by BOTH the Streamlit app
(streamlit_app.py) and the training script (train_model.py).

Why this file exists: a model is only as good as the match between what it
was trained on and what it sees at inference time. Earlier versions of this
project trained on pristine, pre-normalized 8x8 digits but then fed the
model raw, off-center, arbitrarily-scaled freehand drawings at inference
time - a serious train/inference mismatch that no amount of clever
inference-time cropping can fully fix.

By putting the exact same crop -> scale -> center pipeline here and
importing it from both places, the training script can generate augmented
training examples that have gone through IDENTICAL preprocessing to what
the deployed app applies to a real drawing. That is the single biggest
lever for real-world accuracy.
"""

import numpy as np
from PIL import Image


def pixels_to_display_image(pixels_8x8, size=192):
    """8x8 array in [0,16] -> a crisp, upscaled PIL image for display."""
    arr = np.clip(pixels_8x8, 0, 16) / 16.0 * 255
    return Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.NEAREST)


def crop_to_ink(gray_arr, pad_frac=0.07):
    """Tightly crop a grayscale array (ink = brighter than background) down
    to just the bounding box of the drawn/written content, with a small
    margin. Returns a 0-255 float array, or None if nothing was found."""
    arr = gray_arr.astype(np.float64)
    if arr.max() <= arr.min():
        return None
    norm = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    threshold = max(25.0, norm.max() * 0.18)
    mask = norm > threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = (y1 - y0 + 1), (x1 - x0 + 1)
    pad_y, pad_x = max(1, int(h * pad_frac)), max(1, int(w * pad_frac))
    y0, y1 = max(0, y0 - pad_y), min(norm.shape[0] - 1, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(norm.shape[1] - 1, x1 + pad_x)
    return norm[y0:y1 + 1, x0:x1 + 1]


def shift_2d(arr, sy, sx):
    """Shift a 2D array by (sy, sx) pixels with zero-fill (no wraparound)."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    sy0, sy1 = max(0, sy), min(h, h + sy)
    sx0, sx1 = max(0, sx), min(w, w + sx)
    oy0, oy1 = max(0, -sy), min(h, h - sy)
    ox0, ox1 = max(0, -sx), min(w, w - sx)
    out[sy0:sy1, sx0:sx1] = arr[oy0:oy1, ox0:ox1]
    return out


def center_by_mass(canvas8):
    """Nudge an 8x8 array so its center of mass sits at the middle of the
    frame - the same normalization classic digit-recognition pipelines use."""
    total = canvas8.sum()
    if total <= 0:
        return canvas8
    yy, xx = np.mgrid[0:8, 0:8]
    cy, cx = (yy * canvas8).sum() / total, (xx * canvas8).sum() / total
    sy, sx = int(round(3.5 - cy)), int(round(3.5 - cx))
    return shift_2d(canvas8, sy, sx)


def dilate3(arr):
    """3x3 max filter - thickens strokes by ~1px."""
    stack = [shift_2d(arr, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return np.maximum.reduce(stack)


def erode3(arr):
    """3x3 min filter - thins strokes by ~1px."""
    stack = [shift_2d(arr, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return np.minimum.reduce(stack)


def smart_preprocess_to_8x8(gray_arr):
    """Turn an arbitrarily-sized grayscale array (ink = brighter than
    background) into the 8x8, 0-16 format the model expects - cropped to
    the ink's bounding box, scaled to fill most of the frame, and centered
    by mass.

    Each axis is scaled independently within a min/max footprint (not
    forced to share one aspect-ratio-preserving scale factor): a strictly
    proportional resize crushes a tall, narrow shape (a slim figure-eight,
    a looping cursive digit) down to just 1-2 pixels wide, destroying the
    very detail - loops, holes, curves - that distinguishes it from a
    plain "1". Independent-axis scaling with a floor keeps thin digits
    recognizable.
    """
    cropped = crop_to_ink(gray_arr, pad_frac=0.07)
    if cropped is None:
        return np.zeros((8, 8))

    ch, cw = cropped.shape
    FRAME_MAX = 7.0   # the digit's longer side can use nearly the full frame
    FRAME_MIN = 3.2   # ...but neither side is allowed to collapse below this

    base_scale = FRAME_MAX / max(ch, cw)
    new_h = int(round(ch * base_scale))
    new_w = int(round(cw * base_scale))
    new_h = int(np.clip(new_h, FRAME_MIN, 8))
    new_w = int(np.clip(new_w, FRAME_MIN, 8))

    resized = np.array(
        Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    ).astype(np.float64)

    canvas = np.zeros((8, 8), dtype=np.float64)
    off_y, off_x = (8 - new_h) // 2, (8 - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    canvas = center_by_mass(canvas)

    if canvas.max() > 0:
        canvas = canvas / canvas.max() * 16.0
    return canvas


def predict_with_tta(model, pixels_8x8):
    """Run the model on several small, realistic perturbations of the same
    input - shifted by +/-1px, and with strokes slightly thickened/thinned -
    then average the resulting probabilities. Smooths out a single forward
    pass's sensitivity to exact pixel alignment and stroke weight."""
    base = np.clip(pixels_8x8, 0, 16)
    variants = [base]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            variants.append(shift_2d(base, dy, dx))
    variants.append(np.clip(dilate3(base), 0, 16))
    variants.append(np.clip(erode3(base), 0, 16))

    probs_sum = np.zeros(10, dtype=np.float64)
    for v in variants:
        x = (v / 16.0).reshape(1, 8, 8, 1).astype(np.float32)
        probs_sum += model.predict_proba(x)[0]
    return probs_sum / len(variants)
