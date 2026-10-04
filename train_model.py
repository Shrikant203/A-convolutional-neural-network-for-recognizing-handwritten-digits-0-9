"""
train_model.py
===============
Trains the digit-recognition CNN from scratch (forward AND backward pass,
pure NumPy - no autograd, no ML framework) and saves weights compatible
with the DigitCNN class in streamlit_app.py.

The key difference from a "normal" digits-dataset training script: every
training example is put through heavy, realistic augmentation (rotation,
anisotropic scaling, translation, stroke thickening/thinning, noise) and
then passed through digit_utils.smart_preprocess_to_8x8 - the SAME
crop/scale/center pipeline the deployed app applies to real drawings.
That closes the train/inference distribution gap directly, instead of
training on pristine textbook digits and hoping inference-time tricks
paper over the difference.

Run with:  python train_model.py
Produces:  cnn_digits_model.pkl  (drop into the same folder as the app)
"""

import pickle
import time
import hashlib
from datetime import datetime
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

from digit_utils import smart_preprocess_to_8x8, dilate3, erode3

RNG = np.random.default_rng(42)

# ------------------------------------------------------------------
# Forward-pass primitives (identical math to streamlit_app.py's DigitCNN,
# but each function also returns a "cache" so we can run backprop)
# ------------------------------------------------------------------


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


def col2im(dcols, x_shape, kh, kw, stride=1, pad=0):
    """Inverse of im2col: scatter-add gradient contributions back onto the
    (unpadded) input shape."""
    N, H, W, C = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    dcols = dcols.reshape(N, out_h, out_w, kh, kw, C)
    Hp, Wp = H + 2 * pad, W + 2 * pad
    dx_padded = np.zeros((N, Hp, Wp, C), dtype=dcols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            dx_padded[:, i:i_max:stride, j:j_max:stride, :] += dcols[:, :, :, i, j, :]
    if pad > 0:
        return dx_padded[:, pad:-pad, pad:-pad, :]
    return dx_padded


def conv_forward(x, W, b, stride=1, pad=1):
    k = W.shape[0]
    out_ch = W.shape[-1]
    cols, out_h, out_w = im2col(x, k, k, stride, pad)
    N = x.shape[0]
    cols_flat = cols.reshape(N * out_h * out_w, -1)
    W_col = W.reshape(-1, out_ch)
    out = (cols_flat @ W_col + b).reshape(N, out_h, out_w, out_ch)
    cache = (x, W, stride, pad, cols_flat, out_h, out_w)
    return out, cache


def conv_backward(dout, cache):
    x, W, stride, pad, cols_flat, out_h, out_w = cache
    N = x.shape[0]
    out_ch = W.shape[-1]
    k = W.shape[0]
    dout_flat = dout.reshape(N * out_h * out_w, out_ch)
    dW = (cols_flat.T @ dout_flat).reshape(W.shape)
    db = dout_flat.sum(axis=0)
    W_col = W.reshape(-1, out_ch)
    dcols = (dout_flat @ W_col.T).reshape(N, out_h, out_w, -1)
    dx = col2im(dcols, x.shape, k, k, stride, pad)
    return dx, dW, db


def relu_forward(x):
    return np.maximum(0, x), x


def relu_backward(dout, cache):
    return dout * (cache > 0)


def maxpool_forward(x, size=2, stride=2):
    N, H, W, C = x.shape
    out_h, out_w = H // stride, W // stride
    x_crop = x[:, :out_h * stride, :out_w * stride, :]
    out = np.zeros((N, out_h, out_w, C), dtype=x.dtype)
    mask = np.zeros_like(x_crop, dtype=bool)
    for i in range(out_h):
        for j in range(out_w):
            window = x_crop[:, i * stride:i * stride + size, j * stride:j * stride + size, :]
            m = window.max(axis=(1, 2), keepdims=True)
            out[:, i, j, :] = m[:, 0, 0, :]
            mask[:, i * stride:i * stride + size, j * stride:j * stride + size, :] |= (window == m)
    cache = (x.shape, mask, size, stride, out_h, out_w)
    return out, cache


def maxpool_backward(dout, cache):
    x_shape, mask, size, stride, out_h, out_w = cache
    N, H, W, C = x_shape
    dx_crop = np.zeros((N, out_h * stride, out_w * stride, C))
    for i in range(out_h):
        for j in range(out_w):
            window_mask = mask[:, i * stride:i * stride + size, j * stride:j * stride + size, :]
            count = window_mask.sum(axis=(1, 2), keepdims=True)
            count = np.where(count == 0, 1, count)
            contribution = dout[:, i:i + 1, j:j + 1, :] / count
            dx_crop[:, i * stride:i * stride + size, j * stride:j * stride + size, :] += window_mask * contribution
    dx = np.zeros((N, H, W, C))
    dx[:, :out_h * stride, :out_w * stride, :] = dx_crop
    return dx


def dense_forward(x, W, b):
    return x @ W + b, (x, W)


def dense_backward(dout, cache):
    x, W = cache
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss_and_grad(logits, labels):
    probs = softmax(logits)
    N = logits.shape[0]
    log_likelihood = -np.log(probs[np.arange(N), labels] + 1e-12)
    loss = log_likelihood.mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits


# ------------------------------------------------------------------
# The network: Conv(1->C1) -> ReLU -> Pool -> Conv(C1->C2) -> ReLU -> Pool
#              -> Dense(-> H) -> ReLU -> Dense(-> 10) -> Softmax
# ------------------------------------------------------------------

C1, C2, HIDDEN = 10, 20, 64


def init_params():
    def he(shape, fan_in):
        return (RNG.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float64)

    return {
        "conv1_W": he((3, 3, 1, C1), 3 * 3 * 1),
        "conv1_b": np.zeros(C1),
        "conv2_W": he((3, 3, C1, C2), 3 * 3 * C1),
        "conv2_b": np.zeros(C2),
        "fc1_W": he((2 * 2 * C2, HIDDEN), 2 * 2 * C2),
        "fc1_b": np.zeros(HIDDEN),
        "fc2_W": he((HIDDEN, 10), HIDDEN),
        "fc2_b": np.zeros(10),
    }


def forward(x, p):
    out1, c1 = conv_forward(x, p["conv1_W"], p["conv1_b"])
    a1, ca1 = relu_forward(out1)
    pool1, cp1 = maxpool_forward(a1)
    out2, c2 = conv_forward(pool1, p["conv2_W"], p["conv2_b"])
    a2, ca2 = relu_forward(out2)
    pool2, cp2 = maxpool_forward(a2)
    flat = pool2.reshape(pool2.shape[0], -1)
    d1, cd1 = dense_forward(flat, p["fc1_W"], p["fc1_b"])
    a3, ca3 = relu_forward(d1)
    logits, cd2 = dense_forward(a3, p["fc2_W"], p["fc2_b"])
    cache = dict(c1=c1, ca1=ca1, cp1=cp1, c2=c2, ca2=ca2, cp2=cp2,
                 pool2_shape=pool2.shape, cd1=cd1, ca3=ca3, cd2=cd2)
    return logits, cache


def backward(dlogits, cache):
    da3, dfc2W, dfc2b = dense_backward(dlogits, cache["cd2"])
    dd1 = relu_backward(da3, cache["ca3"])
    dflat, dfc1W, dfc1b = dense_backward(dd1, cache["cd1"])
    dpool2 = dflat.reshape(cache["pool2_shape"])
    da2 = maxpool_backward(dpool2, cache["cp2"])
    dout2 = relu_backward(da2, cache["ca2"])
    dpool1, dconv2W, dconv2b = conv_backward(dout2, cache["c2"])
    da1 = maxpool_backward(dpool1, cache["cp1"])
    dout1 = relu_backward(da1, cache["ca1"])
    _, dconv1W, dconv1b = conv_backward(dout1, cache["c1"])
    return {
        "conv1_W": dconv1W, "conv1_b": dconv1b,
        "conv2_W": dconv2W, "conv2_b": dconv2b,
        "fc1_W": dfc1W, "fc1_b": dfc1b,
        "fc2_W": dfc2W, "fc2_b": dfc2b,
    }


# ------------------------------------------------------------------
# Correctness check: numerical gradient checking on a tiny random batch.
# If this doesn't pass, the backward pass has a bug and training would
# silently produce a broken model - so we verify it BEFORE training.
# ------------------------------------------------------------------

def gradient_check():
    p = init_params()
    x = RNG.standard_normal((2, 8, 8, 1)) * 0.5
    labels = np.array([3, 7])

    logits, cache = forward(x, p)
    _, dlogits = cross_entropy_loss_and_grad(logits, labels)
    grads = backward(dlogits, cache)

    def loss_fn(params):
        logits, _ = forward(x, params)
        loss, _ = cross_entropy_loss_and_grad(logits, labels)
        return loss

    eps = 1e-5
    max_rel_err = 0.0
    for name in ["conv1_W", "conv1_b", "conv2_W", "conv2_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b"]:
        arr = p[name]
        flat = arr.reshape(-1)
        analytic_flat = grads[name].reshape(-1)
        # Sample a handful of entries per tensor rather than every single
        # one (fc1_W alone has 3000+ entries) - enough to catch real bugs.
        idxs = RNG.choice(flat.size, size=min(6, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus = loss_fn(p)
            flat[idx] = orig - eps
            loss_minus = loss_fn(p)
            flat[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = analytic_flat[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)

    print(f"Gradient check: max relative error = {max_rel_err:.2e}")
    assert max_rel_err < 1e-3, "Backward pass looks incorrect - aborting before training."
    print("Gradient check passed - backward pass is mathematically correct.\n")


# ------------------------------------------------------------------
# Data augmentation: generate many realistic, freehand-style variants of
# each training digit, each one passed through the SAME normalization the
# live app applies (digit_utils.smart_preprocess_to_8x8), so training and
# inference see identically-shaped data.
# ------------------------------------------------------------------

def elastic_warp(arr, upsample):
    """A cheap elastic-style warp: two smooth, randomly-phased sine
    displacement fields, so straight strokes gain the slight natural
    waviness real handwriting has (which pure affine transforms can't
    produce)."""
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    amp = upsample * 0.035
    freq_y = RNG.uniform(1.5, 3.0)
    freq_x = RNG.uniform(1.5, 3.0)
    phase_y = RNG.uniform(0, 2 * np.pi)
    phase_x = RNG.uniform(0, 2 * np.pi)
    disp_x = amp * np.sin(2 * np.pi * freq_y * yy / h + phase_y)
    disp_y = amp * np.sin(2 * np.pi * freq_x * xx / w + phase_x)
    src_x = np.clip(xx + disp_x, 0, w - 1).astype(np.int32)
    src_y = np.clip(yy + disp_y, 0, h - 1).astype(np.int32)
    return arr[src_y, src_x]


def augment_one(base_8x8, upsample=160):
    """Return one randomly-augmented 8x8, 0-16 version of base_8x8."""
    img255 = np.clip(base_8x8, 0, 16) / 16.0 * 255.0
    img = Image.fromarray(img255.astype(np.uint8)).resize((upsample, upsample), Image.LANCZOS)

    angle = RNG.uniform(-22, 22)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    # Independent per-axis scaling covers ordinary proportion variation.
    # On top of that, sometimes apply a deliberately EXTREME one-axis
    # stretch/squish - this is what makes the model robust to digits like
    # a "4" drawn with an unusually long tail (much taller than wide) or a
    # squat, wide "0" - shapes far outside a "normal" aspect ratio that
    # would otherwise never appear in training.
    if RNG.random() < 0.3:
        if RNG.random() < 0.5:
            sx, sy = RNG.uniform(0.45, 0.7), RNG.uniform(1.3, 1.7)
        else:
            sx, sy = RNG.uniform(1.3, 1.7), RNG.uniform(0.45, 0.7)
    else:
        sx = RNG.uniform(0.65, 1.4)
        sy = RNG.uniform(0.65, 1.4)
    new_w, new_h = max(4, int(upsample * sx)), max(4, int(upsample * sy))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (upsample, upsample), 0)
    paste_x = (upsample - new_w) // 2 if new_w <= upsample else 0
    paste_y = (upsample - new_h) // 2 if new_h <= upsample else 0
    if new_w > upsample or new_h > upsample:
        # An extreme stretch can exceed the canvas - center-crop it back down
        # rather than losing the paste entirely.
        left = max(0, (new_w - upsample) // 2)
        top = max(0, (new_h - upsample) // 2)
        img = img.crop((left, top, left + min(new_w, upsample), top + min(new_h, upsample)))
        new_w, new_h = img.size
        paste_x = (upsample - new_w) // 2
        paste_y = (upsample - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    img = canvas

    tx = RNG.integers(-upsample // 7, upsample // 7 + 1)
    ty = RNG.integers(-upsample // 7, upsample // 7 + 1)
    arr = np.array(img).astype(np.float64)
    from digit_utils import shift_2d
    arr = shift_2d(arr, ty, tx)

    if RNG.random() < 0.6:
        arr = elastic_warp(arr, upsample)

    if RNG.random() < 0.55:
        reps = RNG.integers(1, 3)
        for _ in range(reps):
            arr = dilate3(arr)
    elif RNG.random() < 0.35:
        arr = erode3(arr)

    noise = RNG.normal(0, 7, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    return smart_preprocess_to_8x8(arr)


def build_augmented_dataset(images, labels, augments_per_image=14):
    xs, ys = [], []
    for img, label in zip(images, labels):
        # Always include a "clean" pass through the same normalization
        # pipeline too, so the model still sees crisp, well-formed digits.
        clean255 = np.clip(img, 0, 16) / 16.0 * 255.0
        xs.append(smart_preprocess_to_8x8(clean255))
        ys.append(label)
        for _ in range(augments_per_image):
            xs.append(augment_one(img))
            ys.append(label)
    X = np.stack(xs).astype(np.float64)
    y = np.array(ys, dtype=np.int64)
    return X, y


# ------------------------------------------------------------------
# Adam optimizer (manual, matching the params dict structure)
# ------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(epochs=24, batch_size=64, augments_per_image=18, lr=1e-3):
    print("Loading base digit dataset...")
    digits = load_digits()
    images, labels = digits.images, digits.target

    n = len(images)
    idx = RNG.permutation(n)
    split = int(n * 0.85)
    train_idx, val_idx = idx[:split], idx[split:]

    print(f"Generating augmented training set ({augments_per_image}x per image, "
          f"~{len(train_idx) * (augments_per_image + 1)} examples)... this takes a minute.")
    t0 = time.time()
    X_train, y_train = build_augmented_dataset(images[train_idx], labels[train_idx], augments_per_image)
    print(f"  done in {time.time() - t0:.1f}s -> {X_train.shape[0]} training examples")

    print("Generating augmented validation set (to measure robustness to messy input)...")
    X_val_aug, y_val_aug = build_augmented_dataset(images[val_idx], labels[val_idx], augments_per_image=4)
    X_val_clean = np.stack([
        smart_preprocess_to_8x8(np.clip(img, 0, 16) / 16.0 * 255.0) for img in images[val_idx]
    ])
    y_val_clean = labels[val_idx]

    def to_input(X):
        return (np.clip(X, 0, 16) / 16.0).reshape(-1, 8, 8, 1).astype(np.float64)

    Xt = to_input(X_train)
    Xva, Xvc = to_input(X_val_aug), to_input(X_val_clean)

    params = init_params()
    opt = Adam(params, lr=lr)

    def accuracy(X, y):
        logits, _ = forward(X, params)
        preds = np.argmax(logits, axis=1)
        return (preds == y).mean()

    n_train = Xt.shape[0]
    best_score = -1
    best_params = None
    best_acc_clean = 0.0
    best_acc_aug = 0.0

    for epoch in range(1, epochs + 1):
        order = RNG.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            batch_idx = order[start:start + batch_size]
            xb, yb = Xt[batch_idx], y_train[batch_idx]
            logits, cache = forward(xb, params)
            loss, dlogits = cross_entropy_loss_and_grad(logits, yb)
            grads = backward(dlogits, cache)
            opt.step(params, grads)
            epoch_loss += loss
            n_batches += 1

        acc_clean = accuracy(Xvc, y_val_clean)
        acc_aug = accuracy(Xva, y_val_aug)
        # We care most about robustness to messy, augmented input - that's
        # what real users' drawings look like - so model selection weighs
        # it more heavily than clean-digit accuracy.
        score = 0.35 * acc_clean + 0.65 * acc_aug
        marker = ""
        if score > best_score:
            best_score = score
            best_params = {k: v.copy() for k, v in params.items()}
            best_acc_clean, best_acc_aug = acc_clean, acc_aug
            marker = "  <- best so far, saved"
        print(f"Epoch {epoch:2d}/{epochs} | loss {epoch_loss / n_batches:.4f} "
              f"| clean val acc {acc_clean:.3f} | augmented val acc {acc_aug:.3f}{marker}")

    print(f"\nBest combined score: {best_score:.3f}")
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "epochs": epochs,
        "augments_per_image": augments_per_image,
        "clean_val_accuracy": round(float(best_acc_clean), 4),
        "augmented_val_accuracy": round(float(best_acc_aug), 4),
        "augmented_training": True,
    }
    return best_params, meta


if __name__ == "__main__":
    print("Verifying backward pass correctness with a numerical gradient check...")
    gradient_check()

    trained_params, meta = train()

    # A short content hash lets the app prove *which* weights are actually
    # loaded, so "did my file swap take effect?" is never a mystery again.
    hasher = hashlib.sha256()
    for key in sorted(trained_params):
        hasher.update(trained_params[key].tobytes())
    meta["weights_hash"] = hasher.hexdigest()[:10]

    out_path = "cnn_digits_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({**trained_params, "_meta": meta}, f)
    print(f"\nSaved trained weights to {out_path}")
    print(f"Model fingerprint: {meta['weights_hash']}  (trained {meta['trained_at']})")
    print("Drop this file into the same folder as streamlit_app.py to use it.")
    print(
        "\nIMPORTANT: if streamlit is already running, a plain browser refresh is "
        "NOT enough - Streamlit caches the loaded model in memory. Fully stop and "
        "restart `streamlit run streamlit_app.py` (or use the app's menu -> "
        "'Clear cache') after replacing the file, then check the sidebar's "
        "'Model info' panel to confirm the new fingerprint is loaded."
    )
