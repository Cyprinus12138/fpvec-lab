"""Document degradation: stage 1 geometric similarity (tracked in G),
stage 2 appearance-only ops (Augraphy whitelist + own ops).

Alignment contract: stage 2 ops MUST NOT displace pixels — anything geometric is
expressed in G and propagated to the GT via H = N @ G @ A. The whitelist is
enforced by tests/test_fpgen_render.py::test_registration_alignment.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# stage 1: geometric similarity
# ---------------------------------------------------------------------------


def sample_geometric(img: np.ndarray, rng: np.random.Generator,
                     bg: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rotate/scale/translate the raster; returns (new_img, G 3x3 px->px)."""
    h, w = img.shape[:2]
    angle = 0.0
    if rng.random() < 0.6:
        angle = float(np.clip(rng.normal(0, 1.2), -3.0, 3.0))
    scale = float(rng.uniform(0.85, 1.05))
    a = math.radians(angle)
    ca, sa = math.cos(a) * scale, math.sin(a) * scale
    # y-down rotation matrix
    m = np.array([[ca, -sa], [sa, ca]])
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float) @ m.T
    minx, miny = corners.min(axis=0)
    maxx, maxy = corners.max(axis=0)
    pad = float(rng.uniform(2, 14))
    tx, ty = -minx + pad, -miny + pad
    out_w, out_h = int(math.ceil(maxx - minx + 2 * pad)), int(math.ceil(maxy - miny + 2 * pad))
    g = np.array([[ca, -sa, tx], [sa, ca, ty], [0.0, 0.0, 1.0]])
    warped = cv2.warpAffine(img, g[:2], (out_w, out_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=tuple(int(c) for c in bg))
    return warped, g


# ---------------------------------------------------------------------------
# stage 2: appearance ops (pixel-preserving whitelist)
# ---------------------------------------------------------------------------


def _to3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _gaussian_blur(img, rng):
    k = int(rng.choice([3, 3, 5]))
    return cv2.GaussianBlur(img, (k, k), 0)


def _motion_blur(img, rng):
    k = int(rng.choice([3, 5]))
    kernel = np.zeros((k, k))
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def _salt_pepper(img, rng):
    out = img.copy()
    amount = rng.uniform(0.0005, 0.004)
    n = int(amount * img.shape[0] * img.shape[1])
    ys = rng.integers(0, img.shape[0], n)
    xs = rng.integers(0, img.shape[1], n)
    out[ys[: n // 2], xs[: n // 2]] = 0
    out[ys[n // 2:], xs[n // 2:]] = 255
    return out


def _line_breakup(img, rng):
    """Erode dark strokes then knock out random pixels — scanned line decay."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = gray < 128
    holes = rng.random(dark.shape) < rng.uniform(0.02, 0.08)
    out = img.copy()
    out[dark & holes] = 255
    return out


def _jpeg(img, rng):
    q = int(rng.integers(40, 92))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _down_up(img, rng):
    f = float(rng.uniform(0.55, 0.9))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _watermark_tiled(img, rng):
    """stackedhomes-style diagonal tiled text watermark."""
    h, w = img.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    text = str(rng.choice(["PROPERTYHUB.SG", "STACKEDHOMES", "NEWLAUNCH.SG",
                           "SGCONDO.COM", "FLOORPLANS.SG"]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = float(rng.uniform(0.7, 1.3))
    step_x, step_y = int(rng.integers(220, 380)), int(rng.integers(120, 220))
    canvas = np.zeros((h + 200, w + 200, 3), dtype=np.uint8)
    for y in range(0, h + 200, step_y):
        off = (y // step_y % 2) * step_x // 2
        for x in range(-off, w + 200, step_x):
            cv2.putText(canvas, text, (x, y), font, fs, (255, 255, 255), 2,
                        cv2.LINE_AA)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-40, -20)), 1.0)
    rotated = cv2.warpAffine(canvas[:h, :w], m, (w, h))
    alpha = float(rng.uniform(0.06, 0.18))
    mask = rotated.astype(np.float32) / 255.0 * alpha
    tint = np.array([128, 128, 128], dtype=np.float32)
    return np.clip(img.astype(np.float32) * (1 - mask) + tint * mask, 0,
                   255).astype(np.uint8)


def _vignette(img, rng):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    d = np.sqrt(((x - cx) / w) ** 2 + ((y - cy) / h) ** 2)
    strength = rng.uniform(0.1, 0.35)
    factor = 1.0 - strength * np.clip(d - 0.3, 0, None) * 2
    return np.clip(img.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)


def _augraphy_op(cls_name: str, **kw):
    def run(img, rng):
        import augraphy
        # augraphy uses global RNGs — pin them from our stream (determinism)
        seed = int(rng.integers(0, 2**31 - 1))
        random.seed(seed)
        np.random.seed(seed)
        out = getattr(augraphy, cls_name)(p=1.0, **kw)(img)
        return _to3(out) if out is not None else img
    return run


# name -> callable(img, rng) -> img; ANY new op must be added here and pass
# the registration-alignment test — that's what keeps the whitelist honest
SAFE_OPS = {
    "gaussian_blur": _gaussian_blur,
    "motion_blur": _motion_blur,
    "salt_pepper": _salt_pepper,
    "line_breakup": _line_breakup,
    "jpeg": _jpeg,
    "down_up": _down_up,
    "watermark": _watermark_tiled,
    "vignette": _vignette,
    "badphotocopy": _augraphy_op("BadPhotoCopy"),
    "inkbleed": _augraphy_op("InkBleed"),
    "bleedthrough": _augraphy_op("BleedThrough"),
    "lowink_lines": _augraphy_op("LowInkRandomLines"),
    "subtle_noise": _augraphy_op("SubtleNoise"),
    "noise_texturize": _augraphy_op("NoiseTexturize"),
    "dirty_rollers": _augraphy_op("DirtyRollers"),
    "lighting_gradient": _augraphy_op("LightingGradient"),
    "dithering": _augraphy_op("Dithering"),
    "shadow_cast": _augraphy_op("ShadowCast"),
    "stains": _augraphy_op("Stains"),
}

# condition groups sampled by level
_GROUPS = {
    "scan": ["gaussian_blur", "line_breakup", "salt_pepper", "badphotocopy",
             "lowink_lines", "lighting_gradient", "dirty_rollers", "inkbleed",
             "bleedthrough", "dithering"],
    "photo": ["motion_blur", "vignette", "shadow_cast", "noise_texturize",
              "stains"],
    "web": ["jpeg", "down_up", "watermark", "subtle_noise"],
}

LEVELS = {"none": 0, "light": 1, "medium": 2, "heavy": 4}
LEVEL_PROBS = {"none": 0.3, "light": 0.3, "medium": 0.25, "heavy": 0.15}


def apply(img: np.ndarray, rng: np.random.Generator,
          bg: tuple[int, int, int],
          level: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Full degradation. Returns (image, G 3x3, op names applied)."""
    if level is None:
        names = list(LEVEL_PROBS)
        level = str(rng.choice(names, p=[LEVEL_PROBS[n] for n in names]))
    ops_applied: list[str] = []

    if level == "none":
        return img, np.eye(3), ops_applied

    out, g = sample_geometric(img, rng, bg)
    ops_applied.append(f"geometric(level={level})")

    n_ops = LEVELS[level] + int(rng.integers(0, 2))
    group = str(rng.choice(["scan", "photo", "web"], p=[0.5, 0.2, 0.3]))
    pool = list(_GROUPS[group])
    if level == "heavy":  # heavy samples mix conditions
        pool += _GROUPS["web"]
    for _ in range(n_ops):
        name = str(rng.choice(pool))
        try:
            out = SAFE_OPS[name](out, rng)
            ops_applied.append(name)
        except Exception:
            continue  # an op failing must never kill the sample
    # always end web-ish samples with jpeg re-encode (corpus reality)
    if group == "web" and "jpeg" not in ops_applied:
        out = _jpeg(out, rng)
        ops_applied.append("jpeg")
    return out, g, ops_applied
