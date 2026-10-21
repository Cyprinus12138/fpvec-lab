"""Render + degrade properties: the pixel-alignment guard.

test_registration_alignment is the contract that keeps degrade.SAFE_OPS
honest: registration crosses drawn at known pixel positions must land within
1.5 px of their G-mapped positions after the FULL degradation pipeline, for
every level, across seeds. Any pixel-displacing op added to the whitelist
breaks this test.
"""

import cv2
import numpy as np
import pytest

from fpgen import degrade

CROSSES = [(60, 60), (240, 70), (70, 350), (250, 340)]


def _make_target(w=320, h=420):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for cx, cy in CROSSES:
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (0, 0, 0), 3)
        cv2.line(img, (cx, cy - 12), (cx, cy + 12), (0, 0, 0), 3)
    return img


def _cross_template():
    t = np.full((31, 31), 255, dtype=np.uint8)
    cv2.line(t, (3, 15), (27, 15), 0, 3)
    cv2.line(t, (15, 3), (15, 27), 0, 3)
    return t


TEMPLATE = _cross_template()


def _find_cross(img_gray, near_xy, window=30):
    """NCC template match — robust to noise/shadows that bias centroids."""
    x, y = int(near_xy[0]), int(near_xy[1])
    x0, y0 = max(0, x - window), max(0, y - window)
    patch = img_gray[y0:y + window, x0:x + window]
    th, tw = TEMPLATE.shape
    if patch.shape[0] <= th or patch.shape[1] <= tw:
        return None
    res = cv2.matchTemplate(patch, TEMPLATE, cv2.TM_CCOEFF_NORMED)
    _, conf, _, loc = cv2.minMaxLoc(res)
    if conf < 0.35:
        return None
    return (x0 + loc[0] + tw / 2, y0 + loc[1] + th / 2)


@pytest.mark.parametrize("level", ["light", "medium", "heavy"])
def test_registration_alignment(level):
    worst = 0.0
    checked = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        img = _make_target()
        out, g, ops = degrade.apply(img, rng, (255, 255, 255), level=level)
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        for cx, cy in CROSSES:
            px, py, w = g @ (cx, cy, 1.0)
            found = _find_cross(gray, (px / w, py / w))
            if found is None:
                continue  # heavy noise can drown a cross; skip, don't fail
            err = np.hypot(found[0] - px / w, found[1] - py / w)
            worst = max(worst, err)
            checked += 1
    assert checked >= 24, "too few crosses recovered to trust the check"
    assert worst < 1.5, f"alignment drift {worst:.2f}px at level={level}"


def test_degrade_none_is_identity():
    rng = np.random.default_rng(0)
    img = _make_target()
    out, g, ops = degrade.apply(img, rng, (255, 255, 255), level="none")
    assert np.array_equal(out, img)
    assert np.array_equal(g, np.eye(3))
    assert ops == []


def test_degrade_deterministic():
    img = _make_target()
    a = degrade.apply(img, np.random.default_rng(42), (255, 255, 255), level="heavy")
    b = degrade.apply(img, np.random.default_rng(42), (255, 255, 255), level="heavy")
    assert np.array_equal(a[0], b[0])
    assert a[2] == b[2]


def test_all_safe_ops_preserve_shape():
    img = _make_target()
    for name, op in degrade.SAFE_OPS.items():
        out = op(img.copy(), np.random.default_rng(1))
        assert out.shape[:2] == img.shape[:2], name
        assert out.ndim == 3, name
