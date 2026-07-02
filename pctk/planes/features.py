"""Image -> feature vector for the plane classifier (numpy only).

A compact, dependency-free descriptor combining:
  * a HOG-style histogram of oriented gradients over a cell grid, and
  * a coarse downsampled intensity map,
  * plus a few global intensity statistics.

Good enough for a solid scikit-learn baseline without OpenCV / skimage / torch.
"""

from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


def _load_gray(img, size: int) -> np.ndarray:
    if isinstance(img, str):
        pil = Image.open(img).convert("L")
    elif isinstance(img, np.ndarray):
        a = img
        if a.ndim == 3:
            a = a.mean(axis=2)
        pil = Image.fromarray(a.astype(np.uint8)).convert("L")
    else:
        pil = img.convert("L")
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float64) / 255.0


def _hog(gray: np.ndarray, cell: int, bins: int) -> np.ndarray:
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ori = (np.arctan2(gy, gx) % np.pi)            # 0..pi (unsigned)
    n = gray.shape[0] // cell
    feat = np.zeros((n, n, bins), dtype=np.float64)
    bin_w = np.pi / bins
    idx = np.minimum((ori / bin_w).astype(int), bins - 1)
    for i in range(n):
        for j in range(n):
            r0, c0 = i * cell, j * cell
            m = mag[r0:r0 + cell, c0:c0 + cell].ravel()
            b = idx[r0:r0 + cell, c0:c0 + cell].ravel()
            feat[i, j] = np.bincount(b, weights=m, minlength=bins)
    # L2 block-normalise each cell histogram
    feat = feat.reshape(-1, bins)
    norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-6
    feat = feat / norm
    return feat.ravel()


def extract_features(img, size: int = 64, cell: int = 8, bins: int = 9,
                     coarse: int = 8) -> np.ndarray:
    """Return a 1-D feature vector for one image (path / array / PIL)."""
    gray = _load_gray(img, size)
    hog = _hog(gray, cell=cell, bins=bins)
    small = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).resize(
            (coarse, coarse), Image.BILINEAR), dtype=np.float64).ravel() / 255.0
    stats = np.array([gray.mean(), gray.std(),
                      float((gray > 0.5).mean())])
    return np.concatenate([hog, small, stats])


def features_for_paths(paths: List[str], **kw) -> np.ndarray:
    """Stack feature vectors for many images into an (N, D) matrix."""
    return np.vstack([extract_features(p, **kw) for p in paths])
