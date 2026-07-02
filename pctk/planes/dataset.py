"""Load FETAL_PLANES_DB, or generate a synthetic stand-in with the same layout.

FETAL_PLANES_DB (Burgos-Artizzu et al., 2020; Zenodo record 3904280) ships:
  * ``Images/`` - PNG frames,
  * ``FETAL_PLANES_DB_data.csv`` - ``;``-separated, with columns including
    ``Image_name`` (no extension), ``Plane``, and ``Train`` (1/0).

``load_fetal_planes_db`` reads that into a tidy frame of
``(image_path, label, split)``. ``make_synthetic_planes`` writes the *same*
layout with class-distinct synthetic images so the whole pipeline is runnable
and testable without downloading the real (multi-GB) dataset.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Canonical FETAL_PLANES_DB plane labels.
PLANE_CLASSES: List[str] = [
    "Fetal brain",
    "Fetal abdomen",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]


def _find_csv(root: str) -> Optional[str]:
    for name in ("FETAL_PLANES_DB_data.csv", "FETAL_PLANES_DB.csv"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(root, "*.csv"))
    return hits[0] if hits else None


def _resolve_col(cols, *cands):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for c in cands:
        for lc, orig in low.items():
            if c.lower() in lc:
                return orig
    return None


def load_fetal_planes_db(root: str,
                         images_dirname: str = "Images") -> pd.DataFrame:
    """Return a DataFrame with columns ``image_path``, ``label``, ``split``."""
    csv = _find_csv(root)
    if not csv:
        raise FileNotFoundError(f"no metadata CSV found under {root!r}")
    # The real file is ';'-separated; fall back to sniffing.
    try:
        df = pd.read_csv(csv, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(csv)
    except Exception:
        df = pd.read_csv(csv)

    name_col = _resolve_col(df.columns, "Image_name", "image", "filename")
    plane_col = _resolve_col(df.columns, "Plane", "label", "class")
    train_col = _resolve_col(df.columns, "Train", "split")
    if not name_col or not plane_col:
        raise ValueError(f"could not find image/plane columns in {csv!r}")

    img_dir = os.path.join(root, images_dirname)

    def _path(name: str) -> str:
        name = str(name)
        cand = os.path.join(img_dir, name)
        if os.path.exists(cand):
            return cand
        for ext in (".png", ".jpg", ".jpeg"):
            if os.path.exists(cand + ext):
                return cand + ext
        return cand + ".png"

    out = pd.DataFrame({
        "image_path": df[name_col].map(_path),
        "label": df[plane_col].astype(str).str.strip(),
    })
    if train_col:
        out["split"] = np.where(
            df[train_col].astype(str).str.strip().isin(["1", "1.0", "train",
                                                        "Train", "TRUE", "True"]),
            "train", "test")
    else:
        out["split"] = "train"
    return out


# --------------------------------------------------------------------------- #
# Synthetic stand-in (class-distinct patterns) for demo / tests.
# --------------------------------------------------------------------------- #
def _draw_plane(label: str, size: int, rng: np.random.Generator) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    c = size // 2
    # speckle background
    bg = (rng.random((size, size)) * 30).astype(np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    jitter = lambda a: int(a + rng.integers(-size // 20, size // 20 + 1))
    if label == "Fetal brain":            # concentric skull rings
        for r in range(size // 3, size // 8, -max(2, size // 20)):
            d.ellipse([c - r, jitter(c) - r, c + r, jitter(c) + r], outline=200)
    elif label == "Fetal abdomen":        # filled ellipse (stomach bubble)
        r = size // 3
        d.ellipse([c - r, c - r, c + r, c + r], outline=180, fill=90)
        d.ellipse([c - 6, c - 6, c + 6, c + 6], fill=230)
    elif label == "Fetal femur":          # bright diagonal bone
        d.line([jitter(size // 6), jitter(size // 4),
                jitter(5 * size // 6), jitter(3 * size // 4)],
               fill=240, width=max(3, size // 16))
    elif label == "Fetal thorax":         # ribs + 4-chamber heart
        for k in range(-2, 3):
            d.arc([c - size // 3, c - size // 3, c + size // 3, c + size // 3],
                  200 + k * 8, 340 + k * 8, fill=170)
        d.ellipse([c - 10, c - 10, c + 10, c + 10], fill=210)
    elif label == "Maternal cervix":      # vertical bright band
        d.rectangle([jitter(c - size // 10), 4, jitter(c + size // 10), size - 4],
                    fill=150)
    else:                                  # Other: random blobs
        for _ in range(4):
            x, y = rng.integers(0, size, 2)
            r = rng.integers(4, size // 6)
            d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.integers(80, 200)))
    arr = np.asarray(img).astype(np.int16)
    arr = np.clip(arr + (rng.random(arr.shape) * 25).astype(np.int16), 0, 255)
    return arr.astype(np.uint8)


def make_synthetic_planes(dest: str,
                          n_per_class: int = 40,
                          size: int = 96,
                          seed: int = 0) -> str:
    """Write a synthetic FETAL_PLANES_DB-shaped dataset under ``dest``.

    Produces ``dest/Images/*.png`` and ``dest/FETAL_PLANES_DB_data.csv``.
    """
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(dest, "Images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for label in PLANE_CLASSES:
        for i in range(n_per_class):
            arr = _draw_plane(label, size, rng)
            name = f"{label.replace(' ', '_')}_{i:03d}"
            Image.fromarray(arr).save(os.path.join(img_dir, name + ".png"))
            # ~75% train / 25% test
            train = 1 if (i % 4 != 0) else 0
            rows.append({"Image_name": name, "Plane": label, "Train": train})
    csv = os.path.join(dest, "FETAL_PLANES_DB_data.csv")
    pd.DataFrame(rows).to_csv(csv, sep=";", index=False)
    return dest
