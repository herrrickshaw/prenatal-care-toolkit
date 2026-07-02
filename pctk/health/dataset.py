"""Load the Fetal Health (CTG) dataset, or synthesise a schema-matched stand-in.

The real dataset (Kaggle: andrewmvd/fetal-health-classification) is a single CSV
with 21 CTG feature columns and a ``fetal_health`` target coded 1.0 (normal),
2.0 (suspect), 3.0 (pathological). ``load_fetal_health`` splits it into
``(X, y)``; ``make_synthetic_health`` writes a CSV with the same columns and a
learnable target so the pipeline runs and tests without a download.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

TARGET_COL = "fetal_health"
TARGET_NAMES = {1: "normal", 2: "suspect", 3: "pathological"}

# The 21 SisPorto CTG features, in the dataset's order.
FEATURE_HINT: List[str] = [
    "baseline value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
]


def load_fetal_health(path: str,
                      target_col: str = TARGET_COL
                      ) -> Tuple[pd.DataFrame, pd.Series]:
    """Read the CTG CSV and return ``(X features, y target as int)``."""
    df = pd.read_csv(path)
    tcol = target_col if target_col in df.columns else None
    if tcol is None:
        for c in df.columns:
            if "fetal_health" in c.lower() or c.lower() in ("target", "label"):
                tcol = c
                break
    if tcol is None:
        raise ValueError("could not find the fetal_health target column")
    y = df[tcol].round().astype(int)
    X = df.drop(columns=[tcol])
    return X, y


def make_synthetic_health(path: str,
                          n: int = 1200,
                          seed: int = 0) -> str:
    """Write a schema-matched synthetic CTG CSV with a learnable target.

    The target is driven by a few clinically-plausible signals (decelerations
    and abnormal variability push toward suspect/pathological) plus noise, so a
    classifier does clearly better than the majority baseline.
    """
    rng = np.random.default_rng(seed)
    cols = {}
    cols["baseline value"] = rng.normal(133, 10, n)
    cols["accelerations"] = np.clip(rng.normal(0.003, 0.004, n), 0, None)
    cols["fetal_movement"] = np.clip(rng.normal(0.01, 0.05, n), 0, None)
    cols["uterine_contractions"] = np.clip(rng.normal(0.004, 0.003, n), 0, None)
    cols["light_decelerations"] = np.clip(rng.normal(0.002, 0.003, n), 0, None)
    cols["severe_decelerations"] = np.clip(rng.normal(0.0, 0.001, n), 0, None)
    cols["prolongued_decelerations"] = np.clip(rng.normal(0.0005, 0.001, n), 0, None)
    cols["abnormal_short_term_variability"] = np.clip(rng.normal(47, 17, n), 0, 100)
    cols["mean_value_of_short_term_variability"] = np.clip(rng.normal(1.3, 0.9, n), 0, None)
    cols["percentage_of_time_with_abnormal_long_term_variability"] = \
        np.clip(rng.normal(10, 18, n), 0, 100)
    cols["mean_value_of_long_term_variability"] = np.clip(rng.normal(8, 5, n), 0, None)
    cols["histogram_width"] = rng.normal(70, 39, n)
    cols["histogram_min"] = rng.normal(93, 29, n)
    cols["histogram_max"] = rng.normal(164, 17, n)
    cols["histogram_number_of_peaks"] = np.clip(rng.normal(4, 3, n), 0, None)
    cols["histogram_number_of_zeroes"] = np.clip(rng.normal(0.3, 0.7, n), 0, None)
    cols["histogram_mode"] = rng.normal(137, 16, n)
    cols["histogram_mean"] = rng.normal(134, 16, n)
    cols["histogram_median"] = rng.normal(138, 14, n)
    cols["histogram_variance"] = np.clip(rng.normal(18, 29, n), 0, None)
    cols["histogram_tendency"] = rng.integers(-1, 2, n).astype(float)

    df = pd.DataFrame(cols)[FEATURE_HINT]

    # Risk score -> class. Decelerations & abnormal variability raise risk;
    # accelerations lower it.
    risk = (
        8000 * df["prolongued_decelerations"]
        + 3000 * df["severe_decelerations"]
        + 0.03 * df["abnormal_short_term_variability"]
        + 0.02 * df["percentage_of_time_with_abnormal_long_term_variability"]
        - 60 * df["accelerations"]
        + rng.normal(0, 0.6, n)
    )
    y = np.ones(n, dtype=int)
    y[risk > np.quantile(risk, 0.78)] = 2      # suspect
    y[risk > np.quantile(risk, 0.92)] = 3      # pathological
    df[TARGET_COL] = y.astype(float)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path
