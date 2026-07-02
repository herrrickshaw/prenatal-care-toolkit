"""Scikit-learn fetal-plane classifier: features -> RandomForest.

Trains on a DataFrame of ``(image_path, label, split)`` (from
``dataset.load_fetal_planes_db`` or ``make_synthetic_planes``), evaluates on the
held-out split, and persists to disk with joblib. Reports accuracy, a
per-class classification report, and the confusion matrix.

No fetal-sex target exists anywhere in this model - the labels are anatomical
acquisition planes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features import extract_features, features_for_paths


@dataclass
class TrainConfig:
    size: int = 64
    cell: int = 8
    bins: int = 9
    n_estimators: int = 300
    max_depth: Optional[int] = None
    random_state: int = 0
    feat_kwargs: Dict = field(default_factory=dict)


class PlaneClassifier:
    """Thin wrapper around a scikit-learn pipeline for plane classification."""

    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
        self.pipeline = None
        self.classes_: List[str] = []

    # -- build ------------------------------------------------------------- #
    def _new_pipeline(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )),
        ])

    def _feat(self, paths):
        c = self.config
        return features_for_paths(list(paths), size=c.size, cell=c.cell,
                                  bins=c.bins, **c.feat_kwargs)

    # -- train / eval ------------------------------------------------------ #
    def train(self, df: pd.DataFrame, split: str = "train") -> Dict:
        sub = df[df["split"] == split] if "split" in df.columns else df
        if sub.empty:
            raise ValueError(f"no rows for split={split!r}")
        X = self._feat(sub["image_path"])
        y = sub["label"].to_numpy()
        self.pipeline = self._new_pipeline()
        self.pipeline.fit(X, y)
        self.classes_ = list(self.pipeline.classes_)
        return {"n_train": len(sub), "classes": self.classes_,
                "n_features": X.shape[1]}

    def evaluate(self, df: pd.DataFrame, split: str = "test") -> Dict:
        if self.pipeline is None:
            raise RuntimeError("train() or load() first")
        from sklearn.metrics import (accuracy_score, classification_report,
                                     confusion_matrix)
        sub = df[df["split"] == split] if "split" in df.columns else df
        if sub.empty:
            raise ValueError(f"no rows for split={split!r}")
        X = self._feat(sub["image_path"])
        y = sub["label"].to_numpy()
        pred = self.pipeline.predict(X)
        return {
            "n_test": len(sub),
            "accuracy": float(accuracy_score(y, pred)),
            "report": classification_report(y, pred, zero_division=0),
            "confusion_matrix": confusion_matrix(
                y, pred, labels=self.classes_).tolist(),
            "labels": self.classes_,
        }

    # -- inference --------------------------------------------------------- #
    def predict(self, image) -> Dict:
        if self.pipeline is None:
            raise RuntimeError("train() or load() first")
        c = self.config
        x = extract_features(image, size=c.size, cell=c.cell,
                             bins=c.bins, **c.feat_kwargs).reshape(1, -1)
        label = self.pipeline.predict(x)[0]
        out = {"label": str(label)}
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(x)[0]
            out["proba"] = {cls: float(p)
                            for cls, p in zip(self.classes_, proba)}
            out["confidence"] = float(np.max(proba))
        return out

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"pipeline": self.pipeline, "config": self.config,
                     "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: str) -> "PlaneClassifier":
        import joblib
        blob = joblib.load(path)
        obj = cls(blob.get("config"))
        obj.pipeline = blob["pipeline"]
        obj.classes_ = blob.get("classes", list(getattr(obj.pipeline,
                                                        "classes_", [])))
        return obj
