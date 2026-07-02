"""Scikit-learn fetal-health (CTG) classifier.

StandardScaler -> GradientBoosting (with a RandomForest fallback). Trains on the
21 CTG features, reports accuracy / macro-F1 / per-class report / confusion
matrix on a stratified hold-out, and exposes feature importances. Persists with
joblib. No fetal-sex target anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .dataset import TARGET_NAMES


@dataclass
class HealthConfig:
    test_size: float = 0.25
    random_state: int = 0
    model: str = "gb"          # "gb" (GradientBoosting) | "rf" (RandomForest)
    feature_names: List[str] = field(default_factory=list)


class FetalHealthClassifier:
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.pipeline = None
        self.classes_: List[int] = []

    def _new_pipeline(self):
        from sklearn.ensemble import (GradientBoostingClassifier,
                                      RandomForestClassifier)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.config.model == "rf":
            clf = RandomForestClassifier(
                n_estimators=400, random_state=self.config.random_state,
                n_jobs=-1, class_weight="balanced_subsample")
        else:
            clf = GradientBoostingClassifier(
                random_state=self.config.random_state)
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])

    def fit_eval(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train on a stratified split and return hold-out metrics."""
        from sklearn.metrics import (accuracy_score, classification_report,
                                     confusion_matrix, f1_score)
        from sklearn.model_selection import train_test_split

        self.config.feature_names = list(X.columns)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=y)
        self.pipeline = self._new_pipeline()
        self.pipeline.fit(Xtr, ytr)
        self.classes_ = [int(c) for c in self.pipeline.classes_]
        pred = self.pipeline.predict(Xte)
        target_names = [TARGET_NAMES.get(c, str(c)) for c in self.classes_]
        return {
            "n_train": len(Xtr), "n_test": len(Xte),
            "accuracy": float(accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
            "report": classification_report(
                yte, pred, target_names=target_names, zero_division=0),
            "confusion_matrix": confusion_matrix(
                yte, pred, labels=self.classes_).tolist(),
            "labels": self.classes_,
        }

    def feature_importances(self, top: int = 10) -> List:
        if self.pipeline is None:
            raise RuntimeError("fit_eval() or load() first")
        clf = self.pipeline.named_steps["clf"]
        imp = getattr(clf, "feature_importances_", None)
        if imp is None:
            return []
        names = self.config.feature_names
        order = np.argsort(imp)[::-1][:top]
        return [(names[i], float(imp[i])) for i in order]

    def predict(self, row) -> Dict:
        if self.pipeline is None:
            raise RuntimeError("fit_eval() or load() first")
        if isinstance(row, dict):
            row = pd.DataFrame([row])[self.config.feature_names]
        elif isinstance(row, pd.Series):
            row = row.to_frame().T[self.config.feature_names]
        label = int(self.pipeline.predict(row)[0])
        out = {"label": label, "status": TARGET_NAMES.get(label, str(label))}
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(row)[0]
            out["proba"] = {TARGET_NAMES.get(int(c), str(c)): float(p)
                            for c, p in zip(self.classes_, proba)}
        return out

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"pipeline": self.pipeline, "config": self.config,
                     "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: str) -> "FetalHealthClassifier":
        import joblib
        blob = joblib.load(path)
        obj = cls(blob.get("config"))
        obj.pipeline = blob["pipeline"]
        obj.classes_ = blob.get("classes", [])
        return obj
