"""Fetal-health (cardiotocography) classifier - a sex-neutral wellbeing model.

Classifies a cardiotocogram (CTG) feature record into fetal-health status
(normal / suspect / pathological), the task defined by the public
"Fetal Health Classification" dataset (Ayres-de-Campos et al. SisPorto features;
Kaggle: andrewmvd/fetal-health-classification).

CTG measures fetal heart rate and uterine contractions to flag fetuses at risk -
it is entirely about *wellbeing*, not sex. Like every other module here, there
is no fetal-sex target.

Dependency-light scikit-learn pipeline; trains, evaluates, and predicts on CPU.
"""

from .dataset import (
    FEATURE_HINT,
    TARGET_NAMES,
    load_fetal_health,
    make_synthetic_health,
)
from .model import FetalHealthClassifier

__all__ = [
    "FEATURE_HINT",
    "TARGET_NAMES",
    "load_fetal_health",
    "make_synthetic_health",
    "FetalHealthClassifier",
]
