"""Fetal ultrasound *plane* classifier - a sex-neutral anatomy model.

Classifies an ultrasound frame into a standard **anatomical acquisition plane**
(fetal brain / abdomen / femur / thorax, maternal cervix, or other), the task
defined by the public FETAL_PLANES_DB dataset. This is the everyday first step
of an antenatal scan and it feeds two toolkit goals:

  * **Problem 1 (de-id)** - knowing which anatomical plane a frame shows makes
    burned-in-text redaction anatomy-aware (never blur the diagnostic region).
  * **Problem 4 (biometry)** - plane recognition is the precursor to measuring
    BPD / HC / AC / FL.

It does **not** classify or infer fetal sex. Plane labels are anatomical
regions, consistent with this toolkit's refusal to build a sex detector.

Implementation is a dependency-light scikit-learn pipeline (numpy HOG-style
features -> RandomForest) so it trains and evaluates on CPU with no deep-learning
stack. See ``README`` for the optional CNN extension point.
"""

from .dataset import (
    PLANE_CLASSES,
    load_fetal_planes_db,
    make_synthetic_planes,
)
from .features import extract_features, features_for_paths
from .model import PlaneClassifier

__all__ = [
    "PLANE_CLASSES",
    "load_fetal_planes_db",
    "make_synthetic_planes",
    "extract_features",
    "features_for_paths",
    "PlaneClassifier",
]
