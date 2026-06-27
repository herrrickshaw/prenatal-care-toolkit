"""Problem 4 - Standard fetal biometry and gestational-age estimation.

Educational, sex-neutral obstetric calculations from the standard biometric
measurements a sonographer records:

    BPD - biparietal diameter (mm)
    HC  - head circumference (mm)
    AC  - abdominal circumference (mm)
    FL  - femur length (mm)

Provides gestational-age estimates (Hadlock regressions) and estimated fetal
weight (Hadlock 1985). None of these reveal or depend on fetal sex; they are
the legitimate, everyday output of an antenatal scan.
"""

from .gestation import (
    estimated_fetal_weight_hadlock,
    gestational_age_weeks,
    ga_from_measurements,
)

__all__ = [
    "estimated_fetal_weight_hadlock",
    "gestational_age_weeks",
    "ga_from_measurements",
]
