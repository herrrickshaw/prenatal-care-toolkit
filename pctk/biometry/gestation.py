"""Gestational-age and fetal-weight estimation from biometry.

Formulas are the widely published Hadlock regressions. They are approximations
intended for *education and sanity-checking*, not a substitute for a clinical
report. Measurements are in millimetres unless noted; gestational age in weeks.
"""

from __future__ import annotations

import math
from typing import Dict, Optional


def _ga_bpd(bpd_mm: float) -> float:
    """GA (weeks) from biparietal diameter. Hadlock-style polynomial."""
    b = bpd_mm / 10.0  # to cm
    return 9.54 + 1.482 * b + 0.1676 * b * b


def _ga_hc(hc_mm: float) -> float:
    """GA (weeks) from head circumference."""
    h = hc_mm / 10.0  # cm
    return 8.96 + 0.540 * h + 0.0003 * h ** 3


def _ga_ac(ac_mm: float) -> float:
    """GA (weeks) from abdominal circumference."""
    a = ac_mm / 10.0  # cm
    return 8.14 + 0.753 * a + 0.0036 * a * a


def _ga_fl(fl_mm: float) -> float:
    """GA (weeks) from femur length."""
    f = fl_mm / 10.0  # cm
    return 10.35 + 2.460 * f + 0.170 * f * f


_GA_FUNCS = {
    "bpd": _ga_bpd,
    "hc": _ga_hc,
    "ac": _ga_ac,
    "fl": _ga_fl,
}


def gestational_age_weeks(measure: str, value_mm: float) -> float:
    """GA from a single measurement. ``measure`` in {bpd, hc, ac, fl}."""
    key = measure.lower()
    if key not in _GA_FUNCS:
        raise ValueError(f"unknown measurement {measure!r}; "
                         f"use one of {sorted(_GA_FUNCS)}")
    if value_mm <= 0:
        raise ValueError("measurement must be positive (mm)")
    return round(_GA_FUNCS[key](value_mm), 2)


def ga_from_measurements(
    bpd_mm: Optional[float] = None,
    hc_mm: Optional[float] = None,
    ac_mm: Optional[float] = None,
    fl_mm: Optional[float] = None,
) -> Dict[str, object]:
    """Composite GA: average of the available single-parameter estimates.

    Returns ``{"per_measure": {...}, "ga_weeks": float, "ga_str": "Nw Md"}``.
    """
    per: Dict[str, float] = {}
    for name, val in (("bpd", bpd_mm), ("hc", hc_mm), ("ac", ac_mm), ("fl", fl_mm)):
        if val is not None:
            per[name] = gestational_age_weeks(name, val)
    if not per:
        raise ValueError("provide at least one of bpd/hc/ac/fl")
    ga = sum(per.values()) / len(per)
    weeks = int(ga)
    days = int(round((ga - weeks) * 7))
    if days == 7:
        weeks, days = weeks + 1, 0
    return {
        "per_measure": per,
        "ga_weeks": round(ga, 2),
        "ga_str": f"{weeks}w {days}d",
    }


def estimated_fetal_weight_hadlock(
    ac_mm: float,
    fl_mm: float,
    bpd_mm: Optional[float] = None,
    hc_mm: Optional[float] = None,
) -> float:
    """Estimated fetal weight (grams), Hadlock 1985.

    Picks the most complete published formula given which of BPD/HC are
    supplied (all measures in cm inside the regression).
    """
    ac = ac_mm / 10.0
    fl = fl_mm / 10.0
    bpd = (bpd_mm or 0) / 10.0
    hc = (hc_mm or 0) / 10.0

    if bpd_mm and hc_mm:
        log10w = (1.3596 - 0.00386 * ac * fl + 0.0064 * hc + 0.00061 * bpd * ac
                  + 0.0424 * ac + 0.174 * fl)
    elif hc_mm:
        log10w = (1.326 - 0.00326 * ac * fl + 0.0107 * hc + 0.0438 * ac
                  + 0.158 * fl)
    elif bpd_mm:
        log10w = (1.335 - 0.0034 * ac * fl + 0.0316 * bpd + 0.0457 * ac
                  + 0.1623 * fl)
    else:
        # AC + FL only
        log10w = 1.304 + 0.05281 * ac + 0.1938 * fl - 0.004 * ac * fl
    return round(10 ** log10w, 1)
