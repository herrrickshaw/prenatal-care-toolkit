"""Hadlock biometry against hand-computed values.

Formulas: Hadlock et al. 1984 (GA) and 1985 (EFW), measures in cm.
Educational sanity checks only - not a clinical reference.
"""
import math

import pytest

from pctk.biometry.gestation import (estimated_fetal_weight_hadlock, ga_from_measurements,
                                     gestational_age_weeks)


@pytest.mark.parametrize("m,mm,expected", [
    ("bpd", 50.0, 9.54 + 1.482 * 5 + 0.1676 * 25),        # 21.14
    ("hc", 200.0, 8.96 + 0.540 * 20 + 0.0003 * 8000),      # 22.16
    ("ac", 200.0, 8.14 + 0.753 * 20 + 0.0036 * 400),       # 24.64
    ("fl", 30.0, 10.35 + 2.460 * 3 + 0.170 * 9),           # 19.26
])
def test_single_parameter_ga(m, mm, expected):
    assert gestational_age_weeks(m, mm) == pytest.approx(round(expected, 2))


def test_bad_measurement_rejected():
    with pytest.raises(ValueError):
        gestational_age_weeks("crl", 10)
    with pytest.raises(ValueError):
        gestational_age_weeks("bpd", 0)
    with pytest.raises(ValueError):
        ga_from_measurements()


def test_composite_ga_is_mean_and_rounds_days():
    res = ga_from_measurements(bpd_mm=50.0, fl_mm=30.0)
    assert res["ga_weeks"] == pytest.approx((21.14 + 19.26) / 2)       # 20.20
    assert res["ga_str"] == "20w 1d"                                     # 0.2 * 7 = 1.4 -> 1


def test_ga_str_never_shows_seven_days():
    # choose BPD so GA = 20.95 w: 0.95 * 7 = 6.65 -> rounds to 7 -> must roll to 21w 0d
    b = (-1.482 + math.sqrt(1.482 ** 2 - 4 * 0.1676 * (9.54 - 20.95))) / (2 * 0.1676)
    res = ga_from_measurements(bpd_mm=b * 10)
    assert res["ga_str"] == "21w 0d"


def test_efw_ac_fl_formula():
    # log10 W = 1.304 + 0.05281*30 + 0.1938*6 - 0.004*30*6 = 3.33110
    assert estimated_fetal_weight_hadlock(300.0, 60.0) == pytest.approx(round(10 ** 3.3311, 1), rel=1e-6)


def test_efw_four_parameter_formula():
    ac, fl, bpd, hc = 30.0, 6.0, 8.0, 29.0
    log10w = (1.3596 - 0.00386 * ac * fl + 0.0064 * hc + 0.00061 * bpd * ac + 0.0424 * ac + 0.174 * fl)
    assert estimated_fetal_weight_hadlock(300.0, 60.0, 80.0, 290.0) == pytest.approx(round(10 ** log10w, 1))
