"""Sex-ratio-at-birth statistics on synthetic counts."""
import pandas as pd
import pytest

from pctk.srb.analyze import SRBConfig, _binom_sf, compute_srb, flag_anomalies, from_aggregated_counts


def test_binomial_lower_tail_exact_small_n():
    assert _binom_sf(0, 10, 0.5) == pytest.approx(1 / 1024)
    assert _binom_sf(10, 10, 0.5) == pytest.approx(1.0)
    assert _binom_sf(5, 10, 0.5) == pytest.approx(638 / 1024)
    assert _binom_sf(0, 0, 0.5) == 1.0


def test_binomial_normal_approximation_close_to_exact_at_switch():
    exact = _binom_sf(470, 1000, 0.488)                 # exact branch
    approx = _binom_sf(470 * 1001 // 1000, 1001, 0.488)  # normal branch, same proportion
    assert approx == pytest.approx(exact, abs=0.02)


def test_compute_srb_token_normalisation_and_ratios():
    df = pd.DataFrame({"district": ["X"] * 6 + ["Y"] * 2,
                       "sex": ["M", "male", "1", "F", " female ", "unknown", "boy", "girl"]})
    tab = compute_srb(df).set_index("district")
    assert (tab.loc["X", "males"], tab.loc["X", "females"], tab.loc["X", "total"]) == (3, 2, 5)
    assert tab.loc["X", "srb_f_per_1000_m"] == pytest.approx(2 / 3 * 1000)
    assert tab.loc["Y", "pct_female"] == pytest.approx(0.5)


def test_flag_anomalies_on_synthetic_counts():
    df = pd.DataFrame({"district": ["Skewed", "Natural", "Tiny"],
                       "m": [600, 512, 10], "f": [400, 488, 2]})
    tab = from_aggregated_counts(df, ["district"], "m", "f")
    out = flag_anomalies(tab, SRBConfig(group_cols=["district"]))
    assert list(out["district"]) == ["Skewed", "Natural"]            # Tiny < min_births dropped
    row = out.set_index("district").loc["Skewed"]
    assert bool(row["significant"]) and row["expected_females"] == pytest.approx(488.0)
    assert row["female_deficit"] == pytest.approx(88.0)
    assert not bool(out.set_index("district").loc["Natural", "significant"])
