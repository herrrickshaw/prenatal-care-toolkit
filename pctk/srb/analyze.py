"""Sex-ratio-at-birth computation and anomaly flagging.

Schema expected (column names configurable via :class:`SRBConfig`):

    sex        : "M"/"F" (or 1/2, or "male"/"female")
    <group...> : one or more dimension columns (district, clinic, year, ...)
    count      : optional; if present, rows are pre-aggregated counts

Conventions
-----------
* SRB is reported two ways:
    - ``srb_f_per_1000_m`` : females per 1000 males (Indian Census convention;
      lower = more skewed toward males = more concerning),
    - ``pct_female``       : share of births that are female.
* The natural baseline is ~0.488 female (≈105 boys : 100 girls). A one-sided
  binomial test asks: is the female share *significantly below* baseline?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

NATURAL_PCT_FEMALE = 0.488  # ~105 M : 100 F


@dataclass
class SRBConfig:
    sex_col: str = "sex"
    count_col: Optional[str] = None          # set if data is pre-aggregated
    group_cols: Sequence[str] = field(default_factory=lambda: ["district"])
    female_tokens: Sequence[str] = ("f", "female", "2", "girl")
    male_tokens: Sequence[str] = ("m", "male", "1", "boy")
    baseline_pct_female: float = NATURAL_PCT_FEMALE
    min_births: int = 30                     # ignore tiny, noisy units
    alpha: float = 0.01                      # significance threshold


def load_births(path: str, **read_kwargs) -> pd.DataFrame:
    """Load a births dataset (CSV/TSV/Parquet by extension)."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path, **read_kwargs)
    sep = "\t" if lower.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep, **read_kwargs)


def _normalise_sex(series: pd.Series, cfg: SRBConfig) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(index=s.index, dtype="object")
    out[s.isin([t.lower() for t in cfg.female_tokens])] = "F"
    out[s.isin([t.lower() for t in cfg.male_tokens])] = "M"
    return out


def compute_srb(df: pd.DataFrame, cfg: Optional[SRBConfig] = None) -> pd.DataFrame:
    """Aggregate births to per-group counts and SRB metrics."""
    cfg = cfg or SRBConfig()
    group_cols = list(cfg.group_cols)
    work = df.copy()
    work["_sex"] = _normalise_sex(work[cfg.sex_col], cfg)
    work = work[work["_sex"].isin(["M", "F"])]

    if cfg.count_col and cfg.count_col in work.columns:
        grp = work.groupby(group_cols + ["_sex"])[cfg.count_col].sum()
    else:
        grp = work.groupby(group_cols + ["_sex"]).size()

    tab = grp.unstack("_sex", fill_value=0)
    for col in ("M", "F"):
        if col not in tab.columns:
            tab[col] = 0
    tab = tab.rename(columns={"M": "males", "F": "females"}).reset_index()
    return _finalize_srb(tab)


def _finalize_srb(tab: pd.DataFrame) -> pd.DataFrame:
    """Add total / pct_female / SRB columns to a table that already carries
    integer ``males`` and ``females`` columns (plus any group columns)."""
    tab = tab.copy()
    tab["males"] = tab["males"].fillna(0).astype(int)
    tab["females"] = tab["females"].fillna(0).astype(int)
    tab["total"] = tab["males"] + tab["females"]
    tab["pct_female"] = tab["females"] / tab["total"].replace(0, math.nan)
    tab["srb_f_per_1000_m"] = (
        tab["females"] / tab["males"].replace(0, math.nan) * 1000
    )
    return tab


def from_aggregated_counts(df: pd.DataFrame,
                           group_cols: Sequence[str],
                           males_col: str,
                           females_col: str) -> pd.DataFrame:
    """Build an SRB table from *pre-aggregated* data.

    Many public datasets (CRS, HMIS, data.gov.in) already give male and female
    birth counts per district/year in separate columns. This skips the
    per-birth groupby and goes straight to the metrics that
    :func:`flag_anomalies` consumes.
    """
    group_cols = list(group_cols)
    work = df[group_cols + [males_col, females_col]].copy()
    work = work.rename(columns={males_col: "males", females_col: "females"})
    if group_cols:
        work = work.groupby(group_cols, as_index=False)[["males", "females"]].sum()
    return _finalize_srb(work)


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X~Binom(n, p): one-sided lower-tail (deficit of females).

    Uses a normal approximation with continuity correction for large n, and
    an exact sum for small n - avoids a SciPy dependency.
    """
    if n == 0:
        return 1.0
    if n <= 1000:
        # exact lower tail
        from math import comb
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p)) or 1e-9
    z = (k + 0.5 - mu) / sigma
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def flag_anomalies(srb_table: pd.DataFrame,
                   cfg: Optional[SRBConfig] = None) -> pd.DataFrame:
    """Add a significance test and rank units skewed toward males.

    Returns the table filtered to units with at least ``min_births`` births,
    sorted most-anomalous-first, with columns:
      ``p_value`` (one-sided, female deficit), ``significant`` (bool),
      ``expected_females``, ``female_deficit``.
    """
    cfg = cfg or SRBConfig()
    t = srb_table.copy()
    t = t[t["total"] >= cfg.min_births].copy()
    p = cfg.baseline_pct_female

    t["expected_females"] = (t["total"] * p).round(1)
    t["female_deficit"] = (t["expected_females"] - t["females"]).round(1)
    t["p_value"] = [
        _binom_sf(int(f), int(n), p)
        for f, n in zip(t["females"], t["total"])
    ]
    t["significant"] = (t["p_value"] < cfg.alpha) & (t["pct_female"] < p)

    # Most concerning first: significant deficits, then largest deficit.
    t = t.sort_values(
        by=["significant", "female_deficit", "p_value"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return t
