"""Adapters that map real public Indian natality datasets to the SRB schema.

Public sources rarely ship one-row-per-birth. They ship *aggregated* male and
female birth counts (or a ready-made ratio) per district/state/year, with
inconsistent column names. Each adapter here:

  * accepts a raw DataFrame as published,
  * renames / cleans the relevant columns,
  * returns an SRB table (via ``from_aggregated_counts``) that
    ``flag_anomalies`` can consume directly.

Where a source only publishes a *ratio* (females per 1000 males) and no counts,
no binomial test is possible; ``from_ratio_only`` returns the ratio with a clear
flag so callers don't silently over-interpret it.

Column mappings are best-effort and documented per source; pass an explicit
``column_map`` to override when a particular release differs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from .analyze import NATURAL_PCT_FEMALE, _finalize_srb, from_aggregated_counts


def _resolve(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Find the first column whose lowercased name matches a candidate."""
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # loose contains-match fallback
    for cand in candidates:
        for lc, orig in lower.items():
            if cand.lower() in lc:
                return orig
    return None


# --------------------------------------------------------------------------- #
# Civil Registration System (CRS) / HMIS style: explicit M/F birth counts.
# --------------------------------------------------------------------------- #
def from_crs_hmis(df: pd.DataFrame,
                  group_cols: Optional[Sequence[str]] = None,
                  column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """CRS / HMIS district births with separate male & female count columns.

    Typical headers: ``State``, ``District``, ``Year``,
    ``Male``/``Births - Male``/``live_births_male``, and the female analogue.
    """
    column_map = column_map or {}
    males = column_map.get("males") or _resolve(
        df, ["male", "births_male", "births - male", "live_births_male",
              "male_births", "boys"])
    females = column_map.get("females") or _resolve(
        df, ["female", "births_female", "births - female",
              "live_births_female", "female_births", "girls"])
    if not males or not females:
        raise ValueError(
            "could not locate male/female birth-count columns; pass "
            "column_map={'males': ..., 'females': ...}")
    if group_cols is None:
        group_cols = [c for c in ("State", "District", "Year")
                      if _resolve(df, [c])]
        group_cols = [_resolve(df, [c]) for c in group_cols]
    return from_aggregated_counts(df, group_cols, males, females)


# --------------------------------------------------------------------------- #
# data.gov.in "Sex Ratio at Birth" resources: often ratio-only.
# --------------------------------------------------------------------------- #
def from_ratio_only(df: pd.DataFrame,
                    group_cols: Optional[Sequence[str]] = None,
                    ratio_col: Optional[str] = None,
                    ratio_kind: str = "f_per_1000_m") -> pd.DataFrame:
    """Datasets that publish only SRB (females per 1000 males), no raw counts.

    Returns a table with ``srb_f_per_1000_m``, ``pct_female`` (derived) and a
    ``counts_available=False`` flag. No significance test is attached because
    none is statistically valid without denominators.
    """
    ratio_col = ratio_col or _resolve(
        df, ["srb", "sex ratio at birth", "sex_ratio_at_birth",
              "sexratio", "ratio", "females per 1000 males"])
    if not ratio_col:
        raise ValueError("could not locate the sex-ratio column; pass ratio_col")
    if group_cols is None:
        group_cols = [c for c in df.columns if c != ratio_col]
    out = df[list(group_cols) + [ratio_col]].copy()
    r = pd.to_numeric(out[ratio_col], errors="coerce")
    if ratio_kind == "m_per_100_f":          # convert to females per 1000 males
        r = 1000.0 / (r / 100.0)
    out["srb_f_per_1000_m"] = r
    out["pct_female"] = r / (r + 1000.0)     # F / (F + M) with M = 1000 basis
    out["counts_available"] = False
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Census / SRS style: a single "sex ratio" column, sometimes child sex ratio.
# --------------------------------------------------------------------------- #
def from_census_csr(df: pd.DataFrame,
                    group_cols: Optional[Sequence[str]] = None,
                    csr_col: Optional[str] = None) -> pd.DataFrame:
    """Census Child Sex Ratio (girls per 1000 boys, 0-6 yrs). Ratio-only."""
    csr_col = csr_col or _resolve(
        df, ["child sex ratio", "child_sex_ratio", "csr",
              "0-6 sex ratio", "sex ratio (0-6)"])
    if not csr_col:
        raise ValueError("could not locate the child-sex-ratio column; pass csr_col")
    return from_ratio_only(df, group_cols, ratio_col=csr_col,
                           ratio_kind="f_per_1000_m")


# --------------------------------------------------------------------------- #
# NFHS district factsheets (long format: one indicator row per district).
# --------------------------------------------------------------------------- #
def from_nfhs(df: pd.DataFrame,
              group_cols: Optional[Sequence[str]] = None,
              indicator_col: Optional[str] = None,
              value_col: Optional[str] = None,
              srb_pattern: str = "sex ratio at birth") -> pd.DataFrame:
    """NFHS-4/5 district factsheets (e.g. pratapvardhan/NFHS-5).

    These are long: columns ``State``, ``District``, ``Indicator``, ``NFHS-5``,
    ``NFHS-4`` ... with one row per indicator. We pick the rows whose indicator
    is "Sex ratio at birth ... (females per 1,000 males)" and read the value
    column. Ratio-only -> no significance test.
    """
    indicator_col = indicator_col or _resolve(df, ["indicator", "indicators"])
    value_col = value_col or _resolve(df, ["NFHS-5", "nfhs5", "value", "NFHS-4"])
    if not indicator_col or not value_col:
        raise ValueError("could not locate indicator/value columns; pass "
                         "indicator_col= and value_col=")
    mask = df[indicator_col].astype(str).str.contains(
        srb_pattern, case=False, na=False, regex=True)
    sub = df[mask].copy()
    if sub.empty:
        raise ValueError(f"no rows match indicator pattern {srb_pattern!r}")
    if group_cols is None:
        group_cols = [_resolve(sub, [c]) for c in ("State", "District")
                      if _resolve(sub, [c])]
    sub = sub[list(group_cols) + [value_col]]
    return from_ratio_only(sub, group_cols=group_cols, ratio_col=value_col,
                           ratio_kind="f_per_1000_m")


# --------------------------------------------------------------------------- #
# Census Primary Census Abstract: 0-6 child population by sex (M_06 / F_06).
# The canonical foeticide proxy - and, having raw counts, it supports the
# binomial significance test (unlike the ratio-only census/nfhs sources).
# --------------------------------------------------------------------------- #
def from_census_child_06(df: pd.DataFrame,
                         group_cols: Optional[Sequence[str]] = None,
                         column_map: Optional[Dict[str, str]] = None
                         ) -> pd.DataFrame:
    """Census PCA child (0-6) population counts by sex -> SRB table with counts.

    Resolves the boys/girls 0-6 columns (``M_06``/``F_06`` and common variants).
    """
    column_map = column_map or {}
    males = column_map.get("males") or _resolve(
        df, ["m_06", "male_0_6", "p_06_male", "child_male_0_6", "boys_0_6",
              "m06", "males 0-6", "population 0-6 male"])
    females = column_map.get("females") or _resolve(
        df, ["f_06", "female_0_6", "p_06_female", "child_female_0_6",
              "girls_0_6", "f06", "females 0-6", "population 0-6 female"])
    if not males or not females:
        raise ValueError(
            "could not locate 0-6 child male/female columns (e.g. M_06/F_06); "
            "pass column_map={'males': ..., 'females': ...}")
    if group_cols is None:
        group_cols = [_resolve(df, [c]) for c in ("State", "District")
                      if _resolve(df, [c])]
    return from_aggregated_counts(df, group_cols, males, females)


SOURCE_ADAPTERS = {
    "crs": from_crs_hmis,
    "hmis": from_crs_hmis,
    "ratio": from_ratio_only,
    "census": from_census_csr,
    "nfhs": from_nfhs,
    "child06": from_census_child_06,
}


def adapt(source: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Dispatch to a named adapter: crs | hmis | ratio | census."""
    key = source.lower()
    if key not in SOURCE_ADAPTERS:
        raise ValueError(f"unknown source {source!r}; "
                         f"choose from {sorted(SOURCE_ADAPTERS)}")
    return SOURCE_ADAPTERS[key](df, **kwargs)


# Data dictionary documenting where each source comes from.
SOURCE_NOTES = {
    "crs": "Civil Registration System annual report tables; male & female "
           "registered live births by state/district/year. Has counts -> "
           "full significance testing supported.",
    "hmis": "Health Management Information System (hmis.nhp.gov.in); monthly "
            "live births by sex and district. Has counts.",
    "ratio": "data.gov.in 'Sex Ratio at Birth' resources; usually ratio-only "
             "(females per 1000 males). No counts -> no significance test.",
    "census": "Census of India child sex ratio (0-6 yrs), girls per 1000 boys. "
              "Ratio-only.",
    "nfhs": "NFHS-4/5 district factsheets (rchiips.org/nfhs; pratapvardhan/"
            "NFHS-5). 'Sex ratio at birth, last 5 years (females per 1,000 "
            "males)'. Ratio-only -> no significance test.",
    "child06": "Census Primary Census Abstract 0-6 child population by sex "
               "(M_06/F_06). The canonical foeticide proxy; has counts -> "
               "full significance testing supported.",
}
