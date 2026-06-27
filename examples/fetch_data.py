"""Fetch real public datasets the SRB adapters consume.

Pulls two live sources into ``data/`` so ``pctk srb --source ...`` runs on real
numbers with one command:

  * NFHS-5 district factsheets  -> data/nfhs5_districts.csv   (--source nfhs)
        concatenated from the 21 per-state CSVs in pratapvardhan/NFHS-5.
  * Census 2011 PCA 0-6 child   -> data/census_pca06.csv      (--source child06)
        district totals (M_06 / F_06) from pigshell/india-census-2011.

Uses only the standard library for downloading (+ pandas for the NFHS concat),
so no extra dependencies. All sources are public, openly licensed mirrors of
Government of India data; see README for provenance.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

NFHS_API = ("https://api.github.com/repos/pratapvardhan/NFHS-5/contents/"
            "district-level")
PCA_TOTAL = ("https://raw.githubusercontent.com/pigshell/india-census-2011/"
             "master/pca-total.csv")

_UA = {"User-Agent": "prenatal-care-toolkit/0.1 (+https://example.org)"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_nfhs5_districts(dest: str = None) -> str:
    """Download every per-state NFHS-5 district CSV and concatenate them."""
    import pandas as pd

    dest = dest or os.path.join(DATA, "nfhs5_districts.csv")
    listing = json.loads(_get(NFHS_API).decode("utf-8"))
    files = [f for f in listing if f["name"].endswith(".csv")]
    frames = []
    for i, f in enumerate(files, 1):
        try:
            raw = _get(f["download_url"])
            frames.append(pd.read_csv(io.BytesIO(raw)))
            print(f"  [{i:2}/{len(files)}] {f['name']}")
        except Exception as exc:
            print(f"  [{i:2}/{len(files)}] SKIP {f['name']}: {exc}")
    if not frames:
        raise RuntimeError("no NFHS-5 state files downloaded")
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA, exist_ok=True)
    df.to_csv(dest, index=False)
    n_dist = df["District"].nunique() if "District" in df.columns else "?"
    print(f"-> {dest}  ({len(df)} rows, {n_dist} districts)")
    return dest


def fetch_census_pca06(dest: str = None) -> str:
    """Download the Census 2011 PCA district-total file (has M_06 / F_06)."""
    dest = dest or os.path.join(DATA, "census_pca06.csv")
    os.makedirs(DATA, exist_ok=True)
    raw = _get(PCA_TOTAL)
    with open(dest, "wb") as fh:
        fh.write(raw)
    nlines = raw.count(b"\n")
    print(f"-> {dest}  (~{nlines} district rows; columns include M_06, F_06)")
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch real SRB datasets into data/")
    p.add_argument("--nfhs", action="store_true", help="NFHS-5 districts only")
    p.add_argument("--pca", action="store_true", help="Census PCA 0-6 only")
    args = p.parse_args(argv)
    do_all = not (args.nfhs or args.pca)

    if args.nfhs or do_all:
        print("Fetching NFHS-5 district factsheets ...")
        try:
            fetch_nfhs5_districts()
        except Exception as exc:
            print(f"  NFHS fetch failed: {exc}", file=sys.stderr)
    if args.pca or do_all:
        print("Fetching Census 2011 PCA 0-6 child counts ...")
        try:
            fetch_census_pca06()
        except Exception as exc:
            print(f"  PCA fetch failed: {exc}", file=sys.stderr)

    print("\nNow run, e.g.:")
    print("  pctk srb data/nfhs5_districts.csv --source nfhs")
    print("  pctk srb data/census_pca06.csv  --source child06 --group State Name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
