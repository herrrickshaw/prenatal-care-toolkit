"""Characterisation tests: current behaviour must equal the pre-refactor baseline.

``tests/golden/baseline.json`` was generated from the original code by
``tests/characterise.py`` (synthetic data only). Each section is its own test
so a regression names the module that moved.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import characterise  # noqa: E402

with open(os.path.join(HERE, "golden", "baseline.json")) as _fh:
    GOLDEN = json.load(_fh)


def _roundtrip(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


_NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def _close(a, b, rel=1e-9):
    """Exact on keys/structure/text; floats (also inside CSV text) within ``rel``.

    scipy's p-values differ by ~1 ulp between releases (e.g. ...6326 vs ...6325),
    which is numerical noise, not a behaviour change.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return True
        if _NUM.sub("#", a) != _NUM.sub("#", b):
            return False
        return all(math.isclose(float(x), float(y), rel_tol=rel)
                   for x, y in zip(_NUM.findall(a), _NUM.findall(b)))
    return a == b


@pytest.mark.parametrize("section,func", [
    ("pixel", characterise.pixel_cases),
    ("dicom", characterise.dicom_cases),
    ("compliance", characterise.compliance_cases),
    ("ingest", characterise.ingest_cases),
    ("cli", characterise.cli_cases),
    ("srb", characterise.srb_cases),
])
def test_matches_baseline(section, func):
    actual = _roundtrip(func())
    assert _close(actual, GOLDEN[section]), (actual, GOLDEN[section])
