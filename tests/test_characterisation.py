"""Characterisation tests: current behaviour must equal the pre-refactor baseline.

``tests/golden/baseline.json`` was generated from the original code by
``tests/characterise.py`` (synthetic data only). Each section is its own test
so a regression names the module that moved.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import characterise  # noqa: E402

with open(os.path.join(HERE, "golden", "baseline.json")) as _fh:
    GOLDEN = json.load(_fh)


def _roundtrip(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


@pytest.mark.parametrize("section,func", [
    ("pixel", characterise.pixel_cases),
    ("dicom", characterise.dicom_cases),
    ("compliance", characterise.compliance_cases),
    ("ingest", characterise.ingest_cases),
    ("cli", characterise.cli_cases),
    ("srb", characterise.srb_cases),
])
def test_matches_baseline(section, func):
    assert _roundtrip(func()) == GOLDEN[section]
