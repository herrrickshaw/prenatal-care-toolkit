"""Unit tests for the pure helpers extracted during the SDLC refactor (synthetic data)."""

from __future__ import annotations

import csv
from datetime import date

import numpy as np
import pytest

from pctk.compliance import FormF, MachineRegistration
from pctk.compliance.formf import (
    check_registration,
    check_retention,
    index_registrations,
    validate_form_f,
)
from pctk.compliance.io import (
    append_form_csv,
    load_forms_csv,
    load_registrations_csv,
    parse_csv_date,
)
from pctk.deid import ocr_backends
from pctk.deid.pixel_text import REDACTORS, _redact_region


def _form(**kw):
    base = dict(record_id="r", facility_id="F", machine_id="M", patient_ref="P",
                referring_doctor="Dr", indication="dating", gestational_age_weeks=12,
                performed_by="T", scan_date=date(2025, 1, 1))
    base.update(kw)
    return FormF(**base)


def test_check_registration_covers_dates():
    idx = index_registrations([MachineRegistration("M", "F", date(2024, 1, 1), date(2025, 6, 1))])
    assert check_registration(_form(), idx) is None
    late = check_registration(_form(scan_date=date(2026, 1, 1)), idx)
    assert late.rule == "machine_unregistered" and late.severity == "high"
    assert check_registration(_form(scan_date=None), idx) is None


def test_check_retention_boundary():
    form = _form(scan_date=date(2024, 3, 1))
    assert check_retention(form, date(2026, 3, 1), 2) is None      # exactly on the line
    assert check_retention(form, date(2026, 3, 2), 2).rule == "retention_expired"
    assert check_retention(_form(scan_date=None), date(2030, 1, 1), 2) is None


@pytest.mark.parametrize("ga,flagged", [(3, False), (45, False), (2.9, True), (46, True)])
def test_ga_bounds(ga, flagged):
    rules = {f.rule for f in validate_form_f(_form(gestational_age_weeks=ga))}
    assert ("ga_out_of_range" in rules) is flagged


def test_parse_csv_date_formats():
    assert parse_csv_date("2025-03-10") == date(2025, 3, 10)
    assert parse_csv_date("10/03/2025") == date(2025, 3, 10)
    assert parse_csv_date("20250310") == date(2025, 3, 10)
    assert parse_csv_date("") is None and parse_csv_date("junk") is None


def test_csv_roundtrip(tmp_path):
    path = str(tmp_path / "forms.csv")
    append_form_csv(path, _form(record_id="a"))
    append_form_csv(path, _form(record_id="b", gestational_age_weeks=None))
    rows = list(csv.reader(open(path)))
    assert rows[0][0] == "record_id" and len(rows) == 3
    forms = load_forms_csv(path)
    assert [f.record_id for f in forms] == ["a", "b"]
    assert forms[0].gestational_age_weeks == 12.0 and forms[1].gestational_age_weeks is None

    regs_path = tmp_path / "regs.csv"
    regs_path.write_text("machine_id,facility_id,registered_from,registered_to\n"
                         "M,F,,2025-01-01\n")
    reg = load_registrations_csv(str(regs_path))[0]
    assert reg.registered_from == date.min and reg.registered_to == date(2025, 1, 1)


def test_redactors_shape_preserving():
    region = np.random.RandomState(1).randint(0, 255, (30, 40, 3)).astype("uint8")
    assert set(REDACTORS) == {"blackout", "blur", "pixelate"}
    for fn in REDACTORS.values():
        assert fn(region).shape == region.shape
    assert REDACTORS["blackout"](region).max() == 0


def test_redact_region_clips_and_rejects_unknown():
    arr = np.full((10, 10, 3), 200, dtype=np.uint8)
    _redact_region(arr, -5, -5, 8, 8, "blackout")      # clipped to the top-left corner
    assert arr[:3, :3].max() == 0 and arr[5:, 5:].min() == 200
    with pytest.raises(ValueError):
        _redact_region(arr, 0, 0, 5, 5, "nope")


def test_get_backend_unknown_name_is_none():
    assert ocr_backends.get_backend("nonexistent") is None


def test_build_parser_matches_golden():
    """Pin the CLI surface (help text, defaults, handlers, subcommand order)."""
    import argparse
    import json
    import os

    from pctk.cli import build_parser

    def dump(p):
        out = {"help": p.format_help()}
        for a in p._actions:
            if isinstance(a, argparse._SubParsersAction):
                out["sub"] = {
                    n: dump(sp) | {"func": sp.get_default("func").__name__
                                   if sp.get_default("func") else None}
                    for n, sp in a.choices.items()}
                out["order"] = list(a.choices)
        return out

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "golden", "parser.json")
    with open(path) as fh:
        assert json.dumps(dump(build_parser()), sort_keys=True, indent=1) == \
            json.dumps(json.load(fh), sort_keys=True, indent=1)
