"""Form-F field rules and machine-registration audit (synthetic records only).

Retention rules are deliberately not asserted here: they are being reworked in
PR #2 (fix/pcpndt-retention-minimum). ``today`` is pinned to the scan year so
the retention branch never fires in these tests.
"""
from datetime import date

import pytest

from pctk.compliance.formf import FormF, MachineRegistration, audit_records, validate_form_f

TODAY = date(2026, 6, 30)


def _form(**kw):
    base = dict(patient_ref="PT-SYNTH01", facility_id="FAC-T", machine_id="US-T1",
                scan_date=date(2026, 6, 1), referring_doctor="Dr Synthetic A",
                indication="Anomaly scan", gestational_age_weeks=20.0,
                performed_by="Dr Synthetic B", record_id="R1")
    base.update(kw)
    return FormF(**base)


def _rules(findings):
    return sorted(f.rule for f in findings)


def test_complete_form_has_no_findings():
    assert validate_form_f(_form()) == []


def test_each_missing_required_field_is_named():
    f = validate_form_f(_form(referring_doctor="", gestational_age_weeks=None))
    assert _rules(f) == ["form_f_incomplete"]
    assert "referring_doctor" in f[0].detail and "gestational_age_weeks" in f[0].detail
    assert f[0].severity == "high"


@pytest.mark.parametrize("ga,flag", [(2.9, True), (3.0, False), (45.0, False), (45.1, True)])
def test_gestational_age_plausibility_bounds_are_inclusive(ga, flag):
    assert ("ga_out_of_range" in _rules(validate_form_f(_form(gestational_age_weeks=ga)))) is flag


@pytest.mark.parametrize("indication", ["NA", "n/a", "-", "none", "Routine", "checkup", "."])
def test_original_vague_tokens_still_flagged(indication):
    assert "indication_vague" in _rules(validate_form_f(_form(indication=indication)))


@pytest.mark.parametrize("indication", ["N.A.", "Routine checkup", "routine.", "Nil", "  NA  ", "check up", "general"])
def test_trivially_reworded_non_clinical_indications_are_flagged(indication):
    """The rule was an exact-match 7-word blacklist: punctuation or a second word evaded it."""
    assert "indication_vague" in _rules(validate_form_f(_form(indication=indication)))


@pytest.mark.parametrize("indication", ["Anomaly scan", "Suspected placenta praevia", "Routine anomaly scan",
                                        "Bleeding per vaginum", "Twin pregnancy follow-up"])
def test_clinical_indications_are_not_flagged(indication):
    assert "indication_vague" not in _rules(validate_form_f(_form(indication=indication)))


def test_machine_registration_window_is_inclusive_on_both_ends():
    reg = [MachineRegistration("US-T1", "FAC-T", date(2026, 1, 1), date(2026, 6, 1))]
    on_last_day = audit_records([_form(scan_date=date(2026, 6, 1))], reg, today=TODAY)
    day_after = audit_records([_form(scan_date=date(2026, 6, 2))], reg, today=TODAY)
    day_before = audit_records([_form(scan_date=date(2025, 12, 31))], reg, today=TODAY)
    assert on_last_day["counts"]["high"] == 0
    assert _rules(day_after["findings"]) == ["machine_unregistered"]
    assert _rules(day_before["findings"]) == ["machine_unregistered"]


def test_open_ended_registration_and_wrong_machine():
    reg = [MachineRegistration("US-T1", "FAC-T", date(2020, 1, 1))]
    ok = audit_records([_form()], reg, today=TODAY)
    other = audit_records([_form(machine_id="US-UNKNOWN")], reg, today=TODAY)
    assert ok["counts"]["findings"] == 0
    assert other["facility_summary"]["FAC-T"]["unregistered"] == 1


def test_facility_completeness_and_counts():
    reg = [MachineRegistration("US-T1", "FAC-T", date(2020, 1, 1))]
    forms = [_form(record_id="R1"), _form(record_id="R2", performed_by=""),
             _form(record_id="R3"), _form(record_id="R4", facility_id="")]
    res = audit_records(forms, reg, today=TODAY)
    s = res["facility_summary"]
    assert s["FAC-T"]["total"] == 3 and s["FAC-T"]["incomplete"] == 1
    assert s["FAC-T"]["completeness_pct"] == pytest.approx(66.7)
    assert s["<unknown>"]["completeness_pct"] == 0.0     # missing facility_id is itself incomplete
    assert res["counts"]["records"] == 4 and res["counts"]["high"] == 2


def test_form_without_scan_date_skips_registration_check_but_is_incomplete():
    res = audit_records([_form(scan_date=None)], [], today=TODAY)
    assert _rules(res["findings"]) == ["form_f_incomplete"]
