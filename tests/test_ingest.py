"""Form-F text parsing and the name pseudonym (synthetic names only)."""
import hashlib
from datetime import date

import pytest

from pctk.compliance import ingest
from pctk.compliance.ingest import _hash_ref, _parse_date, parse_form_f_text

# Synthetic, deliberately implausible names.
CANDIDATES = ["Testa Synthetica", "Exa Mple", "Dummy Patientone", "Fictiva Kumari"]

FORM_TEXT = (
    "FORM F\n"
    "Name of pregnant woman: Fictiva Kumari\n"
    "Date of test: 05-Mar-2026\n"
    "Referred by: Dr Synth One\n"
    "Indication for ultrasound: Suspected IUGR\n"
    "Period of gestation: 28 weeks\n"
    "Performed by: Dr Synth Two\n"
    "USG Machine No: US-SYN-9\n"
    "Centre Registration No: FAC-SYN\n"
)


def _unkeyed(name):
    return "PT-" + hashlib.sha256(name.strip().lower().encode()).hexdigest()[:10].upper()


def test_pseudonym_resists_dictionary_attack(monkeypatch):
    """An attacker with a list of common names must not be able to recompute patient_ref.

    The old ref was an unsalted, truncated SHA-256 of the lower-cased name, so
    hashing a name list re-identified every record.
    """
    monkeypatch.delenv("PCTK_PSEUDONYM_KEY", raising=False)
    form, _ = parse_form_f_text(FORM_TEXT)
    lookup = {_unkeyed(n): n for n in CANDIDATES}
    assert form.patient_ref not in lookup


def test_pseudonym_is_stable_and_case_insensitive_under_one_key(monkeypatch):
    monkeypatch.setenv("PCTK_PSEUDONYM_KEY", "synthetic-test-key")
    assert _hash_ref("Exa Mple") == _hash_ref("  exa mple ")
    assert _hash_ref("Exa Mple") != _hash_ref("Exa Mplf")


def test_pseudonym_depends_on_key(monkeypatch):
    monkeypatch.setenv("PCTK_PSEUDONYM_KEY", "key-one")
    a = _hash_ref("Exa Mple")
    monkeypatch.setenv("PCTK_PSEUDONYM_KEY", "key-two")
    b = _hash_ref("Exa Mple")
    assert a != b and a.startswith("PT-") and b.startswith("PT-")


def test_returned_raw_map_does_not_carry_the_name():
    """README: the woman's name 'is never stored'. The raw capture map is returned to callers."""
    form, raw = parse_form_f_text(FORM_TEXT)
    assert "Fictiva" not in " ".join(raw.values())
    assert "Fictiva" not in " ".join(str(v) for v in vars(form).values())


def test_fields_parsed_from_synthetic_form():
    form, _ = parse_form_f_text(FORM_TEXT, facility_id="IGNORED")
    assert form.scan_date == date(2026, 3, 5)
    assert form.gestational_age_weeks == 28.0
    assert form.indication == "Suspected IUGR"
    assert form.machine_id == "US-SYN-9"
    assert form.facility_id == "FAC-SYN"            # parsed value wins over the default
    assert form.performed_by == "Dr Synth Two"


def test_facility_default_used_when_absent():
    form, _ = parse_form_f_text("Date of test: 2026-01-02\n", facility_id="FAC-DEFAULT")
    assert form.facility_id == "FAC-DEFAULT"
    assert form.scan_date == date(2026, 1, 2)
    assert form.patient_ref == ""


@pytest.mark.parametrize("s,expected", [("2026-03-05", date(2026, 3, 5)), ("05/03/2026", date(2026, 3, 5)),
                                        ("05-Mar-2026", date(2026, 3, 5)), ("31-02-2026", None),
                                        ("garbage", None)])
def test_parse_date_formats(s, expected):
    assert _parse_date(s) == expected
