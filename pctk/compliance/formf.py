"""PCPNDT Form-F and machine-registration audit logic.

The data model is intentionally lightweight (dataclasses) so records can be
loaded from a CSV, a clinic's export, or a database without coupling to any
particular storage. ``audit_records`` runs a set of named rules and returns a
list of findings plus a per-facility summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional


# Fields the PCPNDT Form F must carry for a prenatal scan to be compliant.
REQUIRED_FORM_F_FIELDS = (
    "patient_ref",          # pseudonymous patient reference
    "facility_id",
    "machine_id",
    "scan_date",
    "referring_doctor",
    "indication",           # clinical reason for the scan
    "gestational_age_weeks",
    "performed_by",
)

DEFAULT_RETENTION_YEARS = 2


@dataclass
class MachineRegistration:
    machine_id: str
    facility_id: str
    registered_from: date
    registered_to: Optional[date] = None   # None = currently valid

    def valid_on(self, when: date) -> bool:
        if when < self.registered_from:
            return False
        if self.registered_to is not None and when > self.registered_to:
            return False
        return True


@dataclass
class FormF:
    patient_ref: str = ""
    facility_id: str = ""
    machine_id: str = ""
    scan_date: Optional[date] = None
    referring_doctor: str = ""
    indication: str = ""
    gestational_age_weeks: Optional[float] = None
    performed_by: str = ""
    record_id: str = ""

    def missing_fields(self) -> List[str]:
        miss = []
        for f in REQUIRED_FORM_F_FIELDS:
            val = getattr(self, f, None)
            if val in (None, ""):
                miss.append(f)
        return miss


@dataclass
class Finding:
    record_id: str
    facility_id: str
    rule: str
    severity: str          # "high" | "medium" | "low"
    detail: str


# Indications that carry no clinical information (a classic sex-selection red flag).
VAGUE_INDICATIONS = frozenset({"na", "n/a", "-", "none", "routine", "checkup", "."})
# Plausible gestational-age window (weeks) for a prenatal scan.
GA_MIN_WEEKS = 3
GA_MAX_WEEKS = 45


def validate_form_f(form: FormF) -> List[Finding]:
    """Field-level validation of a single Form F."""
    findings: List[Finding] = []
    miss = form.missing_fields()
    if miss:
        findings.append(
            Finding(
                form.record_id, form.facility_id, "form_f_incomplete", "high",
                f"missing required fields: {', '.join(miss)}",
            )
        )
    if form.indication and form.indication.strip().lower() in VAGUE_INDICATIONS:
        findings.append(
            Finding(
                form.record_id, form.facility_id, "indication_vague", "medium",
                f"non-clinical indication: {form.indication!r}",
            )
        )
    if (form.gestational_age_weeks is not None
            and not (GA_MIN_WEEKS <= form.gestational_age_weeks <= GA_MAX_WEEKS)):
        findings.append(
            Finding(
                form.record_id, form.facility_id, "ga_out_of_range", "low",
                f"implausible gestational age: {form.gestational_age_weeks}w",
            )
        )
    return findings


def index_registrations(
    registrations: Iterable[MachineRegistration],
) -> Dict[str, List[MachineRegistration]]:
    """Group registrations by ``machine_id``."""
    index: Dict[str, List[MachineRegistration]] = {}
    for r in registrations:
        index.setdefault(r.machine_id, []).append(r)
    return index


def check_registration(
    form: FormF, reg_index: Dict[str, List[MachineRegistration]]
) -> Optional[Finding]:
    """Finding if the scan date is not covered by a valid machine registration."""
    if form.scan_date is None:
        return None
    regs = reg_index.get(form.machine_id, [])
    if any(r.valid_on(form.scan_date) for r in regs):
        return None
    return Finding(
        form.record_id, form.facility_id, "machine_unregistered", "high",
        f"machine {form.machine_id!r} not validly registered on {form.scan_date}",
    )


def check_retention(
    form: FormF, today: date, retention_years: int
) -> Optional[Finding]:
    """Finding if the record is older than the retention window."""
    if form.scan_date is None:
        return None
    # NB: raises ValueError for a 29-Feb scan date when the target year is not a
    # leap year (pre-existing behaviour, preserved by the refactor).
    keep_until = form.scan_date.replace(year=form.scan_date.year + retention_years)
    if today <= keep_until:
        return None
    return Finding(
        form.record_id, form.facility_id, "retention_expired", "low",
        f"record past {retention_years}y retention "
        f"(scan {form.scan_date}, keep until {keep_until})",
    )


def _completeness_pct(counts: Dict[str, int]) -> float:
    if not counts["total"]:
        return 0.0
    return round(100.0 * (counts["total"] - counts["incomplete"]) / counts["total"], 1)


def audit_records(
    forms: Iterable[FormF],
    registrations: Iterable[MachineRegistration],
    today: Optional[date] = None,
    retention_years: int = DEFAULT_RETENTION_YEARS,
) -> Dict[str, object]:
    """Run the full audit over a set of Form F records.

    Returns ``{"findings": [...], "facility_summary": {...}, "counts": {...}}``.
    """
    today = today or date.today()
    reg_index = index_registrations(registrations)

    findings: List[Finding] = []
    per_facility: Dict[str, Dict[str, int]] = {}

    forms = list(forms)
    for form in forms:
        fac = per_facility.setdefault(
            form.facility_id or "<unknown>",
            {"total": 0, "incomplete": 0, "unregistered": 0, "expired_retention": 0},
        )
        fac["total"] += 1

        field_findings = validate_form_f(form)
        findings.extend(field_findings)
        if any(f.rule == "form_f_incomplete" for f in field_findings):
            fac["incomplete"] += 1

        unregistered = check_registration(form, reg_index)
        if unregistered:
            fac["unregistered"] += 1
            findings.append(unregistered)

        expired = check_retention(form, today, retention_years)
        if expired:
            fac["expired_retention"] += 1
            findings.append(expired)

    for counts in per_facility.values():
        counts["completeness_pct"] = _completeness_pct(counts)

    counts = {
        "records": len(forms),
        "findings": len(findings),
        **{sev: sum(1 for f in findings if f.severity == sev)
           for sev in ("high", "medium", "low")},
    }
    return {
        "findings": findings,
        "facility_summary": per_facility,
        "counts": counts,
    }
