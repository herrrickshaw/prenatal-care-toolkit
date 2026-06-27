"""PCPNDT Form-F and machine-registration audit logic.

The data model is intentionally lightweight (dataclasses) so records can be
loaded from a CSV, a clinic's export, or a database without coupling to any
particular storage. ``audit_records`` runs a set of named rules and returns a
list of findings plus a per-facility summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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
    # An empty / vague indication is a classic red flag for sex-selection.
    if form.indication and form.indication.strip().lower() in {
        "na", "n/a", "-", "none", "routine", "checkup", "."
    }:
        findings.append(
            Finding(
                form.record_id, form.facility_id, "indication_vague", "medium",
                f"non-clinical indication: {form.indication!r}",
            )
        )
    if (form.gestational_age_weeks is not None
            and not (3 <= form.gestational_age_weeks <= 45)):
        findings.append(
            Finding(
                form.record_id, form.facility_id, "ga_out_of_range", "low",
                f"implausible gestational age: {form.gestational_age_weeks}w",
            )
        )
    return findings


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
    reg_index: Dict[str, List[MachineRegistration]] = {}
    for r in registrations:
        reg_index.setdefault(r.machine_id, []).append(r)

    findings: List[Finding] = []
    per_facility: Dict[str, Dict[str, int]] = {}

    forms = list(forms)
    for form in forms:
        fac = per_facility.setdefault(
            form.facility_id or "<unknown>",
            {"total": 0, "incomplete": 0, "unregistered": 0, "expired_retention": 0},
        )
        fac["total"] += 1

        # 1) Field validation.
        fvs = validate_form_f(form)
        findings.extend(fvs)
        if any(f.rule == "form_f_incomplete" for f in fvs):
            fac["incomplete"] += 1

        # 2) Machine registration check.
        regs = reg_index.get(form.machine_id, [])
        if form.scan_date is not None:
            if not any(r.valid_on(form.scan_date) for r in regs):
                fac["unregistered"] += 1
                findings.append(
                    Finding(
                        form.record_id, form.facility_id,
                        "machine_unregistered", "high",
                        f"machine {form.machine_id!r} not validly registered "
                        f"on {form.scan_date}",
                    )
                )

        # 3) Retention window check.
        if form.scan_date is not None:
            keep_until = form.scan_date.replace(
                year=form.scan_date.year + retention_years
            )
            if today > keep_until:
                fac["expired_retention"] += 1
                findings.append(
                    Finding(
                        form.record_id, form.facility_id,
                        "retention_expired", "low",
                        f"record past {retention_years}y retention "
                        f"(scan {form.scan_date}, keep until {keep_until})",
                    )
                )

    # Facility-level completeness ratio.
    for fac, c in per_facility.items():
        c["completeness_pct"] = round(
            100.0 * (c["total"] - c["incomplete"]) / c["total"], 1
        ) if c["total"] else 0.0

    counts = {
        "records": len(forms),
        "findings": len(findings),
        "high": sum(1 for f in findings if f.severity == "high"),
        "medium": sum(1 for f in findings if f.severity == "medium"),
        "low": sum(1 for f in findings if f.severity == "low"),
    }
    return {
        "findings": findings,
        "facility_summary": per_facility,
        "counts": counts,
    }
