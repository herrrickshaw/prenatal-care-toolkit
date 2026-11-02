"""CSV input/output for Form-F records and machine registrations.

Kept separate from the pure audit rules in :mod:`pctk.compliance.formf` so the
rules never touch the filesystem.
"""

from __future__ import annotations

import csv
import os
from datetime import date, datetime
from typing import List, Optional

from .formf import FormF, MachineRegistration

# Accepted date layouts in user-supplied CSVs, tried in order.
CSV_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%Y%m%d", "%d/%m/%Y")

# Column order when appending an ingested Form F to a forms CSV.
FORM_CSV_HEADER = (
    "record_id", "facility_id", "machine_id", "scan_date", "referring_doctor",
    "indication", "gestational_age_weeks", "performed_by", "patient_ref",
)


def parse_csv_date(s: Optional[str]) -> Optional[date]:
    """Parse a CSV date cell (see ``CSV_DATE_FORMATS``); ``None`` if blank/invalid."""
    if not s:
        return None
    for fmt in CSV_DATE_FORMATS:
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except ValueError:
            continue
    return None


def load_forms_csv(path: str) -> List[FormF]:
    """Read Form-F rows from a CSV file."""
    forms: List[FormF] = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            ga = row.get("gestational_age_weeks") or ""
            forms.append(FormF(
                patient_ref=row.get("patient_ref", ""),
                facility_id=row.get("facility_id", ""),
                machine_id=row.get("machine_id", ""),
                scan_date=parse_csv_date(row.get("scan_date")),
                referring_doctor=row.get("referring_doctor", ""),
                indication=row.get("indication", ""),
                gestational_age_weeks=float(ga) if ga else None,
                performed_by=row.get("performed_by", ""),
                record_id=row.get("record_id", f"row{i}"),
            ))
    return forms


def load_registrations_csv(path: str) -> List[MachineRegistration]:
    """Read machine registrations from a CSV file."""
    regs: List[MachineRegistration] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            regs.append(MachineRegistration(
                machine_id=row.get("machine_id", ""),
                facility_id=row.get("facility_id", ""),
                registered_from=parse_csv_date(row.get("registered_from")) or date.min,
                registered_to=parse_csv_date(row.get("registered_to")),
            ))
    return regs


def append_form_csv(path: str, form: FormF) -> None:
    """Append one Form F to a forms CSV, writing the header if the file is new."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(FORM_CSV_HEADER)
        w.writerow([getattr(form, h) if getattr(form, h) is not None else ""
                    for h in FORM_CSV_HEADER])
