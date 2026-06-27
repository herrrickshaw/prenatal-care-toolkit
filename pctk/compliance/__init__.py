"""Problem 3 - PCPNDT compliance and audit tooling.

The Pre-Conception and Pre-Natal Diagnostic Techniques (PCPNDT) Act, 1994
requires every facility running a prenatal ultrasound to:
  * be registered, with each machine on the registration,
  * record a Form F for every pregnancy scan (referral reason, referring
    doctor, etc.),
  * retain records (commonly 2 years) for inspection.

This module provides a small data model and an auditor's rule engine that
flags the gaps a PCPNDT Appropriate Authority looks for:
  * scans with a missing / incomplete Form F,
  * scans on an unregistered or lapsed machine,
  * records past their retention window,
  * facilities whose Form-F completeness is poor.

Pure validation logic - no imaging, no sex.
"""

from .formf import (
    FormF,
    MachineRegistration,
    audit_records,
    validate_form_f,
)
from .ingest import (
    extract_text,
    ingest_form_f,
    parse_form_f_text,
)

__all__ = [
    "FormF", "MachineRegistration", "audit_records", "validate_form_f",
    "extract_text", "ingest_form_f", "parse_form_f_text",
]
