"""prenatal-care-toolkit (pctk)

A toolkit of *defensive* and *analytical* utilities for prenatal imaging and
data, built around preventing sex-selective abortion (female foeticide) and
protecting patient privacy.

Design principle
----------------
This toolkit deliberately does NOT detect, classify, or reveal fetal sex.
It contains no fetal-sex / fetal-genitalia detector, because such a detector
is precisely the capability that enables foeticide and that the PCPNDT Act,
1994 criminalises. Instead it attacks the problem where it is actually
fightable:

  1. deid       - remove patient-identifying info (PHI) from images/headers
  2. srb        - surface anomalous sex-ratio-at-birth signals for auditors
  3. compliance - PCPNDT Form-F and machine-registration audit tooling
  4. biometry   - standard, sex-neutral fetal biometry / gestational age

Modules degrade gracefully when optional dependencies (OCR engines, etc.)
are missing.
"""

__version__ = "0.3.0"

__all__ = ["deid", "srb", "compliance", "biometry", "planes", "health",
           "__version__"]
