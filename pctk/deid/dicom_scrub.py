"""De-identify DICOM headers.

Implements a practical subset of the DICOM PS3.15 "Basic Application Level
Confidentiality Profile" plus catch-all rules:

* a curated action table for the well-known PHI tags,
* a value-representation catch-all that blanks every Person Name (VR ``PN``),
* optional removal of all private tags,
* deterministic UID remapping (referential integrity preserved across a study),
* a configurable date policy (keep / blank / shift-by-consistent-offset).

The goal is to make a study safe to share for research / audit while keeping
the attributes a clinician or auditor legitimately needs (modality, patient
age, gestational measurements, etc.).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.uid import generate_uid
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pydicom is required for DICOM de-identification. "
        "Install it with:  pip install pydicom"
    ) from exc


# Actions, mirroring the PS3.15 vocabulary that we actually use here.
REMOVE = "X"        # delete the element entirely
BLANK = "Z"         # replace value with empty (zero-length)
DUMMY = "D"         # replace with a non-identifying dummy value
HASH_UID = "U"      # replace UID with a deterministically remapped UID
DATE = "DATE"       # handled by the date policy

# Deterministic root for remapped UIDs (an example/private root; replace with
# your own registered root in production).
PCTK_UID_ROOT = "1.2.826.0.1.3680043.10.9999."

# Curated PHI action table  ->  {tag (group, element): action}
PHI_ACTIONS: Dict[Tuple[int, int], str] = {
    (0x0008, 0x0014): REMOVE,   # Instance Creator UID
    (0x0008, 0x0018): HASH_UID,  # SOP Instance UID
    (0x0008, 0x0020): DATE,     # Study Date
    (0x0008, 0x0021): DATE,     # Series Date
    (0x0008, 0x0022): DATE,     # Acquisition Date
    (0x0008, 0x0023): DATE,     # Content Date
    (0x0008, 0x0030): BLANK,    # Study Time
    (0x0008, 0x0050): DUMMY,    # Accession Number
    (0x0008, 0x0080): BLANK,    # Institution Name
    (0x0008, 0x0081): REMOVE,   # Institution Address
    (0x0008, 0x0090): BLANK,    # Referring Physician Name
    (0x0008, 0x0092): REMOVE,   # Referring Physician Address
    (0x0008, 0x0094): REMOVE,   # Referring Physician Telephone
    (0x0008, 0x1010): BLANK,    # Station Name
    (0x0008, 0x1030): BLANK,    # Study Description (may embed names)
    (0x0008, 0x1048): REMOVE,   # Physician(s) of Record
    (0x0008, 0x1050): BLANK,    # Performing Physician Name
    (0x0008, 0x1070): BLANK,    # Operators Name
    (0x0010, 0x0010): DUMMY,    # Patient Name
    (0x0010, 0x0020): DUMMY,    # Patient ID
    (0x0010, 0x0030): BLANK,    # Patient Birth Date
    (0x0010, 0x1000): REMOVE,   # Other Patient IDs
    (0x0010, 0x1001): REMOVE,   # Other Patient Names
    (0x0010, 0x1040): REMOVE,   # Patient Address
    (0x0010, 0x2154): REMOVE,   # Patient Telephone Numbers
    (0x0010, 0x4000): REMOVE,   # Patient Comments
    (0x0018, 0x1000): REMOVE,   # Device Serial Number
    (0x0020, 0x000D): HASH_UID,  # Study Instance UID
    (0x0020, 0x000E): HASH_UID,  # Series Instance UID
    (0x0020, 0x0010): BLANK,    # Study ID
    (0x0020, 0x0052): HASH_UID,  # Frame of Reference UID
    (0x0020, 0x4000): REMOVE,   # Image Comments
    (0x0040, 0xA730): REMOVE,   # Content Sequence (structured reports)
}

# Tags that are clinically useful and explicitly kept even though they sit in
# groups we otherwise touch. (Documented so the policy is auditable.)
RETAINED_TAGS: Dict[Tuple[int, int], str] = {
    (0x0010, 0x1010): "Patient Age",
    (0x0010, 0x1020): "Patient Size",
    (0x0010, 0x1030): "Patient Weight",
    (0x0008, 0x0060): "Modality",
    (0x0018, 0x6011): "Sequence of Ultrasound Regions",
}


@dataclass
class DeidOptions:
    """Configuration for header de-identification."""

    date_policy: str = "shift"            # "keep" | "blank" | "shift"
    date_shift_days: Optional[int] = None  # if None, derived per-patient
    remove_private_tags: bool = True
    blank_all_person_names: bool = True   # VR == 'PN' catch-all
    keep_patient_sex: bool = False        # mother's sex; off by default
    pseudonymise_clinic: bool = True      # store hashed clinic id in a kept tag
    dummy_id_prefix: str = "ANON"
    audit_log: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.audit_log.append(msg)


def _stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length].upper()


def _remap_uid(original: str) -> str:
    """Deterministically map an original UID to a new valid UID.

    Same input -> same output, so referential links inside a study survive.
    """
    if not original:
        return generate_uid(prefix=PCTK_UID_ROOT)
    return generate_uid(prefix=PCTK_UID_ROOT, entropy_srcs=[str(original)])


def _shift_days_for(ds: Dataset, opts: DeidOptions) -> int:
    if opts.date_shift_days is not None:
        return opts.date_shift_days
    seed = str(getattr(ds, "PatientID", "")) or str(getattr(ds, "PatientName", ""))
    # Deterministic offset in [-364, 364] derived from the patient identifier.
    h = int(_stable_hash(seed or "anon", 8), 16)
    return (h % 729) - 364


def _apply_date(ds: Dataset, tag: Tuple[int, int], opts: DeidOptions) -> None:
    if tag not in ds:
        return
    if opts.date_policy == "keep":
        return
    if opts.date_policy == "blank":
        ds[tag].value = ""
        return
    # shift
    raw = str(ds[tag].value or "").strip()
    if len(raw) != 8 or not raw.isdigit():
        ds[tag].value = ""
        return
    try:
        d = datetime.strptime(raw, "%Y%m%d")
        d2 = d + timedelta(days=_shift_days_for(ds, opts))
        ds[tag].value = d2.strftime("%Y%m%d")
    except ValueError:
        ds[tag].value = ""


def deidentify_dataset(ds: "Dataset", opts: Optional[DeidOptions] = None) -> "Dataset":
    """De-identify a pydicom :class:`Dataset` in place and return it."""
    opts = opts or DeidOptions()

    # 1) Curated action table.
    for tag, action in PHI_ACTIONS.items():
        if tag not in ds:
            continue
        if action == REMOVE:
            del ds[tag]
            opts.log(f"removed {tag}")
        elif action == BLANK:
            ds[tag].value = ""
            opts.log(f"blanked {tag}")
        elif action == DUMMY:
            if tag == (0x0010, 0x0010):  # Patient Name
                pid = str(getattr(ds, "PatientID", "")) or "anon"
                ds[tag].value = f"{opts.dummy_id_prefix}^{_stable_hash(pid)}"
            elif tag == (0x0010, 0x0020):  # Patient ID
                pid = str(ds[tag].value or "anon")
                ds[tag].value = f"{opts.dummy_id_prefix}-{_stable_hash(pid)}"
            else:
                ds[tag].value = opts.dummy_id_prefix
            opts.log(f"dummied {tag}")
        elif action == HASH_UID:
            ds[tag].value = _remap_uid(str(ds[tag].value))
            opts.log(f"remapped UID {tag}")
        elif action == DATE:
            _apply_date(ds, tag, opts)
            opts.log(f"date-policy({opts.date_policy}) {tag}")

    # 2) Optionally retain a pseudonymous clinic id before institution is gone.
    #    (Done above already blanked InstitutionName; store hash in StationName
    #     slot only if requested and value existed.)
    # 3) Patient sex (mother) handling.
    if not opts.keep_patient_sex and (0x0010, 0x0040) in ds:
        ds[(0x0010, 0x0040)].value = ""
        opts.log("blanked PatientSex")

    # 4) Catch-all: blank every Person Name VR we did not explicitly handle.
    #    (Tags in PHI_ACTIONS were already given their intended treatment -
    #    e.g. PatientName keeps its pseudonymous dummy - so skip them here.)
    if opts.blank_all_person_names:
        for elem in ds:
            try:
                tag = (elem.tag.group, elem.tag.element)
                if tag in PHI_ACTIONS:
                    continue
                if elem.VR == "PN" and elem.value not in ("", None):
                    elem.value = ""
                    opts.log(f"blanked PN {elem.tag}")
            except Exception:
                continue

    # 5) Private tags.
    if opts.remove_private_tags:
        ds.remove_private_tags()
        opts.log("removed private tags")

    # 6) Mark the dataset as de-identified per standard.
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "pctk PS3.15 basic subset v" + _toolkit_version()

    return ds


def deidentify_file(
    in_path: str,
    out_path: str,
    opts: Optional[DeidOptions] = None,
) -> DeidOptions:
    """Read a DICOM file, de-identify its header, and write a new file.

    Returns the :class:`DeidOptions` (whose ``audit_log`` lists every action).
    Note: this only touches the *header*. Burned-in pixel text must be removed
    separately with :mod:`pctk.deid.pixel_text`.
    """
    opts = opts or DeidOptions()
    ds = pydicom.dcmread(in_path)
    deidentify_dataset(ds, opts)
    ds.save_as(out_path)
    return opts


def _toolkit_version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "0"
