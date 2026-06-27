"""Ingest a PCPNDT Form F from a PDF or scanned image into a :class:`FormF`.

Handles three input kinds transparently:

  * **digital PDF**  - text extracted directly with ``pypdf`` (no OCR needed),
  * **scanned PDF**  - pages rasterised with ``PyMuPDF`` then OCR'd,
  * **image**        - PNG/JPG of a form, OCR'd directly.

Field values are pulled out with label-aliased regexes. Crucially, the
pregnant woman's **name is never stored** - it is converted to a one-way
pseudonymous ``patient_ref`` hash, consistent with this toolkit's privacy
stance and the PCPNDT requirement to keep records auditable without exposing
the woman to sex-selection pressure.

Optional dependencies degrade gracefully:
  * no ``pypdf``  -> digital-PDF text extraction unavailable,
  * no ``fitz``   -> scanned-PDF rasterisation unavailable,
  * no OCR engine -> image / scanned-PDF text extraction unavailable.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .formf import FormF
from ..deid import ocr_backends


# Label aliases per field -> regex value capture (rest of line).
_FIELD_PATTERNS: Dict[str, List[str]] = {
    "_patient_name": [
        r"name of (?:the )?(?:pregnant woman|patient|woman)",
        r"patient\s*name",
        r"name of woman",
    ],
    "referring_doctor": [
        r"referred by", r"referring (?:doctor|physician|practitioner|medical)",
        r"name of (?:the )?referring",
    ],
    "indication": [
        # most specific first so the trailing noun isn't captured as the value
        r"indication(?:s)? for (?:the )?(?:ultrasound|usg|scan|test|procedure|"
        r"sonography|examination)",
        r"reason for (?:the )?(?:scan|test|procedure|usg|ultrasound)",
        r"purpose of (?:the )?(?:scan|test|ultrasound|procedure)",
        r"indication(?:s)? for", r"indication",
    ],
    "performed_by": [
        r"performed by", r"conducted by", r"sonologist", r"radiologist",
        r"name of (?:the )?(?:person|doctor) (?:performing|conducting)",
    ],
    "machine_id": [
        r"machine (?:id|no\.?|number|sl\.? no)", r"usg machine",
        r"equipment (?:id|no\.?|number)", r"ultrasound machine",
    ],
    "facility_id": [
        r"(?:centre|center|clinic|facility) (?:registration|reg\.?|id|code|no\.?)",
        r"registration (?:no\.?|number)", r"genetic clinic registration",
    ],
}

_GA_PATTERNS = [
    r"gestation(?:al)?\s*age[^0-9]{0,12}(\d{1,2})\s*(?:weeks|wks|w)\b",
    r"period of gestation[^0-9]{0,12}(\d{1,2})\s*(?:weeks|wks|w)\b",
    r"\bGA[^0-9]{0,6}(\d{1,2})\s*(?:weeks|wks|w)\b",
    r"(\d{1,2})\s*weeks(?:\s*\d{1,2}\s*days)?",
]
_DATE_PATTERNS = [
    r"date of (?:scan|test|examination|procedure|ultrasound)[^0-9]{0,12}"
    r"(\d{1,2}[-/][A-Za-z0-9]{2,3}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})",
    r"\bdate\b[^0-9]{0,12}(\d{1,2}[-/][A-Za-z0-9]{2,3}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})",
]


def _hash_ref(name: str, prefix: str = "PT") -> str:
    h = hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{h}"


def _parse_date(s: str):
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d/%b/%Y",
                "%d-%B-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def _pdf_digital_text(path: str) -> str:
    try:
        import pypdf
    except Exception:
        return ""
    try:
        reader = pypdf.PdfReader(path)
        return "\n".join((pg.extract_text() or "") for pg in reader.pages)
    except Exception:
        return ""


def _pdf_render_pages(path: str, dpi: int = 200) -> List[np.ndarray]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    pages = []
    doc = fitz.open(path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for pg in doc:
        pix = pg.get_pixmap(matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = arr[:, :, :3]
        pages.append(np.ascontiguousarray(arr))
    doc.close()
    return pages


def extract_text(path: str, ocr_backend: Optional[str] = None,
                 min_digital_chars: int = 40) -> str:
    """Return the text of a Form F, choosing the right strategy by file type."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = _pdf_digital_text(path)
        if len(text.strip()) >= min_digital_chars:
            return text
        # Likely a scanned PDF -> rasterise + OCR.
        backend = ocr_backends.get_backend(ocr_backend)
        chunks = [ocr_backends.image_to_lines(p, backend=backend)
                  for p in _pdf_render_pages(path)]
        return "\n".join(c for c in chunks if c)
    # Image input.
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB"))
    return ocr_backends.image_to_lines(arr, prefer=ocr_backend)


# --------------------------------------------------------------------------- #
# Field parsing
# --------------------------------------------------------------------------- #
def _match_label(text: str, aliases: List[str]) -> Optional[str]:
    for alias in aliases:
        m = re.search(alias + r"\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # stop at an obvious next-label boundary
            val = re.split(r"\s{2,}|\s+(?:name|date|indication|referred)\s*[:\-]",
                           val, maxsplit=1)[0].strip()
            # drop a leading "No.:" / "Number:" / "Reg No:" left over from a
            # label like "Centre Registration No: FAC-A"
            val = re.sub(r"^(?:reg\.?\s*)?(?:no\.?|number|id|code)\s*[:\-.]?\s*",
                         "", val, flags=re.IGNORECASE).strip()
            if val:
                return val
    return None


def parse_form_f_text(text: str,
                      facility_id: Optional[str] = None
                      ) -> Tuple[FormF, Dict[str, str]]:
    """Parse raw Form-F text into a :class:`FormF` plus the raw captured map."""
    raw: Dict[str, str] = {}
    form = FormF()

    for field, aliases in _FIELD_PATTERNS.items():
        val = _match_label(text, aliases)
        if not val:
            continue
        raw[field] = val
        if field == "_patient_name":
            form.patient_ref = _hash_ref(val)   # name -> pseudonym, never stored
        else:
            setattr(form, field, val)

    for pat in _GA_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                form.gestational_age_weeks = float(m.group(1))
                raw["gestational_age_weeks"] = m.group(1)
            except ValueError:
                pass
            break

    for pat in _DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            d = _parse_date(m.group(1))
            if d:
                form.scan_date = d
                raw["scan_date"] = m.group(1)
            break

    if facility_id and not form.facility_id:
        form.facility_id = facility_id
    return form, raw


def ingest_form_f(path: str,
                  facility_id: Optional[str] = None,
                  ocr_backend: Optional[str] = None,
                  record_id: Optional[str] = None
                  ) -> Tuple[FormF, Dict[str, str]]:
    """Full pipeline: file -> text -> parsed :class:`FormF`."""
    text = extract_text(path, ocr_backend=ocr_backend)
    form, raw = parse_form_f_text(text, facility_id=facility_id)
    form.record_id = record_id or os.path.splitext(os.path.basename(path))[0]
    return form, raw
