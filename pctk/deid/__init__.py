"""Problem 1 - De-identification of prenatal ultrasound studies.

Two complementary layers:

* ``dicom_scrub`` removes / replaces patient-identifying DICOM *header* tags
  (DICOM PS3.15 Basic Application Level Confidentiality Profile, practical
  subset) while preserving clinically useful, non-identifying attributes.

* ``pixel_text`` finds and redacts patient-identifying text *burned into the
  pixels* of an ultrasound frame (name, MRN, clinic, date overlaid by the
  machine), using a pluggable OCR backend with a deterministic margin-based
  fallback when no OCR engine is installed.

Neither layer looks at fetal anatomy.
"""

from .dicom_scrub import (
    DeidOptions,
    deidentify_dataset,
    deidentify_file,
)
from .pixel_text import (
    RedactOptions,
    TextBox,
    redact_image_file,
    redact_array,
)

__all__ = [
    "DeidOptions",
    "deidentify_dataset",
    "deidentify_file",
    "RedactOptions",
    "TextBox",
    "redact_image_file",
    "redact_array",
]
