"""Generate synthetic sample data for testing the toolkit.

Creates, under ``data/``:
  * ``sample_ultrasound.dcm`` - a tiny synthetic DICOM with realistic PHI tags
    and a small grayscale "ultrasound" pixel array with burned-in text bands.
  * ``sample_ultrasound.png`` - a standalone image with patient text overlaid
    in the corners (for pixel-text redaction tests).

No real patient data is used or required.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")


def _fake_ultrasound(w=512, h=384) -> np.ndarray:
    """A vaguely ultrasound-looking grayscale image (cone + speckle)."""
    rng = np.random.default_rng(7)
    img = (rng.random((h, w)) * 40).astype(np.uint8)  # speckle background
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2.0
    # Fan/cone of brighter tissue in the centre.
    cone = (np.abs(xx - cx) < (yy * 0.6 + 30)) & (yy > 20)
    img[cone] = np.clip(img[cone] + 120 + (rng.random(img[cone].shape) * 60), 0, 255)
    return img.astype(np.uint8)


def _burn_text(img: np.ndarray) -> np.ndarray:
    """Overlay identifying text into the corners/margins, like a US machine."""
    pil = Image.fromarray(img).convert("L")
    draw = ImageDraw.Draw(pil)
    draw.text((6, 4), "NAME: PRIYA SHARMA", fill=255)
    draw.text((6, 16), "MRN: 99213  AGE: 27Y", fill=255)
    draw.text((6, 28), "SUNRISE DIAGNOSTICS, PUNE", fill=255)
    w = pil.size[0]
    draw.text((w - 150, 4), "27-JUN-2026 10:42", fill=255)
    draw.text((6, pil.size[1] - 14), "DR A MEHTA  GA 19w2d", fill=255)
    return np.asarray(pil)


def make_png() -> str:
    arr = _burn_text(_fake_ultrasound())
    path = os.path.join(DATA, "sample_ultrasound.png")
    Image.fromarray(arr).save(path)
    return path


def make_dicom() -> str:
    arr = _burn_text(_fake_ultrasound())

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"  # US Image
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    path = os.path.join(DATA, "sample_ultrasound.dcm")
    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # --- PHI we expect to be scrubbed ---
    ds.PatientName = "SHARMA^PRIYA"
    ds.PatientID = "MRN-99213"
    ds.PatientBirthDate = "19990115"
    ds.PatientSex = "F"
    ds.PatientAge = "027Y"
    ds.InstitutionName = "Sunrise Diagnostics"
    ds.InstitutionAddress = "MG Road, Pune"
    ds.ReferringPhysicianName = "MEHTA^ANAND"
    ds.PerformingPhysicianName = "RAO^S"
    ds.OperatorsName = "TECH^1"
    ds.StationName = "US-ROOM-2"
    ds.DeviceSerialNumber = "SN-7782211"
    ds.AccessionNumber = "ACC-554433"
    ds.StudyDate = "20260627"
    ds.SeriesDate = "20260627"
    ds.StudyID = "ST-1"

    # --- clinically useful, should survive ---
    ds.Modality = "US"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # --- pixel data ---
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path)
    return path


def make_births_csv() -> str:
    """Synthetic district births: most natural, two districts skewed male."""
    import csv as _csv

    rng = np.random.default_rng(11)
    districts = {
        "Anandpur": 0.488, "Bagh": 0.487, "Chandni": 0.489, "Devnagar": 0.486,
        "Erode": 0.490, "Fatehpur": 0.420,   # skewed
        "Gokul": 0.488, "Hampi": 0.405,      # skewed
    }
    path = os.path.join(DATA, "births.csv")
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["district", "sex", "year"])
        for dist, pf in districts.items():
            n = rng.integers(400, 900)
            for _ in range(int(n)):
                sex = "F" if rng.random() < pf else "M"
                w.writerow([dist, sex, 2025])
    return path


def make_compliance_csvs():
    """Synthetic Form-F records + machine registrations with seeded issues."""
    import csv as _csv

    forms = os.path.join(DATA, "forms.csv")
    regs = os.path.join(DATA, "registrations.csv")

    with open(regs, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["machine_id", "facility_id", "registered_from", "registered_to"])
        w.writerow(["US-100", "FAC-A", "2023-01-01", ""])           # valid
        w.writerow(["US-200", "FAC-B", "2022-01-01", "2024-12-31"])  # lapsed 2025+
        # US-900 deliberately absent -> unregistered machine

    rows = [
        # record_id, facility, machine, scan_date, ref_doctor, indication, ga, by
        ["r1", "FAC-A", "US-100", "2025-03-10", "Dr Rao", "dating scan", "12", "Tech1"],
        ["r2", "FAC-A", "US-100", "2025-03-11", "Dr Rao", "routine", "20", "Tech1"],   # vague
        ["r3", "FAC-A", "US-100", "2025-03-12", "", "anomaly scan", "19", "Tech1"],     # missing ref
        ["r4", "FAC-B", "US-200", "2025-06-01", "Dr Sen", "growth scan", "28", "Tech2"],# lapsed reg
        ["r5", "FAC-B", "US-900", "2025-06-02", "Dr Sen", "dating scan", "11", "Tech2"],# unregistered
        ["r6", "FAC-A", "US-100", "2021-02-01", "Dr Rao", "dating scan", "10", "Tech1"],# retention expired
        ["r7", "FAC-A", "US-100", "2025-03-15", "Dr Rao", "anomaly scan", "60", "Tech1"],# GA out of range
    ]
    with open(forms, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["record_id", "facility_id", "machine_id", "scan_date",
                    "referring_doctor", "indication", "gestational_age_weeks",
                    "performed_by", "patient_ref"])
        for i, r in enumerate(rows):
            w.writerow(r + [f"PT-{i+1:03d}"])
    return forms, regs


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    png = make_png()
    dcm = make_dicom()
    births = make_births_csv()
    forms, regs = make_compliance_csvs()
    for label, p in [("png", png), ("dcm", dcm), ("births", births),
                     ("forms", forms), ("registrations", regs)]:
        print(f"wrote {label}:", p)


if __name__ == "__main__":
    main()
