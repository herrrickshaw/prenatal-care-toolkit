"""Deterministic snapshot of observable behaviour (synthetic data only).

``collect()`` exercises the modules covered by the SDLC refactor and returns a
JSON-serialisable dict. ``tests/golden/baseline.json`` was produced by running
it against the pre-refactor code; ``tests/test_characterisation.py`` asserts
the current code reproduces it exactly.

    python tests/characterise.py [repo_root] > out.json
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from datetime import date


def _setup(root):
    if root:
        sys.path.insert(0, root)


# --------------------------------------------------------------------------- #
class FakeBackend:
    """OCR stand-in returning fixed boxes (scaled with the image size)."""
    name = "fake"

    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, img, min_conf=0.3):
        h, w = img.shape[:2]
        return [b for b in self._boxes(w, h) if b[5] >= min_conf]


def _boxes(w, h):
    return [
        (int(w * 0.05), int(h * 0.02), int(w * 0.30), int(h * 0.05), "PRIYA SHARMA", 0.9),
        (int(w * 0.40), int(h * 0.50), int(w * 0.10), int(h * 0.05), "centre", 0.8),
        (int(w * 0.10), int(h * 0.93), int(w * 0.30), int(h * 0.04), "SUNRISE CLINIC", 0.7),
        (int(w * 0.60), int(h * 0.60), int(w * 0.10), int(h * 0.04), "lowconf", 0.1),
    ]


def _image():
    import numpy as np
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, size=(120, 160, 3)).astype("uint8")


def _digest(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def pixel_cases():
    from pctk.deid import pixel_text as pt
    from pctk.deid import ocr_backends as ob
    out = {}
    arr = _image()
    orig_get = ob.get_backend
    try:
        for label, backend in (("fake", FakeBackend(_boxes)), ("none", None)):
            ob.get_backend = lambda prefer=None, languages=None, _b=backend: _b
            for method in ("blackout", "blur", "pixelate"):
                for kw in ({}, {"margin_only": False}, {"also_redact_bands": True},
                           {"upscale": 1.0}, {"fallback_bands": False}):
                    opts = pt.RedactOptions(method=method, **kw)
                    res, boxes = pt.redact_array(arr, opts)
                    out[f"{label}|{method}|{sorted(kw.items())}"] = {
                        "sha": _digest(res), "boxes": [list(b) for b in boxes],
                        "log": [_tag_text(x) for x in opts.audit_log]}
        # grayscale / float input normalisation
        ob.get_backend = lambda prefer=None, languages=None: None
        gray = (arr[:, :, 0].astype("float64") / 7.0)
        res, boxes = pt.redact_array(gray, pt.RedactOptions())
        out["gray_float"] = {"sha": _digest(res), "shape": list(res.shape)}
        try:
            pt.redact_array(arr, pt.RedactOptions(method="bogus"))
            out["bad_method"] = "no-error"
        except ValueError as exc:
            out["bad_method"] = str(exc)
        # image_to_lines with fake backend
        ob.get_backend = orig_get
        out["lines"] = ob.image_to_lines(arr, backend=FakeBackend(_boxes), min_conf=0.2)
        out["lines_empty"] = ob.image_to_lines(arr, backend=FakeBackend(lambda w, h: []))
        out["rgb_gray"] = _digest(ob._to_rgb_uint8(arr[:, :, 0]))
        out["rgb_float"] = _digest(ob._to_rgb_uint8(arr.astype("float64")))
        out["rgb_rgba"] = list(ob._to_rgb_uint8(
            __import__("numpy").dstack([arr, arr[:, :, :1]])).shape)
        out["backends_type"] = isinstance(ob.available_backends(), list)
        out["get_backend_unknown"] = ob.get_backend("nonexistent")
    finally:
        ob.get_backend = orig_get
    return out


def _dicom_ds():
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PatientName = "SHARMA^PRIYA"
    ds.PatientID = "MRN12345"
    ds.PatientBirthDate = "19980101"
    ds.PatientSex = "F"
    ds.PatientAge = "027Y"
    ds.Modality = "US"
    ds.StudyDate = "20260301"
    ds.SeriesDate = "20260301"
    ds.AcquisitionDate = "bad"
    ds.ContentDate = "20260229"
    ds.StudyTime = "101500"
    ds.AccessionNumber = "ACC77"
    ds.InstitutionName = "SUNRISE CLINIC"
    ds.InstitutionAddress = "1 Road"
    ds.ReferringPhysicianName = "MEHTA^A"
    ds.PerformingPhysicianName = "RAO^S"
    ds.OperatorsName = "OP^ONE"
    ds.StationName = "ST1"
    ds.StudyDescription = "Obstetric for Priya"
    ds.DeviceSerialNumber = "SN-1"
    ds.ImageComments = "note"
    ds.PatientComments = "pc"
    ds.StudyInstanceUID = "1.2.3.4"
    ds.SeriesInstanceUID = "1.2.3.5"
    ds.SOPInstanceUID = "1.2.3.6"
    ds.FrameOfReferenceUID = "1.2.3.7"
    ds.StudyID = "S1"
    ds.OtherPatientNames = ["X^Y"]
    ds.RequestingPhysician = "DR^REQ"            # PN not in table
    ds.add_new((0x0009, 0x0010), "LO", "PRIV")
    ds.add_new((0x0009, 0x1001), "LO", "secret")
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    return ds


_TAG_RE = re.compile(r"\(([0-9A-Fa-f]{4}), ?([0-9A-Fa-f]{4})\)")


def _tag_text(text):
    """pydicom 2.x prints "(0020, 000d)", 3.x "(0020,000D)": normalise to the 3.x form."""
    return _TAG_RE.sub(lambda m: "(%s,%s)" % (m.group(1).upper(), m.group(2).upper()), text)


def _tag(tag):
    return _tag_text(str(tag))


def _dump(ds):
    rows = []
    for el in ds:
        rows.append([_tag(el.tag), el.VR, str(el.value)])
    return rows


def dicom_cases():
    from pctk.deid.dicom_scrub import DeidOptions, deidentify_dataset
    out = {}
    variants = {
        "default": {}, "keep": {"date_policy": "keep"},
        "blank": {"date_policy": "blank"},
        "fixed_shift": {"date_shift_days": 10},
        "keep_priv_sex": {"remove_private_tags": False, "keep_patient_sex": True},
        "no_pn": {"blank_all_person_names": False},
        "prefix": {"dummy_id_prefix": "ZZ"},
    }
    for name, kw in variants.items():
        opts = DeidOptions(**kw)
        ds = deidentify_dataset(_dicom_ds(), opts)
        out[name] = {"elements": _dump(ds), "log": [_tag_text(x) for x in opts.audit_log]}
    ds = _dicom_ds()
    del ds.PatientID
    del ds.PatientName
    opts = DeidOptions()
    out["no_ids"] = {"elements": _dump(deidentify_dataset(ds, opts)),
                     "log": [_tag_text(x) for x in opts.audit_log]}
    return out


def compliance_cases():
    from pctk.compliance import FormF, MachineRegistration, audit_records, validate_form_f
    regs = [MachineRegistration("US-100", "FAC-A", date(2023, 1, 1)),
            MachineRegistration("US-200", "FAC-B", date(2020, 1, 1), date(2024, 6, 30)),
            MachineRegistration("US-100", "FAC-A", date(2019, 1, 1), date(2019, 12, 31))]
    full = dict(patient_ref="P", performed_by="T", referring_doctor="Dr", gestational_age_weeks=12)
    forms = [
        FormF(record_id="ok", facility_id="FAC-A", machine_id="US-100",
              scan_date=date(2025, 3, 10), indication="dating scan", **full),
        FormF(record_id="vague", facility_id="FAC-A", machine_id="US-100",
              scan_date=date(2025, 3, 10), indication=" Routine ", **full),
        FormF(record_id="unreg", facility_id="FAC-A", machine_id="US-900",
              scan_date=date(2025, 3, 10), indication="x", **full),
        FormF(record_id="lapsed", facility_id="FAC-B", machine_id="US-200",
              scan_date=date(2024, 9, 1), indication="x", **full),
        FormF(record_id="old", facility_id="FAC-B", machine_id="US-200",
              scan_date=date(2021, 5, 5), indication="x", **full),
        FormF(record_id="noga", facility_id="", machine_id="US-100",
              scan_date=None, indication="x", patient_ref="P", performed_by="T",
              referring_doctor="", gestational_age_weeks=None),
        FormF(record_id="ga", facility_id="FAC-A", machine_id="US-100",
              scan_date=date(2025, 1, 1), indication="x", patient_ref="P",
              performed_by="T", referring_doctor="D", gestational_age_weeks=60),
        FormF(record_id="ga2", facility_id="FAC-A", machine_id="US-100",
              scan_date=date(2025, 1, 1), indication="x", patient_ref="P",
              performed_by="T", referring_doctor="D", gestational_age_weeks=2),
    ]
    out = {}
    for ret in (2, 1, 5):
        res = audit_records(forms, regs, today=date(2026, 6, 1), retention_years=ret)
        out[f"ret{ret}"] = {
            "findings": [[f.record_id, f.facility_id, f.rule, f.severity, f.detail]
                         for f in res["findings"]],
            "summary": res["facility_summary"], "counts": res["counts"]}
    out["empty"] = audit_records([], [], today=date(2026, 6, 1))["counts"]
    out["validate"] = [[f.rule, f.detail] for f in validate_form_f(forms[5])]
    out["missing"] = forms[5].missing_fields()
    return out


FORM_TEXTS = [
    ("FORM F (PCPNDT Act, 1994)\nName of pregnant woman: PRIYA SHARMA\n"
     "Date of test: 27-06-2026\nReferred by: Dr A Mehta\n"
     "Indication for ultrasound: Anomaly scan\nPeriod of gestation: 19 weeks\n"
     "Performed by: Dr S Rao\nUSG Machine No: US-100\n"
     "Centre Registration No: FAC-A\n"),
    "Patient Name: Asha Rao\nDate: 2025-03-10\nGestational age: 12 wks\n"
    "Reason for scan: dating  Referred by: Dr Z\nSonologist: Dr Q\n"
    "Equipment ID: M7\nFacility code: F9\n",
    "GA 22 weeks 3 days\nDate of scan 5/Mar/2024\nIndication: -\n",
    "nothing useful here",
    "",
]


def ingest_cases():
    from pctk.compliance import parse_form_f_text
    from pctk.compliance import ingest
    out = []
    for t in FORM_TEXTS:
        form, raw = parse_form_f_text(t, facility_id="DEFAULT-FAC")
        out.append({"form": {k: str(v) for k, v in vars(form).items()}, "raw": raw})
    out.append({"hash": ingest._hash_ref("  Priya SHARMA "), "hash2": ingest._hash_ref("x", "ZZ")})
    out.append({"dates": [str(ingest._parse_date(s)) for s in
                          ("2026-06-27", "27-06-2026", "27/06/2026", "27-Jun-2026",
                           "27/Jun/2026", "27-June-2026", "20260627", " 2026-06-27 ", "junk")]})
    out.append({"label": [ingest._match_label("Machine No: No. US-1  Name: x", ["machine no"]),
                          ingest._match_label("Centre Registration No: FAC-A", ["registration no"]),
                          ingest._match_label("nothing", ["zzz"])]})
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "scan_001.pdf")
        open(p, "wb").write(b"%PDF-1.4 not really")
        out.append({"pdf_digital_bad": ingest._pdf_digital_text(p)})
        orig = ingest.extract_text
        ingest.extract_text = lambda path, ocr_backend=None, **kw: FORM_TEXTS[0]
        try:
            form, raw = ingest.ingest_form_f(p, facility_id="F")
            out.append({"record_id": form.record_id, "ref": form.patient_ref})
            form, raw = ingest.ingest_form_f(p, record_id="R1")
            out.append({"record_id": form.record_id})
        finally:
            ingest.extract_text = orig
    return out


def _run_cli(argv):
    from pctk import cli
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = "exit:%s" % exc.code
    return {"code": code, "out": buf.getvalue(), "err": err.getvalue()}


def cli_cases():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        def P(name):
            return os.path.join(td, name)

        def clean(d):
            return {k: (_tag_text(v.replace(td, "<TMP>")) if isinstance(v, str) else v)
                    for k, v in d.items()}

        with open(P("forms.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["record_id", "patient_ref", "facility_id", "machine_id", "scan_date",
                        "referring_doctor", "indication", "gestational_age_weeks", "performed_by"])
            w.writerow(["r1", "P1", "FAC-A", "US-100", "2025-03-10", "Dr", "dating", "12", "T"])
            w.writerow(["", "P2", "FAC-A", "US-900", "10/03/2025", "", "routine", "", "T"])
            w.writerow(["r3", "P3", "FAC-B", "US-200", "garbage", "Dr", "x", "70", "T"])
        with open(P("regs.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["machine_id", "facility_id", "registered_from", "registered_to"])
            w.writerow(["US-100", "FAC-A", "2023-01-01", ""])
            w.writerow(["US-200", "FAC-B", "", "2024-06-30"])
        out["compliance"] = clean(_run_cli(["compliance", P("forms.csv"), P("regs.csv")]))
        out["compliance_v"] = clean(_run_cli(["compliance", P("forms.csv"), P("regs.csv"), "-v"]))

        rows = []
        for d, m, f in (("A", 60, 40), ("B", 55, 55), ("C", 80, 30), ("D", 2, 1)):
            rows += [(d, "M")] * m + [(d, "F")] * f
        with open(P("births.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["district", "sex"])
            w.writerows(rows)
        out["srb"] = clean(_run_cli(["srb", P("births.csv"), "--out", P("o.csv")]))
        out["srb_file"] = open(P("o.csv")).read()
        with open(P("agg.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["State", "District", "M_06", "F_06"])
            w.writerows([("S", "A", 1000, 800), ("S", "B", 1000, 960), ("S", "C", 10, 10)])
        out["srb_child06"] = clean(_run_cli(["srb", P("agg.csv"), "--source", "child06"]))
        out["srb_males"] = clean(_run_cli(["srb", P("agg.csv"), "--males-col", "M_06",
                                           "--females-col", "F_06", "--group", "District"]))
        with open(P("ratio.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["District", "Sex Ratio at Birth"])
            w.writerows([("A", 900), ("B", 950)])
        out["srb_ratio"] = clean(_run_cli(["srb", P("ratio.csv"), "--source", "ratio",
                                           "--group", "District", "--out", P("r.csv")]))
        out["srb_ratio_file"] = open(P("r.csv")).read()

        out["biometry"] = _run_cli(["biometry", "--bpd", "46", "--hc", "170", "--ac", "150", "--fl", "30"])
        out["biometry_partial"] = _run_cli(["biometry", "--bpd", "46"])
        out["version"] = _run_cli(["--version"])
        out["nocmd"] = _run_cli([])["code"]

        from pctk.compliance import ingest
        orig = ingest.extract_text
        ingest.extract_text = lambda path, ocr_backend=None, **kw: FORM_TEXTS[0]
        try:
            out["ingest"] = clean(_run_cli(["ingest-formf", P("scan_9.pdf"), "--append-csv", P("a.csv")]))
            out["ingest2"] = clean(_run_cli(["ingest-formf", P("scan_9.pdf"), "--append-csv", P("a.csv"),
                                             "--facility-id", "X"]))
            out["append_file"] = open(P("a.csv")).read()
        finally:
            ingest.extract_text = orig

        # deid-dicom end to end on a synthetic file
        ds = _dicom_ds()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.6"
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.AcquisitionDate = "20260301"
        ds.ContentDate = "20260301"
        try:
            ds.save_as(P("in.dcm"), enforce_file_format=True)   # pydicom >= 3
        except TypeError:
            ds.save_as(P("in.dcm"), write_like_original=False)  # pydicom 2.x equivalent
        r = _run_cli(["deid-dicom", P("in.dcm"), P("out.dcm"), "-v", "--date-policy", "blank"])
        out["deid_dicom"] = clean(r)
        import pydicom
        out["deid_dicom_elements"] = _dump(pydicom.dcmread(P("out.dcm")))

        import numpy as np
        from PIL import Image
        Image.fromarray(_image()).save(P("in.png"))
        # The baseline was captured with no OCR engine installed; CI runners have
        # tesseract, so pin "no backend" here to keep the test machine-independent.
        from pctk.deid import ocr_backends as _ob
        _orig_get = _ob.get_backend
        _ob.get_backend = lambda prefer=None, languages=None: None
        try:
            out["deid_image"] = clean(_run_cli(["deid-image", P("in.png"), P("o.png"),
                                                "--also-redact-bands", "--method", "pixelate"]))
        finally:
            _ob.get_backend = _orig_get
        out["deid_image_sha"] = _digest(np.asarray(Image.open(P("o.png"))))
        out["backends"] = _run_cli(["backends"])["code"]
    return out


def srb_cases():
    import pandas as pd
    from pctk.srb import adapters
    out = {}
    df = pd.DataFrame({"State": ["S", "S"], "District": ["A", "B"],
                       "Male": [100, 200], "Female": [80, 190]})
    out["crs"] = adapters.from_crs_hmis(df).to_dict("list")
    out["crs_map"] = adapters.from_crs_hmis(
        df.rename(columns={"Male": "x", "Female": "y"}),
        column_map={"males": "x", "females": "y"}).to_dict("list")
    r = pd.DataFrame({"District": ["A", "B"], "SRB": [900, 950]})
    out["ratio"] = adapters.from_ratio_only(r).to_dict("list")
    out["ratio_m"] = adapters.from_ratio_only(r, ratio_kind="m_per_100_f").to_dict("list")
    out["census"] = adapters.from_census_csr(
        pd.DataFrame({"District": ["A"], "Child Sex Ratio": [880]})).to_dict("list")
    nf = pd.DataFrame({"State": ["S"] * 3, "District": ["A", "A", "B"],
                       "Indicator": ["Sex ratio at birth for children born in the last five years (females per 1,000 males)",
                                     "Other", "Sex ratio at birth (females per 1,000 males)"],
                       "NFHS-5": [910, 5, 940]})
    out["nfhs"] = adapters.from_nfhs(nf).to_dict("list")
    c6 = pd.DataFrame({"State": ["S", "S"], "District": ["A", "B"], "M_06": [1000, 900], "F_06": [800, 880]})
    out["child06"] = adapters.from_census_child_06(c6).to_dict("list")
    for name, call in (("err_crs", lambda: adapters.from_crs_hmis(r)),
                       ("err_unknown", lambda: adapters.adapt("zzz", r)),
                       ("err_nfhs", lambda: adapters.from_nfhs(r)),
                       ("err_c6", lambda: adapters.from_census_child_06(r)),
                       ("err_census", lambda: adapters.from_census_csr(c6))):
        try:
            call()
            out[name] = "no-error"
        except ValueError as exc:
            out[name] = str(exc)
    return out


def collect():
    return {
        "pixel": pixel_cases(), "dicom": dicom_cases(),
        "compliance": compliance_cases(), "ingest": ingest_cases(),
        "cli": cli_cases(), "srb": srb_cases(),
    }


if __name__ == "__main__":
    _setup(sys.argv[1] if len(sys.argv) > 1 else None)
    import pctk
    sys.stderr.write("pctk from %s\n" % pctk.__file__)
    json.dump(collect(), sys.stdout, indent=1, sort_keys=True, default=str)
