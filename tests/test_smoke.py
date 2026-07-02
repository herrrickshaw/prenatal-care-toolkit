"""Smoke tests for the prenatal-care-toolkit.

Runs with pytest *or* as a plain script:  python tests/test_smoke.py
Regenerates sample data if missing.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")


def _ensure_data():
    if not os.path.exists(os.path.join(DATA, "sample_ultrasound.dcm")):
        sys.path.insert(0, os.path.join(ROOT, "examples"))
        import make_sample_data
        make_sample_data.main()


# ---- Problem 1: DICOM header de-id ----
def test_dicom_scrub_removes_phi():
    _ensure_data()
    import pydicom
    from pctk.deid import deidentify_file, DeidOptions

    out = os.path.join(DATA, "_t_deid.dcm")
    deidentify_file(os.path.join(DATA, "sample_ultrasound.dcm"), out, DeidOptions())
    ds = pydicom.dcmread(out)
    assert "SHARMA" not in str(ds.PatientName).upper()
    assert str(ds.PatientName).startswith("ANON")
    assert str(ds.get("InstitutionName", "")) == ""
    assert str(ds.get("ReferringPhysicianName", "")) == ""
    assert (0x0018, 0x1000) not in ds          # device serial removed
    assert ds.PatientIdentityRemoved == "YES"
    assert ds.PatientAge == "027Y"             # clinically useful kept
    assert ds.Modality == "US"


# ---- Problem 1: pixel-text redaction ----
def test_pixel_redaction_keeps_centre():
    _ensure_data()
    from PIL import Image
    from pctk.deid import redact_image_file, RedactOptions

    src = os.path.join(DATA, "sample_ultrasound.png")
    out = os.path.join(DATA, "_t_redacted.png")
    # also_redact_bands guarantees the overlay bands are blanked whether or not
    # an OCR engine is installed (belt-and-suspenders).
    redact_image_file(src, out, RedactOptions(method="blackout",
                                              also_redact_bands=True))
    orig = np.asarray(Image.open(src).convert("L"))
    red = np.asarray(Image.open(out).convert("L"))
    band = int(orig.shape[0] * 0.18)
    assert red[:band].mean() == 0          # top band redacted
    assert red[-band:].mean() == 0         # bottom band redacted
    assert np.array_equal(orig[band:-band], red[band:-band])  # centre untouched


def test_pixel_redaction_ocr_if_available():
    """When an OCR engine is present, identifying strings are detected."""
    _ensure_data()
    from pctk.deid.ocr_backends import available_backends
    from pctk.deid import RedactOptions
    from pctk.deid.pixel_text import redact_array
    from PIL import Image

    if not available_backends():
        return  # no OCR installed; covered by the band-fallback test above
    arr = np.asarray(Image.open(os.path.join(DATA, "sample_ultrasound.png")
                                ).convert("RGB"))
    opts = RedactOptions(method="blackout", upscale=2.0)
    _, boxes = redact_array(arr, opts)
    found = " ".join(b[4].upper() for b in boxes)
    assert "PRIYA" in found or "SHARMA" in found or "SUNRISE" in found


# ---- Problem 2: SRB anomalies ----
def test_srb_flags_skewed_districts():
    _ensure_data()
    from pctk.srb import load_births, compute_srb, flag_anomalies, SRBConfig

    cfg = SRBConfig(group_cols=["district"])
    df = load_births(os.path.join(DATA, "births.csv"))
    flagged = flag_anomalies(compute_srb(df, cfg), cfg)
    sig = set(flagged[flagged["significant"]]["district"])
    assert "Hampi" in sig and "Fatehpur" in sig
    assert "Anandpur" not in sig            # natural district not flagged


# ---- Problem 3: compliance audit ----
def test_compliance_finds_violations():
    _ensure_data()
    from datetime import date
    from pctk.compliance import FormF, MachineRegistration, audit_records

    regs = [MachineRegistration("US-100", "FAC-A", date(2023, 1, 1))]
    forms = [
        FormF(record_id="ok", facility_id="FAC-A", machine_id="US-100",
              scan_date=date(2025, 3, 10), referring_doctor="Dr Rao",
              indication="dating scan", gestational_age_weeks=12,
              performed_by="T1", patient_ref="P1"),
        FormF(record_id="bad", facility_id="FAC-A", machine_id="US-900",
              scan_date=date(2025, 3, 10), referring_doctor="",
              indication="routine", gestational_age_weeks=12,
              performed_by="T1", patient_ref="P2"),
    ]
    res = audit_records(forms, regs)
    rules = {f.rule for f in res["findings"]}
    assert "machine_unregistered" in rules
    assert "form_f_incomplete" in rules
    assert "indication_vague" in rules
    assert res["counts"]["high"] >= 2


# ---- Problem 4: biometry ----
def test_biometry_reasonable():
    from pctk.biometry import ga_from_measurements, estimated_fetal_weight_hadlock

    res = ga_from_measurements(bpd_mm=46, hc_mm=170, ac_mm=150, fl_mm=30)
    assert 18 <= res["ga_weeks"] <= 22      # ~20 weeks
    efw = estimated_fetal_weight_hadlock(150, 30, 46, 170)
    assert 200 <= efw <= 450                # ~300g at 20w


# ---- Problem 2: aggregated (wide) counts via adapter ----
def test_srb_aggregated_counts():
    import pandas as pd
    from pctk.srb import from_aggregated_counts, flag_anomalies, SRBConfig

    df = pd.DataFrame({
        "district": ["A", "B"],
        "Births - Male": [520, 1000],
        "Births - Female": [500, 600],   # B is heavily skewed
    })
    table = from_aggregated_counts(df, ["district"],
                                   "Births - Male", "Births - Female")
    flagged = flag_anomalies(table, SRBConfig(group_cols=["district"]))
    sig = set(flagged[flagged["significant"]]["district"])
    assert "B" in sig and "A" not in sig


# ---- Problem 2: NFHS (long format, ratio-only) adapter ----
def test_nfhs_adapter():
    import pandas as pd
    from pctk.srb import from_nfhs

    df = pd.DataFrame({
        "State": ["KA", "KA", "KA"],
        "District": ["Udupi", "Udupi", "Uttara Kannada"],
        "Indicator": [
            "Total fertility rate",
            "4. Sex ratio at birth for children born in the last five years "
            "(females per 1,000 males)",
            "4. Sex ratio at birth for children born in the last five years "
            "(females per 1,000 males)",
        ],
        "NFHS-5": [1.7, 957.0, 724.0],
    })
    out = from_nfhs(df)
    assert len(out) == 2                       # only the two SRB rows
    assert bool(out["counts_available"].any()) is False
    low = out.sort_values("srb_f_per_1000_m").iloc[0]
    assert low["District"] == "Uttara Kannada"


# ---- Problem 2: Census 0-6 child counts -> significance test ----
def test_child06_adapter_significance():
    import pandas as pd
    from pctk.srb import from_census_child_06, flag_anomalies, SRBConfig

    df = pd.DataFrame({
        "State": ["Haryana", "Kerala"],
        "District": ["Jhajjar", "Kannur"],
        "M_06": [45928, 40833],
        "F_06": [34647, 39230],               # Jhajjar heavily skewed
    })
    table = from_census_child_06(df)
    flagged = flag_anomalies(table, SRBConfig(group_cols=["State", "District"]))
    sig = set(flagged[flagged["significant"]]["District"])
    assert "Jhajjar" in sig and "Kannur" not in sig


# ---- Problem 3: Form-F parsing + name pseudonymisation ----
def test_formf_parse_and_pseudonymise():
    from pctk.compliance import parse_form_f_text

    text = (
        "FORM F (PCPNDT Act, 1994)\n"
        "Name of pregnant woman: PRIYA SHARMA\n"
        "Date of test: 27-06-2026\n"
        "Referred by: Dr A Mehta\n"
        "Indication for ultrasound: Anomaly scan\n"
        "Period of gestation: 19 weeks\n"
        "Performed by: Dr S Rao\n"
        "USG Machine No: US-100\n"
        "Centre Registration No: FAC-A\n"
    )
    form, raw = parse_form_f_text(text)
    assert form.referring_doctor == "Dr A Mehta"
    assert form.indication == "Anomaly scan"
    assert form.machine_id == "US-100"
    assert form.facility_id == "FAC-A"
    assert form.gestational_age_weeks == 19.0
    assert str(form.scan_date) == "2026-06-27"
    # name converted to a stable pseudonym and NEVER stored verbatim
    assert form.patient_ref.startswith("PT-")
    assert "PRIYA" not in form.patient_ref
    blob = " ".join(str(getattr(form, f)) for f in vars(form))
    assert "PRIYA" not in blob and "SHARMA" not in blob


# ---- Fetal-plane classifier (sex-neutral) ----
def test_planes_pipeline():
    try:
        import sklearn  # noqa: F401
    except Exception:
        return  # sklearn is optional (extras: ml); skip if absent
    import tempfile
    from pctk.planes import (make_synthetic_planes, load_fetal_planes_db,
                             PlaneClassifier, PLANE_CLASSES)

    root = tempfile.mkdtemp(prefix="pctk_planes_test_")
    make_synthetic_planes(root, n_per_class=16, seed=1)
    df = load_fetal_planes_db(root)
    assert set(df["label"]) == set(PLANE_CLASSES)
    clf = PlaneClassifier()
    clf.train(df)
    res = clf.evaluate(df)
    assert res["accuracy"] >= 0.8            # distinct synthetic classes
    # round-trip persistence + prediction
    mp = os.path.join(root, "m.joblib")
    clf.save(mp)
    clf2 = PlaneClassifier.load(mp)
    pred = clf2.predict(df["image_path"].iloc[0])
    assert pred["label"] in PLANE_CLASSES
    # the model has no notion of sex
    assert not any("sex" in c.lower() or "gender" in c.lower()
                   for c in clf2.classes_)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
