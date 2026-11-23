"""pctk.cli: drive main(argv) for every sub-command on small synthetic inputs (no OCR engine, no network)."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("pydicom")
pytest.importorskip("sklearn")

import pctk  # noqa: E402
from pctk import cli  # noqa: E402
from pctk.deid import ocr_backends  # noqa: E402
from _dcm import write_dicom  # noqa: E402


@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    """Deterministic: behave as if no OCR engine is installed."""
    monkeypatch.setattr(ocr_backends, "get_backend", lambda prefer=None, languages=None: None)
    monkeypatch.setattr(ocr_backends, "available_backends", lambda: [])


def run(capsys, *argv):
    rc = cli.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


# ---------------------------------- misc ----------------------------------

def test_parse_date_formats():
    from datetime import date
    for s in ("2026-03-04", "04-03-2026", "20260304", "04/03/2026", " 2026-03-04 "):
        assert cli._parse_date(s) == date(2026, 3, 4)
    assert cli._parse_date("") is None and cli._parse_date(None) is None and cli._parse_date("3rd March") is None


def test_version_and_no_command(capsys):
    rc, out, _ = run(capsys, "--version")
    assert rc == 0 and out == f"pctk {pctk.__version__}\n"
    rc, out, _ = run(capsys)
    assert rc == 1 and "usage: pctk" in out


def test_importerror_is_reported_with_exit_2(monkeypatch, capsys):
    def boom(args):
        raise ImportError("scikit-learn missing")
    monkeypatch.setattr(cli, "cmd_backends", boom)
    rc, out, err = run(capsys, "backends")
    assert rc == 2 and err == "[pctk] scikit-learn missing\n"


def test_backends_listing(monkeypatch, capsys):
    rc, out, _ = run(capsys, "backends")
    assert rc == 0 and "no OCR engine installed" in out and "pip install easyocr" in out
    monkeypatch.setattr(ocr_backends, "available_backends", lambda: ["easyocr", "pytesseract"])
    rc, out, _ = run(capsys, "backends")
    assert out == "[backends] OCR available: easyocr, pytesseract\n"


def test_main_module_entry(monkeypatch):
    import runpy
    monkeypatch.setattr("sys.argv", ["pctk", "--version"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("pctk", run_name="__main__")
    assert e.value.code == 0


# ---------------------------------- Problem 1 ----------------------------------

def test_deid_dicom_cli(tmp_path, capsys):
    import pydicom
    src, dst = tmp_path / "in.dcm", tmp_path / "out.dcm"
    write_dicom(src)
    rc, out, _ = run(capsys, "deid-dicom", str(src), str(dst), "--date-policy", "blank", "-v")
    assert rc == 0 and out.startswith(f"[deid-dicom] wrote {dst}  (") and "\n   - " in out
    ds = pydicom.dcmread(str(dst))
    assert str(ds.PatientID).startswith("ANON-") and ds.StudyDate == "" and ds.PatientIdentityRemoved == "YES"


def test_deid_dicom_cli_quiet_and_keep_flags(tmp_path, capsys):
    import pydicom
    src, dst = tmp_path / "in.dcm", tmp_path / "out.dcm"
    write_dicom(src)
    rc, out, _ = run(capsys, "deid-dicom", str(src), str(dst), "--date-policy", "keep", "--keep-private",
                     "--keep-patient-sex")
    assert rc == 0 and "\n   - " not in out
    assert pydicom.dcmread(str(dst)).StudyDate == "20260627"


def _png(path, h=100, w=100):
    Image.fromarray(np.full((h, w, 3), 200, np.uint8)).save(path)


def test_deid_image_cli_margin_fallback(tmp_path, capsys):
    src, dst = tmp_path / "a.png", tmp_path / "b.png"
    _png(src)
    rc, out, _ = run(capsys, "deid-image", str(src), str(dst), "--method", "blackout")
    assert rc == 0 and f"[deid-image] wrote {dst}" in out and "no OCR backend; using margin-band fallback" in out
    arr = np.asarray(Image.open(dst))
    assert arr[0, 0].sum() == 0 and arr[50, 50].tolist() == [200, 200, 200] and arr[-1, 0].sum() == 0


def test_deid_image_cli_flags(tmp_path, capsys):
    src, dst = tmp_path / "a.png", tmp_path / "b.png"
    _png(src)
    rc, out, _ = run(capsys, "deid-image", str(src), str(dst), "--whole-image", "--also-redact-bands",
                     "--method", "pixelate")
    assert rc == 0


def test_deid_frame_cli(tmp_path, capsys):
    src, dst = tmp_path / "in.dcm", tmp_path / "f.png"
    write_dicom(src, rows=100, cols=100, frames=2)
    rc, out, _ = run(capsys, "deid-frame", str(src), str(dst), "--frame", "1", "--method", "blackout")
    assert rc == 0 and f"[deid-frame] wrote {dst}" in out
    assert np.asarray(Image.open(dst))[0, 0].sum() == 0


# ---------------------------------- Problem 2 ----------------------------------

def _births(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["district", "sex"])
        for d, f, m in rows:
            w.writerows([[d, "F"]] * f + [[d, "M"]] * m)


def test_srb_cli_default_long_format(tmp_path, capsys):
    p, out_csv = tmp_path / "b.csv", tmp_path / "out.csv"
    _births(p, [("Skew", 30, 170), ("Fine", 98, 102), ("Tiny", 1, 3)])
    rc, out, _ = run(capsys, "srb", str(p), "--min-births", "30", "--top", "5", "--out", str(out_csv))
    assert rc == 0 and "[srb] 2 units >= 30 births; 1 flagged at alpha=0.01" in out
    assert f"[srb] full table -> {out_csv}" in out
    t = pd.read_csv(out_csv)
    assert set(t["district"]) == {"Skew", "Fine"} and t.set_index("district").loc["Skew", "significant"]


def test_srb_cli_aggregated_counts_and_custom_group(tmp_path, capsys):
    p = tmp_path / "agg.csv"
    pd.DataFrame({"state": ["A", "A"], "district": ["x", "y"], "boys": [200, 100], "girls": [100, 98]}).to_csv(p, index=False)
    rc, out, _ = run(capsys, "srb", str(p), "--males-col", "boys", "--females-col", "girls",
                     "--group", "state", "district", "--alpha", "0.05")
    assert rc == 0 and "units >= 30 births; 1 flagged at alpha=0.05" in out


def test_srb_cli_ratio_only_adapter(tmp_path, capsys):
    p, out_csv = tmp_path / "r.csv", tmp_path / "o.csv"
    pd.DataFrame({"district": ["A", "B"], "SRB": [850, 950]}).to_csv(p, index=False)
    rc, out, _ = run(capsys, "srb", str(p), "--source", "ratio", "--group", "district", "--out", str(out_csv))
    assert rc == 0 and "[srb] adapter 'ratio'" in out and "ratio-only source: 2 units" in out
    assert list(pd.read_csv(out_csv)["district"]) == ["A", "B"]               # lowest female ratio first
    assert f"[srb] full table -> {out_csv}" in out


def test_srb_cli_ratio_only_without_out(tmp_path, capsys):
    p = tmp_path / "r.csv"
    pd.DataFrame({"district": ["A"], "SRB": [900]}).to_csv(p, index=False)
    rc, out, _ = run(capsys, "srb", str(p), "--source", "ratio")
    assert rc == 0 and "full table" not in out


def test_srb_cli_adapter_with_counts(tmp_path, capsys):
    p = tmp_path / "c.csv"
    pd.DataFrame({"State": ["A", "A"], "District": ["x", "y"], "M_06": [1000, 1000], "F_06": [700, 990]}).to_csv(p, index=False)
    rc, out, _ = run(capsys, "srb", str(p), "--source", "child06")
    assert rc == 0 and "adapter 'child06'" in out and "flagged at alpha=0.01" in out


# ---------------------------------- Problem 3 ----------------------------------

def test_compliance_cli(tmp_path, capsys):
    forms, regs = tmp_path / "f.csv", tmp_path / "r.csv"
    pd.DataFrame([
        {"record_id": "r1", "patient_ref": "P1", "facility_id": "FAC-A", "machine_id": "US-1", "scan_date": "2025-03-10",
         "referring_doctor": "Dr R", "indication": "dating scan", "gestational_age_weeks": "12", "performed_by": "T"},
        {"record_id": "", "patient_ref": "P2", "facility_id": "FAC-A", "machine_id": "US-9", "scan_date": "10/03/2025",
         "referring_doctor": "", "indication": "routine", "gestational_age_weeks": "", "performed_by": "T"},
    ]).to_csv(forms, index=False)
    pd.DataFrame([{"machine_id": "US-1", "facility_id": "FAC-A", "registered_from": "2023-01-01", "registered_to": ""}]).to_csv(regs, index=False)
    rc, out, _ = run(capsys, "compliance", str(forms), str(regs))
    assert rc == 0 and "[compliance] 2 records," in out and "facility completeness:" in out and "FAC-A" in out
    assert "[high" not in out
    rc, out, _ = run(capsys, "compliance", str(forms), str(regs), "-v")
    assert "machine_unregistered" in out and "indication_vague" in out


# ---------------------------------- ingest-formf ----------------------------------

def test_ingest_formf_cli_with_append(tmp_path, monkeypatch, capsys):
    from datetime import date
    from pctk.compliance import FormF
    form = FormF(record_id="rec1", patient_ref="PT-1", facility_id="FAC", machine_id="M1", scan_date=date(2026, 1, 2),
                 referring_doctor="", indication="dating", gestational_age_weeks=12.0, performed_by="Dr X")
    seen = {}

    def fake_ingest(path, facility_id=None, ocr_backend=None):
        seen.update(path=path, fac=facility_id, ocr=ocr_backend)
        return form, {}
    import pctk.compliance as comp
    monkeypatch.setattr(comp, "ingest_form_f", fake_ingest)
    out_csv = tmp_path / "forms.csv"
    rc, out, _ = run(capsys, "ingest-formf", "scan.pdf", "--facility-id", "FAC", "--ocr-backend", "easyocr",
                     "--append-csv", str(out_csv))
    assert rc == 0 and seen == {"path": "scan.pdf", "fac": "FAC", "ocr": "easyocr"}
    assert "record_id              = rec1" in out and "referring_doctor" in out and "<not found>" in out
    assert "! missing required fields:" in out and f"-> appended to {out_csv}" in out
    run(capsys, "ingest-formf", "scan.pdf", "--append-csv", str(out_csv))
    rows = list(csv.reader(open(out_csv)))
    assert len(rows) == 3 and rows[0][0] == "record_id" and rows[1][0] == "rec1" and rows[1][3] == "2026-01-02"
    assert rows[1][-1] == "PT-1"


# ---------------------------------- Problem 4 ----------------------------------

def test_biometry_cli(capsys):
    rc, out, _ = run(capsys, "biometry", "--bpd", "46", "--hc", "170", "--ac", "150", "--fl", "30")
    assert rc == 0 and "composite GA:" in out and "estimated fetal weight (Hadlock):" in out and "BPD" in out
    rc, out, _ = run(capsys, "biometry", "--bpd", "46")
    assert "estimated fetal weight" not in out
    rc, out, _ = run(capsys, "biometry", "--ac", "150")                       # AC without FL: no EFW
    assert "estimated fetal weight" not in out


def test_planes_and_health_cli_round_trip(tmp_path, capsys):
    out_model = tmp_path / "demo.joblib"
    rc, out, _ = run(capsys, "planes-demo", "--data", str(tmp_path / "pl"), "--n-per-class", "8", "--out", str(out_model))
    assert rc == 0 and "test accuracy:" in out and f"model -> {out_model}" in out and out_model.exists()

    m = tmp_path / "m.joblib"
    rc, out, _ = run(capsys, "planes-train", str(tmp_path / "pl"), "--out", str(m))
    assert rc == 0 and "[planes-train:sklearn]" in out and "held-out accuracy:" in out
    rc, out, _ = run(capsys, "planes-eval", str(tmp_path / "pl"), "--model", str(m))
    assert rc == 0 and "[planes-eval] accuracy:" in out
    img = next((tmp_path / "pl" / "Images").iterdir())
    rc, out, _ = run(capsys, "planes-predict", str(img), "--model", str(m))
    assert rc == 0 and "plane:" in out and "(confidence" in out

    h = tmp_path / "ctg.csv"
    hm = tmp_path / "h.joblib"
    rc, out, _ = run(capsys, "health-demo", "--data", str(h), "--n", "300", "--out", str(hm))
    assert rc == 0 and "top features:" in out and f"model -> {hm}" in out
    rc, out, _ = run(capsys, "health-train", str(h), "--out", str(tmp_path / "h2.joblib"))
    assert rc == 0 and "[health-train] accuracy=" in out
    X = pd.read_csv(h).drop(columns=["fetal_health"])
    row = tmp_path / "row.json"
    row.write_text(json.dumps(X.iloc[0].to_dict()))
    rc, out, _ = run(capsys, "health-predict", str(row), "--model", str(hm))
    assert rc == 0 and "[health-predict] status:" in out


def test_planes_demo_default_temp_dir(capsys):
    rc, out, _ = run(capsys, "planes-demo", "--n-per-class", "6")
    assert rc == 0 and "generating synthetic FETAL_PLANES_DB-shaped data in " in out


def test_health_demo_default_temp_csv(capsys):
    rc, out, _ = run(capsys, "health-demo", "--n", "240")
    assert rc == 0 and "generating synthetic CTG data ->" in out and "model ->" not in out


def test_plane_backend_selection(monkeypatch):
    import types
    import sys
    from pctk.planes import PlaneClassifier
    assert isinstance(cli._plane_clf("sklearn"), PlaneClassifier)
    fake = types.ModuleType("pctk.planes.torch_cnn")

    class Cfg:
        def __init__(self, epochs):
            self.epochs = epochs

    class CNN:
        def __init__(self, cfg):
            self.cfg = cfg

        @classmethod
        def load(cls, path):
            return ("loaded", path)
    fake.CNNPlaneClassifier, fake.CNNConfig = CNN, Cfg
    monkeypatch.setitem(sys.modules, "pctk.planes.torch_cnn", fake)
    monkeypatch.setattr("pctk.planes.torch_cnn", fake, raising=False)
    assert cli._plane_clf("torch", epochs=3).cfg.epochs == 3
    assert cli._load_plane_clf("torch", "p") == ("loaded", "p")
