"""ingest (PDF/OCR paths faked), SRB adapters, planes download, and small gaps in biometry/health/planes/deid."""
from __future__ import annotations

import os
import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pctk.compliance import ingest as ing
from pctk.deid import ocr_backends as ob


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("PCTK_PSEUDONYM_KEY", "unit-test-key")


FORM = """Name of the pregnant woman: Asha Rao
Centre Registration No: FAC-77
Machine ID: US-5
Referred by: Dr Kumar
Indication for the ultrasound: dating scan
Performed by: Dr Lee
Gestational age: 14 weeks
Date of scan: 12-Mar-2026
"""


# ---------------------------------- pseudonyms / dates ----------------------------------

def test_pseudonym_is_keyed_stable_and_case_insensitive(monkeypatch):
    a = ing._hash_ref("Asha Rao")
    assert a == ing._hash_ref("  asha RAO ") and a.startswith("PT-") and len(a) == 19 and a[3:] == a[3:].upper()
    monkeypatch.setenv("PCTK_PSEUDONYM_KEY", "other")
    assert ing._hash_ref("Asha Rao") != a
    assert ing._hash_ref("x", prefix="ZZ").startswith("ZZ-")


def test_random_process_key_warns_once_and_is_cached(monkeypatch):
    monkeypatch.delenv("PCTK_PSEUDONYM_KEY", raising=False)
    monkeypatch.setattr(ing, "_process_key", None)
    with pytest.warns(UserWarning, match="PCTK_PSEUDONYM_KEY not set"):
        k1 = ing._pseudonym_key()
    assert len(k1) == 32 and ing._pseudonym_key() is k1


@pytest.mark.parametrize("s,want", [("2026-03-04", date(2026, 3, 4)), ("04-03-2026", date(2026, 3, 4)),
                                    ("04/03/2026", date(2026, 3, 4)), ("04-Mar-2026", date(2026, 3, 4)),
                                    ("04/Mar/2026", date(2026, 3, 4)), ("04-March-2026", date(2026, 3, 4)),
                                    ("20260304", date(2026, 3, 4)), (" 2026-03-04 ", date(2026, 3, 4)),
                                    ("soon", None)])
def test_parse_date(s, want):
    assert ing._parse_date(s) == want


# ---------------------------------- field parsing ----------------------------------

def test_parse_form_f_text_extracts_fields_and_never_stores_name():
    form, raw = ing.parse_form_f_text(FORM)
    assert form.facility_id == "FAC-77" and form.machine_id == "US-5" and form.referring_doctor == "Dr Kumar"
    assert form.indication == "dating scan" and form.performed_by == "Dr Lee"
    assert form.gestational_age_weeks == 14.0 and form.scan_date == date(2026, 3, 12)
    assert form.patient_ref == ing._hash_ref("Asha Rao") and "Asha" not in repr(raw) and "Asha" not in repr(form)
    assert raw["_patient_name"] == form.patient_ref and raw["scan_date"] == "12-Mar-2026"
    assert raw["gestational_age_weeks"] == "14"


def test_parse_form_f_text_facility_default_only_when_missing():
    form, _ = ing.parse_form_f_text("Machine ID: M1", facility_id="DEFAULT")
    assert form.facility_id == "DEFAULT" and form.machine_id == "M1"
    form2, _ = ing.parse_form_f_text("Registration No: REAL-1", facility_id="DEFAULT")
    assert form2.facility_id == "REAL-1"
    empty, raw = ing.parse_form_f_text("nothing useful here")
    assert raw == {} and empty.facility_id in ("", None) and empty.scan_date is None


@pytest.mark.parametrize("text,weeks", [("Gestational age: 20 wks", 20.0), ("Period of gestation 31w", 31.0),
                                        ("GA: 9 weeks", 9.0), ("seen at 27 weeks 3 days", 27.0)])
def test_ga_patterns(text, weeks):
    assert ing.parse_form_f_text(text)[0].gestational_age_weeks == weeks


def test_date_patterns_and_unparseable_date_is_skipped():
    assert ing.parse_form_f_text("Date of examination: 2026-01-05")[0].scan_date == date(2026, 1, 5)
    assert ing.parse_form_f_text("Date: 05/01/2026")[0].scan_date == date(2026, 1, 5)
    form, raw = ing.parse_form_f_text("Date of scan: 99-Zzz-2026")
    assert form.scan_date is None and "scan_date" not in raw


def test_match_label_cleans_values():
    assert ing._match_label("Machine ID: No.: 77", [r"machine id"]) == "77"
    assert ing._match_label("Referred by:   Dr A    Indication: x", [r"referred by"]) == "Dr A"
    assert ing._match_label("Referred by: Dr A name: Bob", [r"referred by"]) == "Dr A"
    assert ing._match_label("nothing", [r"referred by"]) is None
    assert ing._match_label("zzz: 1\nreferred by - Dr B", [r"nope", r"referred by"]) == "Dr B"


# ---------------------------------- text extraction ----------------------------------

def _fake_pypdf(monkeypatch, pages=None, boom=False):
    mod = types.ModuleType("pypdf")

    class Reader:
        def __init__(self, path):
            if boom:
                raise RuntimeError("corrupt")
            self.pages = [types.SimpleNamespace(extract_text=lambda t=t: t) for t in (pages or [])]
    mod.PdfReader = Reader
    monkeypatch.setitem(sys.modules, "pypdf", mod)


def test_pdf_digital_text(monkeypatch):
    _fake_pypdf(monkeypatch, ["page one", None, "page three"])
    assert ing._pdf_digital_text("x.pdf") == "page one\n\npage three"
    _fake_pypdf(monkeypatch, boom=True)
    assert ing._pdf_digital_text("x.pdf") == ""
    monkeypatch.setitem(sys.modules, "pypdf", None)
    assert ing._pdf_digital_text("x.pdf") == ""


def test_pdf_render_pages(monkeypatch):
    assert True
    mod = types.ModuleType("fitz")
    closed = []
    used = {}

    class Pix:
        def __init__(self, n):
            self.height, self.width, self.n = 2, 3, n
            self.samples = bytes(range(2 * 3 * n))

    class Page:
        def __init__(self, n):
            self.n = n

        def get_pixmap(self, matrix):
            used["matrix"] = matrix
            return Pix(self.n)

    class Doc(list):
        def close(self):
            closed.append(1)
    mod.Matrix = lambda a, b: (a, b)
    mod.open = lambda path: Doc([Page(3), Page(4)])
    monkeypatch.setitem(sys.modules, "fitz", mod)
    pages = ing._pdf_render_pages("x.pdf", dpi=144)
    assert [p.shape for p in pages] == [(2, 3, 3), (2, 3, 3)] and closed == [1] and used["matrix"] == (2.0, 2.0)
    assert pages[0].flags["C_CONTIGUOUS"]
    monkeypatch.setitem(sys.modules, "fitz", None)
    assert ing._pdf_render_pages("x.pdf") == []


def test_extract_text_digital_pdf_threshold(monkeypatch):
    monkeypatch.setattr(ing, "_pdf_digital_text", lambda p: "x" * 40)
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: pytest.fail("no OCR needed"))
    assert ing.extract_text("A.PDF") == "x" * 40                       # extension match is case-insensitive; 40 chars suffice
    monkeypatch.setattr(ing, "_pdf_digital_text", lambda p: " " * 100 + "short")
    seen = {}
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: seen.update(p=prefer) or "BACKEND")
    monkeypatch.setattr(ing, "_pdf_render_pages", lambda p: [np.zeros((2, 2, 3), np.uint8)] * 3)
    texts = iter(["page1", "", "page3"])
    monkeypatch.setattr(ob, "image_to_lines", lambda img, backend=None, **k: (backend == "BACKEND") and next(texts))
    assert ing.extract_text("scan.pdf", ocr_backend="easyocr") == "page1\npage3" and seen["p"] == "easyocr"


def test_extract_text_image_uses_ocr(monkeypatch, tmp_path):
    p = tmp_path / "f.png"
    Image.fromarray(np.zeros((5, 7, 3), np.uint8)).save(p)
    seen = {}
    monkeypatch.setattr(ob, "image_to_lines", lambda arr, prefer=None, **k: seen.update(shape=arr.shape, prefer=prefer) or FORM)
    assert ing.extract_text(str(p), ocr_backend="pytesseract") == FORM and seen == {"shape": (5, 7, 3), "prefer": "pytesseract"}


def test_ingest_form_f_sets_record_id_and_facility(monkeypatch):
    monkeypatch.setattr(ing, "extract_text", lambda path, ocr_backend=None: FORM)
    form, raw = ing.ingest_form_f("/data/forms/rec42.pdf", facility_id="FALLBACK")
    assert form.record_id == "rec42" and form.facility_id == "FAC-77" and "scan_date" in raw
    form2, _ = ing.ingest_form_f("a/b.c.png", record_id="custom")
    assert form2.record_id == "custom"
    form3, _ = ing.ingest_form_f("a/b.c.png")
    assert form3.record_id == "b.c"


# ---------------------------------- SRB adapters ----------------------------------

from pctk.srb import adapters as ad  # noqa: E402


def test_resolve_exact_then_contains():
    df = pd.DataFrame(columns=[" Male ", "Total Female Births"])
    assert ad._resolve(df, ["male"]) == " Male "
    assert ad._resolve(df, ["nothing", "female births"]) == "Total Female Births"
    assert ad._resolve(df, ["zzz"]) is None


def test_crs_default_groups_and_explicit_map():
    df = pd.DataFrame({"State": ["A", "A"], "district": ["x", "x"], "Year": [2020, 2020], "Births - Male": [10, 5], "Births - Female": [8, 4]})
    t = ad.from_crs_hmis(df)
    assert list(t.columns[:3]) == ["State", "district", "Year"] and t.loc[0, "males"] == 15 and t.loc[0, "females"] == 12
    df2 = pd.DataFrame({"d": ["x"], "b": [3], "g": [2]})
    t2 = ad.from_crs_hmis(df2, group_cols=["d"], column_map={"males": "b", "females": "g"})
    assert t2.loc[0, "total"] == 5
    with pytest.raises(ValueError, match="could not locate male/female"):
        ad.from_crs_hmis(pd.DataFrame({"a": [1]}))


def test_ratio_only_kinds_and_errors():
    df = pd.DataFrame({"district": ["A", "B"], "SRB": [900.0, "bad"]})
    t = ad.from_ratio_only(df)
    assert t["counts_available"].tolist() == [False, False] and t.loc[0, "pct_female"] == pytest.approx(900 / 1900)
    assert np.isnan(t.loc[1, "srb_f_per_1000_m"])
    m = ad.from_ratio_only(pd.DataFrame({"d": ["A"], "ratio": [125.0]}), group_cols=["d"], ratio_kind="m_per_100_f")
    assert m.loc[0, "srb_f_per_1000_m"] == pytest.approx(800.0)
    with pytest.raises(ValueError, match="could not locate the sex-ratio column"):
        ad.from_ratio_only(pd.DataFrame({"a": [1]}))
    assert ad.from_census_csr(pd.DataFrame({"d": ["A"], "Child Sex Ratio": [910]})).loc[0, "srb_f_per_1000_m"] == 910
    with pytest.raises(ValueError, match="child-sex-ratio"):
        ad.from_census_csr(pd.DataFrame({"a": [1]}))


def test_nfhs_picks_indicator_rows():
    df = pd.DataFrame({"State": ["S", "S", "S"], "District": ["d1", "d1", "d2"],
                       "Indicator": ["Sex ratio at birth for children born in the last five years (females per 1,000 males)",
                                     "Literacy", "SEX RATIO AT BIRTH ... "],
                       "NFHS-5": [920, 70, 880], "NFHS-4": [1, 2, 3]})
    t = ad.from_nfhs(df)
    assert list(t["District"]) == ["d1", "d2"] and list(t["srb_f_per_1000_m"]) == [920, 880]
    t4 = ad.from_nfhs(df, value_col="NFHS-4")
    assert list(t4["srb_f_per_1000_m"]) == [1, 3]
    with pytest.raises(ValueError, match="no rows match"):
        ad.from_nfhs(df, srb_pattern="zzz")
    with pytest.raises(ValueError, match="could not locate indicator/value"):
        ad.from_nfhs(pd.DataFrame({"a": [1]}))


def test_child06_and_dispatch():
    df = pd.DataFrame({"State": ["S"], "District": ["d"], "M_06": [1000], "F_06": [900]})
    t = ad.from_census_child_06(df)
    assert t.loc[0, "srb_f_per_1000_m"] == 900 and t.loc[0, "total"] == 1900
    with pytest.raises(ValueError, match="0-6 child"):
        ad.from_census_child_06(pd.DataFrame({"a": [1]}))
    assert ad.adapt("CHILD06", df).equals(t)
    with pytest.raises(ValueError, match="unknown source 'x'"):
        ad.adapt("x", df)
    assert set(ad.SOURCE_NOTES) == set(ad.SOURCE_ADAPTERS)


# ---------------------------------- planes download (no network, no kaggle) ----------------------------------

from pctk.planes import download as dl  # noqa: E402


def test_download_kaggle_requires_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(dl.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(RuntimeError, match="Kaggle credentials not found"):
        dl.download_kaggle(str(tmp_path / "d"))
    assert (tmp_path / "d").is_dir()


@pytest.mark.parametrize("unzip,expect_flag", [(True, True), (False, False)])
def test_download_kaggle_command(monkeypatch, tmp_path, capsys, unzip, expect_flag):
    monkeypatch.setenv("KAGGLE_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = {}
    monkeypatch.setattr(dl.subprocess, "run", lambda cmd, check: seen.update(cmd=cmd, check=check))
    out = dl.download_kaggle(str(tmp_path / "d"), slug="a/b", unzip=unzip)
    assert out == str(tmp_path / "d") and seen["check"] is True
    assert seen["cmd"][:5] == ["kaggle", "datasets", "download", "-d", "a/b"] and ("--unzip" in seen["cmd"]) is expect_flag
    assert seen["cmd"][5:7] == ["-p", str(tmp_path / "d")] and capsys.readouterr().out.startswith("running: kaggle datasets download")


def test_download_kaggle_token_file_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    called = []
    monkeypatch.setattr(dl.subprocess, "run", lambda cmd, check: called.append(cmd))
    dl.download_kaggle(str(tmp_path / "d"))
    assert called


def test_download_zenodo_url_and_progress(monkeypatch, tmp_path, capsys):
    import urllib.request
    seen = {}

    def fake_retrieve(url, out, reporthook):
        seen.update(url=url, out=out)
        reporthook(1, 100, 1000)
        reporthook(50, 100, 1000)
        reporthook(5, 100, 0)                      # unknown size: no progress output
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    out = dl.download_zenodo(str(tmp_path / "z"))
    assert out == os.path.join(str(tmp_path / "z"), dl.ZENODO_FILE)
    assert seen["url"] == f"https://zenodo.org/record/{dl.ZENODO_RECORD}/files/{dl.ZENODO_FILE}?download=1"
    text = capsys.readouterr().out
    assert "  10%" in text and " 100%" in text and text.count("%") == 2 and text.endswith("\n")


# ---------------------------------- small gaps ----------------------------------

def test_hadlock_efw_all_branches():
    from pctk.biometry import estimated_fetal_weight_hadlock as efw
    both = efw(300, 70, bpd_mm=85, hc_mm=310)
    hc_only = efw(300, 70, hc_mm=310)
    bpd_only = efw(300, 70, bpd_mm=85)
    ac_fl = efw(300, 70)
    assert len({both, hc_only, bpd_only, ac_fl}) == 4 and all(1500 < v < 5000 for v in (both, hc_only, bpd_only, ac_fl))
    assert both == round(10 ** (1.3596 - 0.00386 * 30 * 7 + 0.0064 * 31 + 0.00061 * 8.5 * 30 + 0.0424 * 30 + 0.174 * 7), 1)
    assert ac_fl == round(10 ** (1.304 + 0.05281 * 30 + 0.1938 * 7 - 0.004 * 30 * 7), 1)


def test_health_loader_target_detection_and_errors(tmp_path):
    from pctk.health import load_fetal_health
    p = tmp_path / "a.csv"
    pd.DataFrame({"x": [1, 2], "Fetal_Health_Class": [1.0, 2.6]}).to_csv(p, index=False)
    X, y = load_fetal_health(str(p))
    assert list(X.columns) == ["x"] and y.tolist() == [1, 3]                  # rounded to int
    q = tmp_path / "b.csv"
    pd.DataFrame({"x": [1, 2], "Label": [1, 2]}).to_csv(q, index=False)
    assert load_fetal_health(str(q))[1].tolist() == [1, 2]
    r = tmp_path / "c.csv"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(r, index=False)
    with pytest.raises(ValueError, match="could not find the fetal_health target"):
        load_fetal_health(str(r))


def test_health_model_variants_and_guards(tmp_path):
    from pctk.health import FetalHealthClassifier, make_synthetic_health, load_fetal_health
    from pctk.health.model import HealthConfig
    p = tmp_path / "ctg.csv"
    make_synthetic_health(str(p), n=240, seed=4)
    X, y = load_fetal_health(str(p))
    fresh = FetalHealthClassifier()
    with pytest.raises(RuntimeError, match="fit_eval"):
        fresh.feature_importances()
    with pytest.raises(RuntimeError, match="fit_eval"):
        fresh.predict(X.iloc[0])
    rf = FetalHealthClassifier(HealthConfig(model="rf"))
    res = rf.fit_eval(X, y)
    assert res["n_train"] + res["n_test"] == 240 and len(rf.feature_importances(top=3)) == 3
    imps = [v for _, v in rf.feature_importances(top=100)]
    assert imps == sorted(imps, reverse=True)
    a = rf.predict(X.iloc[0].to_dict())
    b = rf.predict(X.iloc[0])
    assert a == b and a["status"] and abs(sum(a["proba"].values()) - 1) < 1e-9


def test_planes_loader_variants(tmp_path):
    from pctk.planes import load_fetal_planes_db
    from pctk.planes.dataset import _find_csv
    with pytest.raises(FileNotFoundError, match="no metadata CSV"):
        load_fetal_planes_db(str(tmp_path))
    (tmp_path / "Images").mkdir()
    for n in ("a.png", "b.jpg"):
        Image.fromarray(np.zeros((4, 4), np.uint8)).save(tmp_path / "Images" / n)
    pd.DataFrame({"image": ["a", "b", "missing"], "label": [" Fetal brain ", "Other", "Other"], "split": ["train", "test", "x"]}).to_csv(
        tmp_path / "meta.csv", index=False)
    assert _find_csv(str(tmp_path)).endswith("meta.csv")
    df = load_fetal_planes_db(str(tmp_path))
    assert df["label"].tolist() == ["Fetal brain", "Other", "Other"]
    assert [os.path.basename(x) for x in df["image_path"]] == ["a.png", "b.jpg", "missing.png"]
    assert df["split"].tolist() == ["train", "test", "test"]
    pd.DataFrame({"Image_name": ["a"], "Plane": ["Other"]}).to_csv(tmp_path / "FETAL_PLANES_DB_data.csv", sep=";", index=False)
    assert _find_csv(str(tmp_path)).endswith("FETAL_PLANES_DB_data.csv")        # preferred over any *.csv
    df2 = load_fetal_planes_db(str(tmp_path))
    assert df2["split"].tolist() == ["train"]                                  # no split column -> train
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "FETAL_PLANES_DB_data.csv", index=False)
    with pytest.raises(ValueError, match="could not find image/plane columns"):
        load_fetal_planes_db(str(tmp_path))


def test_planes_train_and_evaluate_guards_and_features(tmp_path):
    from pctk.planes import PlaneClassifier, make_synthetic_planes, load_fetal_planes_db
    from pctk.planes.features import _load_gray
    c = PlaneClassifier()
    df = pd.DataFrame({"image_path": [], "label": [], "split": []})
    with pytest.raises(RuntimeError, match="train"):
        c.evaluate(df)
    with pytest.raises(RuntimeError, match="train"):
        c.predict("x.png")
    with pytest.raises(ValueError, match="no rows for split='train'"):
        c.train(df)
    root = str(tmp_path / "pl")
    make_synthetic_planes(root, n_per_class=6, seed=2)
    full = load_fetal_planes_db(root)
    c.train(full)
    with pytest.raises(ValueError, match="no rows for split='val'"):
        c.evaluate(full, split="val")
    unsplit = full.drop(columns=["split"])
    assert c.evaluate(unsplit)["n_test"] == len(unsplit)
    assert c.train(unsplit)["n_train"] == len(unsplit)
    arr = np.random.default_rng(0).integers(0, 255, (10, 12, 3), dtype=np.uint8)
    assert _load_gray(arr, 8).shape == (8, 8) and _load_gray(arr[:, :, 0], 8).max() <= 1.0
    assert _load_gray(Image.fromarray(arr), 8).shape == (8, 8)


def test_srb_loaders_and_count_col(tmp_path):
    from pctk.srb import load_births, compute_srb, SRBConfig
    df = pd.DataFrame({"district": ["a", "a", "b"], "sex": ["Girl", "2", "boy"], "n": [5, 7, 3]})
    cfg = SRBConfig(group_cols=["district"], count_col="n")
    t = compute_srb(df, cfg)
    assert t.set_index("district").loc["a", "females"] == 12 and t.set_index("district").loc["b", "females"] == 0
    t2 = compute_srb(df.assign(sex=["F", "F", "F"]), SRBConfig(group_cols=["district"]))
    assert (t2["males"] == 0).all() and t2["srb_f_per_1000_m"].isna().all()
    tsv, pq = tmp_path / "b.tsv", tmp_path / "b.parquet"
    df.to_csv(tsv, sep="\t", index=False)
    assert load_births(str(tsv)).shape == (3, 3)
    try:
        df.to_parquet(pq)
    except Exception:
        return                                                           # no parquet engine installed: skip that branch
    assert load_births(str(pq)).shape == (3, 3)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="BUG ingest.py _match_label: the next-label split regex is case-sensitive, so a "
                          "capitalised following label ('Name:', 'Date:', 'Indication:') is not recognised and "
                          "is swallowed into the value ('Dr A Name: Bob')")
def test_match_label_stops_at_capitalised_next_label():
    assert ing._match_label("Referred by: Dr A Name: Bob", [r"referred by"]) == "Dr A"


def test_dicom_scrub_edge_cases(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset
    from pctk.deid import dicom_scrub as ds_mod
    from pctk.deid.dicom_scrub import DeidOptions, deidentify_dataset, deidentify_file
    from _dcm import write_dicom

    assert ds_mod._remap_uid("").startswith(ds_mod.PCTK_UID_ROOT)
    assert ds_mod._remap_uid("1.2.3") == ds_mod._remap_uid("1.2.3") != ds_mod._remap_uid("1.2.4")
    d = Dataset()
    d.StudyDate = "20261399"                          # eight digits but not a real date
    ds_mod._apply_date(d, (0x0008, 0x0020), DeidOptions())
    assert d.StudyDate == ""
    ds_mod._apply_date(d, (0x0008, 0x0021), DeidOptions())      # absent tag: nothing happens
    assert (0x0008, 0x0021) not in d

    a, b = Dataset(), Dataset()
    a.PatientID = "P-1"; a.StudyDate = "20260101"
    b.PatientID = "P-1"; b.StudyDate = "20260101"
    assert deidentify_dataset(a).StudyDate == deidentify_dataset(b).StudyDate != "20260101"   # default shift derived from the ID
    shift = ds_mod._shift_days_for(a, DeidOptions())
    assert -364 <= shift <= 364 and ds_mod._shift_days_for(a, DeidOptions(date_shift_days=0)) == 0
    c = Dataset()
    c.PatientName = "Only^Name"
    assert -364 <= ds_mod._shift_days_for(c, DeidOptions()) <= 364

    src, dst = tmp_path / "i.dcm", tmp_path / "o.dcm"
    write_dicom(src)
    opts = deidentify_file(str(src), str(dst))
    out = pydicom.dcmread(str(dst))
    assert str(out.PatientName).startswith("ANON^") and str(out.PatientID).startswith("ANON-")
    assert opts.audit_log and any("removed private tags" in line for line in opts.audit_log)
    assert out.DeidentificationMethod.startswith("pctk PS3.15 basic subset v")

    e = Dataset()
    e.AccessionNumber = "A1"
    e.PatientSex = "F"
    e.PatientIdentityRemoved = "NO"
    keep = deidentify_dataset(e, DeidOptions(keep_patient_sex=True, dummy_id_prefix="ZZ"))
    assert keep.AccessionNumber == "ZZ" and keep.PatientSex == "F"
    assert deidentify_dataset(Dataset(), DeidOptions(remove_private_tags=False, blank_all_person_names=False)).PatientIdentityRemoved == "YES"


def test_planes_resolve_col_and_health_importances_fallback(monkeypatch):
    from pctk.planes.dataset import _resolve_col
    assert _resolve_col(["Image_Name", "Plane"], "image_name") == "Image_Name"
    assert _resolve_col(["my_filename_col"], "image", "filename") == "my_filename_col"
    assert _resolve_col(["a"], "zzz") is None
    from pctk.health import FetalHealthClassifier
    c = FetalHealthClassifier()
    c.pipeline = types.SimpleNamespace(named_steps={"clf": object()})
    assert c.feature_importances() == []


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="BUG ingest.py _match_label: a blank field ('Referred by:' then newline) lets "
                          "`\\s*` run across the newline so the NEXT line is captured as this field's value "
                          "(and at end of text the value becomes ':')")
def test_blank_field_does_not_capture_next_line():
    form, _ = ing.parse_form_f_text("Referred by:\nPerformed by: Dr Lee\n")
    assert not form.referring_doctor
