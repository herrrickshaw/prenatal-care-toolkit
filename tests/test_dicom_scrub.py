"""DICOM header de-identification on in-memory synthetic datasets."""
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from pctk.deid.dicom_scrub import DeidOptions, deidentify_dataset


def _ds():
    ds = Dataset()
    ds.PatientName = "Synthetica^Testa"
    ds.PatientID = "SYN-0001"
    ds.PatientBirthDate = "19900101"
    ds.PatientSex = "F"
    ds.StudyDate = "20260315"
    ds.ReferringPhysicianName = "Synth^Doctor"
    ds.InstitutionName = "Synthetic Clinic"
    ds.StudyInstanceUID = "1.2.3.4.5.6"
    ds.Modality = "US"
    ds.PatientAge = "030Y"
    return ds


def test_core_phi_removed_and_clinical_fields_kept():
    ds = deidentify_dataset(_ds(), DeidOptions(date_shift_days=10))
    assert "Synthetica" not in str(ds.PatientName)
    assert str(ds.PatientID).startswith("ANON-")
    assert ds.PatientBirthDate == "" and ds.PatientSex == ""
    assert ds.ReferringPhysicianName == "" and ds.InstitutionName == ""
    assert ds.StudyDate == "20260325"                          # shifted +10 days
    assert ds.StudyInstanceUID != "1.2.3.4.5.6"
    assert ds.Modality == "US" and ds.PatientAge == "030Y"
    assert ds.PatientIdentityRemoved == "YES"


def test_uid_remap_is_deterministic_for_referential_integrity():
    a = deidentify_dataset(_ds()).StudyInstanceUID
    b = deidentify_dataset(_ds()).StudyInstanceUID
    assert a == b


def test_date_policies():
    assert deidentify_dataset(_ds(), DeidOptions(date_policy="blank")).StudyDate == ""
    assert deidentify_dataset(_ds(), DeidOptions(date_policy="keep")).StudyDate == "20260315"
    bad = _ds(); bad.StudyDate = "2026-03"
    assert deidentify_dataset(bad, DeidOptions(date_shift_days=5)).StudyDate == ""


def test_person_names_inside_sequences_are_blanked():
    """The PN catch-all claims to blank every Person Name; it only walked the top level."""
    ds = _ds()
    item = Dataset()
    item.ScheduledPerformingPhysicianName = "Nested^Synthname"
    ds.ScheduledProcedureStepSequence = Sequence([item])
    out = deidentify_dataset(ds)
    assert str(out.ScheduledProcedureStepSequence[0].ScheduledPerformingPhysicianName) == ""


def test_private_tags_removed_by_default():
    ds = _ds()
    ds.add_new(0x00091010, "LO", "private synthetic")
    out = deidentify_dataset(ds)
    assert 0x00091010 not in out
