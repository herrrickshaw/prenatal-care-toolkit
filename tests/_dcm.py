"""Test helper: write a tiny synthetic DICOM (no real patient data)."""
import numpy as np


def write_dicom(path, rows=40, cols=60, frames=1, rgb=False):
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "SYNTH^PATIENT"
    ds.PatientID = "MRN-1"
    ds.PatientBirthDate = "19990115"
    ds.StudyDate = "20260627"
    ds.Modality = "US"
    ds.StudyInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    rng = np.random.default_rng(3)
    shape = (frames, rows, cols, 3) if rgb else (frames, rows, cols)
    arr = rng.integers(60, 200, size=shape, dtype=np.uint8)
    ds.SamplesPerPixel = 3 if rgb else 1
    ds.PhotometricInterpretation = "RGB" if rgb else "MONOCHROME2"
    if rgb:
        ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = rows, cols
    if frames > 1:
        ds.NumberOfFrames = str(frames)
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))
    return arr
