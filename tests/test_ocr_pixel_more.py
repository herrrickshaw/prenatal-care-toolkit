"""OCR backends (engines faked via sys.modules) and pixel-text redaction branches. No real OCR, no network."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pydicom")

from pctk.deid import ocr_backends as ob  # noqa: E402
from pctk.deid import pixel_text as pt  # noqa: E402
from _dcm import write_dicom  # noqa: E402


# ---------------------------------- helpers: fake engines ----------------------------------

def _fake_easyocr(monkeypatch, results):
    mod = types.ModuleType("easyocr")
    seen = {}

    class Reader:
        def __init__(self, langs, gpu, verbose):
            seen.update(langs=langs, gpu=gpu, verbose=verbose)

        def readtext(self, rgb):
            seen["shape"] = rgb.shape
            return results
    mod.Reader = Reader
    monkeypatch.setitem(sys.modules, "easyocr", mod)
    return seen


def _fake_pytesseract(monkeypatch, data):
    mod = types.ModuleType("pytesseract")
    seen = {}
    mod.Output = types.SimpleNamespace(DICT="dict")

    def image_to_data(img, lang, output_type):
        seen.update(lang=lang, output_type=output_type, size=img.size)
        return data
    mod.image_to_data = image_to_data
    monkeypatch.setitem(sys.modules, "pytesseract", mod)
    return seen


# ---------------------------------- _to_rgb_uint8 ----------------------------------

def test_to_rgb_uint8_conversions():
    g = np.array([[0, 5], [10, 20]], dtype=np.uint8)
    out = ob._to_rgb_uint8(g)
    assert out.shape == (2, 2, 3) and (out[:, :, 0] == g).all()
    f = ob._to_rgb_uint8(np.array([[10.0, 20.0], [30.0, 110.0]]))               # normalised to 0..255
    assert f.dtype == np.uint8 and f.min() == 0 and f.max() == 255 and f[0, 1, 0] == 25
    flat = ob._to_rgb_uint8(np.zeros((2, 2)))                                    # constant image: no divide by zero
    assert flat.max() == 0
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[..., 3] = 9
    assert ob._to_rgb_uint8(rgba).shape == (2, 2, 3)
    rgb = np.ones((2, 2, 3), np.uint8)
    assert ob._to_rgb_uint8(rgb) is not None and ob._to_rgb_uint8(rgb).shape == (2, 2, 3)


# ---------------------------------- engines ----------------------------------

def test_easyocr_backend_filters_and_converts_boxes(monkeypatch):
    seen = _fake_easyocr(monkeypatch, [
        ([[10, 20], [50, 20], [50, 40], [10, 40]], "NAME", 0.9),
        ([[0, 0], [5, 0], [5, 5], [0, 5]], "low", 0.2),            # below min_conf
        ([[1, 1], [9, 1], [9, 9], [1, 9]], "  ", 0.99),            # blank text
        ([[3.7, 4.2], [30.9, 4.2], [30.9, 12.1], [3.7, 12.1]], "edge", 0.3),   # == min_conf kept
    ])
    b = ob._EasyOCRBackend()
    assert seen["langs"] == ["en"] and seen["gpu"] is False and seen["verbose"] is False
    out = b.detect(np.zeros((8, 8), np.uint8))
    assert out == [(10, 20, 40, 20, "NAME", 0.9), (3, 4, 27, 8, "edge", 0.3)]
    assert seen["shape"] == (8, 8, 3) and b.name == "easyocr"
    ob._EasyOCRBackend(["hi", "en"])
    assert seen["langs"] == ["hi", "en"]
    assert b.detect(np.zeros((8, 8), np.uint8), min_conf=0.95) == []


def test_tesseract_backend_parses_dict_output(monkeypatch):
    data = {"text": ["HELLO", "", "low", "bad", " x "], "conf": ["90", "95", "10", "oops", "-1"],
            "left": [1, 2, 3, 4, 5], "top": [6, 7, 8, 9, 10], "width": [11, 12, 13, 14, 15], "height": [16, 17, 18, 19, 20]}
    seen = _fake_pytesseract(monkeypatch, data)
    b = ob._TesseractBackend()
    out = b.detect(np.zeros((4, 6), np.uint8), min_conf=0.3)
    assert out == [(1, 6, 11, 16, "HELLO", 0.9)]
    assert seen == {"lang": "eng", "output_type": "dict", "size": (6, 4)} and b.name == "pytesseract"
    ob._TesseractBackend(["eng", "hin"])
    b2 = ob._TesseractBackend(["eng", "hin"])
    b2.detect(np.zeros((2, 2), np.uint8))
    assert seen["lang"] == "eng+hin"
    data["conf"][2] = "30"                                        # exactly min_conf is kept
    assert [o[4] for o in b.detect(np.zeros((4, 6), np.uint8), min_conf=0.3)] == ["HELLO", "low"]


def test_get_backend_order_and_fallbacks(monkeypatch):
    _fake_easyocr(monkeypatch, [])
    _fake_pytesseract(monkeypatch, {})
    assert ob.get_backend().name == "easyocr"
    assert ob.get_backend("pytesseract").name == "pytesseract"
    assert ob.get_backend("easyocr").name == "easyocr"
    monkeypatch.setitem(sys.modules, "easyocr", None)               # import fails -> next engine
    assert ob.get_backend().name == "pytesseract"
    assert ob.get_backend("easyocr") is None                         # forced engine unavailable: no silent substitution
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    assert ob.get_backend() is None
    assert ob.get_backend("nonsense") is None


def test_available_backends(monkeypatch):
    monkeypatch.setitem(sys.modules, "easyocr", None)
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    assert ob.available_backends() == []
    _fake_pytesseract(monkeypatch, {})
    assert ob.available_backends() == ["pytesseract"]
    _fake_easyocr(monkeypatch, [])
    assert ob.available_backends() == ["easyocr", "pytesseract"]


class _Boxes:
    name = "fake"

    def __init__(self, boxes):
        self.boxes, self.min_conf = boxes, None

    def detect(self, img, min_conf=0.3):
        self.min_conf = min_conf
        return self.boxes


def test_image_to_lines_groups_into_reading_order(monkeypatch):
    b = _Boxes([(100, 52, 40, 10, "B2", 0.9), (10, 50, 40, 10, "A1", 0.9),      # same line, out of x-order
                (10, 10, 40, 10, "TOP", 0.9), (5, 120, 30, 10, " last ", 0.9)])
    out = ob.image_to_lines(np.zeros((200, 200), np.uint8), backend=b)
    assert out == "TOP\nA1 B2\nlast" and b.min_conf == 0.2


def test_image_to_lines_no_backend_or_no_boxes(monkeypatch):
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: None)
    assert ob.image_to_lines(np.zeros((4, 4), np.uint8)) == ""
    assert ob.image_to_lines(np.zeros((4, 4), np.uint8), backend=_Boxes([])) == ""
    seen = {}
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: seen.update(p=prefer) or _Boxes([(0, 0, 5, 5, "x", 1)]))
    assert ob.image_to_lines(np.zeros((4, 4), np.uint8), prefer="easyocr") == "x" and seen["p"] == "easyocr"


def test_image_to_lines_line_tolerance_depends_on_box_height():
    far = [(0, 0, 10, 40, "A", 1.0), (50, 20, 10, 40, "B", 1.0)]             # centres 20 apart, tol = max(6, 40*0.6=24)
    assert ob.image_to_lines(np.zeros((9, 9), np.uint8), backend=_Boxes(far)) == "A B"
    small = [(0, 0, 10, 10, "A", 1.0), (50, 20, 10, 10, "B", 1.0)]           # tol = 6 -> two lines
    assert ob.image_to_lines(np.zeros((9, 9), np.uint8), backend=_Boxes(small)) == "A\nB"


# ---------------------------------- pixel_text ----------------------------------

def _arr(h=100, w=100, v=200):
    return np.full((h, w, 3), v, np.uint8)


def _use_backend(monkeypatch, boxes):
    b = _Boxes(boxes)
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: b)
    return b


def test_in_margin_geometry():
    shape = (100, 200)
    assert pt._in_margin((0, 40, 10, 10, "", 1), shape, 0.18)           # left (cx 5 < 36)
    assert pt._in_margin((190, 40, 8, 10, "", 1), shape, 0.18)           # right (cx 194 > 164)
    assert pt._in_margin((90, 2, 10, 10, "", 1), shape, 0.18)            # top (cy 7 < 18)
    assert pt._in_margin((90, 90, 10, 6, "", 1), shape, 0.18)            # bottom (cy 93 > 82)
    assert not pt._in_margin((90, 45, 20, 10, "", 1), shape, 0.18)       # centre


def test_redact_region_methods_and_clipping():
    for method in ("blackout", "blur", "pixelate"):
        a = np.random.default_rng(1).integers(50, 250, (60, 60, 3), dtype=np.uint8)
        before = a.copy()
        pt._redact_region(a, -5, -5, 30, 30, method)                       # clipped to the image
        assert not (a[:25, :25] == before[:25, :25]).all(), method
        assert (a[30:, 30:] == before[30:, 30:]).all(), method
    a = _arr(20, 20)
    pt._redact_region(a, 30, 30, 5, 5, "blackout")                          # fully outside
    pt._redact_region(a, 5, 5, 0, 5, "blackout")                            # empty
    assert (a == 200).all()
    with pytest.raises(ValueError, match="unknown redaction method: nope"):
        pt._redact_region(a, 0, 0, 5, 5, "nope")
    b = _arr(30, 30)
    pt._redact_region(b, 0, 0, 30, 30, "blackout")
    assert b.sum() == 0
    c = np.zeros((40, 40, 3), np.uint8)
    c[:, :20] = 255
    pt._redact_region(c, 0, 0, 40, 40, "pixelate")
    assert c.shape == (40, 40, 3)


def test_fallback_band_boxes():
    assert pt._fallback_band_boxes((100, 50), 0.2) == [(0, 0, 50, 20, "<top-band>", 1.0), (0, 80, 50, 20, "<bottom-band>", 1.0)]


def test_redact_array_with_ocr_margin_only_and_pad(monkeypatch):
    _use_backend(monkeypatch, [(2, 2, 20, 10, "NAME", 0.9), (40, 45, 20, 10, "ANATOMY", 0.9)])
    opts = pt.RedactOptions(upscale=1.0, pad=2)
    src = _arr()
    out, boxes = pt.redact_array(src, opts)
    assert [b[4] for b in boxes] == ["NAME"]
    assert out[5, 5].sum() == 0 and out[0, 0].sum() == 0 and out[50, 50].tolist() == [200, 200, 200]
    assert out[13, 23].sum() == 0 and out[14, 24].tolist() == [200, 200, 200]    # pad=2 grows the 20x10 box to x<24, y<14
    assert (src == 200).all()                                                     # input not mutated
    assert "ocr backend: fake" in opts.audit_log and "kept (central): 'ANATOMY'" in opts.audit_log
    assert any(line.startswith("redacted blackout: 'NAME' (0.90)") for line in opts.audit_log)
    assert not any("upscale" in line for line in opts.audit_log)


def test_redact_array_whole_image_redacts_central_text(monkeypatch):
    _use_backend(monkeypatch, [(40, 45, 20, 10, "ANATOMY", 0.9)])
    out, boxes = pt.redact_array(_arr(), pt.RedactOptions(upscale=1.0, margin_only=False))
    assert len(boxes) == 1 and out[50, 50].sum() == 0


def test_redact_array_upscale_maps_boxes_back(monkeypatch):
    b = _use_backend(monkeypatch, [(40, 40, 40, 20, "NAME", 0.9)])          # in 2x coordinates -> (20,20,20,10)
    opts = pt.RedactOptions(upscale=2.0, pad=0, margin_only=False, min_conf=0.55)
    out, boxes = pt.redact_array(_arr(), opts)
    assert boxes == [(20, 20, 20, 10, "NAME", 0.9)] and b.min_conf == 0.55
    assert out[25, 25].sum() == 0 and out[25, 45].tolist() == [200, 200, 200]
    assert "ocr upscale: 2x" in opts.audit_log


def test_redact_array_also_redact_bands_and_default_options(monkeypatch):
    _use_backend(monkeypatch, [])
    opts = pt.RedactOptions(upscale=1.0, also_redact_bands=True)
    out, boxes = pt.redact_array(_arr(), opts)
    assert [b[4] for b in boxes] == ["<top-band>", "<bottom-band>"]
    assert out[0, 0].sum() == 0 and out[99, 99].sum() == 0 and out[50, 50, 0] == 200
    assert "also redacted margin bands (belt-and-suspenders)" in opts.audit_log


def test_redact_array_no_backend_fallback_and_disabled(monkeypatch):
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: None)
    opts = pt.RedactOptions()
    out, boxes = pt.redact_array(_arr(), opts)
    assert len(boxes) == 2 and out[0, 0].sum() == 0 and out[99, 0].sum() == 0 and out[50, 0].tolist() == [200] * 3
    assert opts.audit_log == ["no OCR backend; using margin-band fallback"]
    opts2 = pt.RedactOptions(fallback_bands=False)
    out2, boxes2 = pt.redact_array(_arr(), opts2)
    assert boxes2 == [] and (out2 == 200).all() and opts2.audit_log == ["no OCR backend and fallback disabled; image unchanged"]
    out3, _ = pt.redact_array(_arr(), None)                              # default options
    assert out3[0, 0].sum() == 0


def test_redact_array_input_normalisation(monkeypatch):
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: None)
    gray, _ = pt.redact_array(np.full((50, 50), 7, np.uint8))
    assert gray.shape == (50, 50, 3)                                     # 2-D input becomes RGB
    flt, _ = pt.redact_array(np.linspace(0, 1, 2500).reshape(50, 50))
    assert flt.dtype == np.uint8 and flt[25, 25].max() > 0               # float input normalised to 0..255
    const, _ = pt.redact_array(np.zeros((50, 50)))
    assert const.max() == 0


def test_redact_image_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: None)
    src, dst = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(_arr()).save(src)
    opts = pt.redact_image_file(str(src), str(dst))
    assert isinstance(opts, pt.RedactOptions) and opts.audit_log
    arr = np.asarray(Image.open(dst))
    assert arr[0, 0].sum() == 0 and arr[50, 50].tolist() == [200] * 3


@pytest.mark.parametrize("frames,rgb", [(1, False), (2, False), (2, True), (1, True)])
def test_redact_dicom_frame_shapes(monkeypatch, tmp_path, frames, rgb):
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: None)
    src, dst = tmp_path / "in.dcm", tmp_path / "f.png"
    write_dicom(src, rows=60, cols=80, frames=frames, rgb=rgb)
    opts = pt.redact_dicom_frame(str(src), str(dst), frame=min(1, frames - 1))
    img = Image.open(dst)
    assert img.size == (80, 60) and opts.audit_log
    assert np.asarray(img)[0, 0].sum() == 0


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="BUG pixel_text.py redact_dicom_frame: a 3-frame grayscale stack (3, H, W) is mistaken "
                          "for a planar 3-channel image (shape[0] in (1, 3)), so no frame is selected and "
                          "Image.fromarray raises TypeError")
def test_redact_dicom_frame_three_frame_grayscale_stack(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "get_backend", lambda prefer=None, languages=None: None)
    src, dst = tmp_path / "in.dcm", tmp_path / "f.png"
    write_dicom(src, rows=60, cols=80, frames=3)
    pt.redact_dicom_frame(str(src), str(dst), frame=1)
    assert Image.open(dst).size == (80, 60)
