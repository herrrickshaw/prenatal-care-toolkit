"""Detect and redact patient-identifying text burned into ultrasound pixels.

Ultrasound machines overlay the patient name, MRN, clinic, operator and date
directly onto the image margins. Those pixels survive header de-identification,
so they must be removed separately.

Pipeline:
  1. load an image (PNG/JPG) or a frame extracted from a DICOM,
  2. find text boxes with an OCR backend (``ocr_backends``),
  3. redact each box (blackout / blur / pixelate),
  4. optionally restrict redaction to the margins so the central anatomy is
     never touched,
  5. when no OCR engine is installed, fall back to redacting the standard
     top/bottom overlay bands.

Implemented with PIL + numpy only (no OpenCV dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from . import ocr_backends

TextBox = Tuple[int, int, int, int, str, float]

# Pixelation block size in pixels.
PIXELATE_BLOCK = 12


@dataclass
class RedactOptions:
    method: str = "blackout"          # "blackout" | "blur" | "pixelate"
    pad: int = 4                       # grow each box by this many px
    min_conf: float = 0.3
    margin_only: bool = True          # only redact text in the outer margins
    margin_frac: float = 0.18         # band thickness as fraction of dimension
    ocr_backend: Optional[str] = None  # force "easyocr"/"pytesseract" or auto
    languages: Optional[List[str]] = None
    fallback_bands: bool = True       # redact overlay bands if OCR unavailable
    upscale: float = 2.0              # enlarge before OCR to boost recall (1=off)
    also_redact_bands: bool = False   # belt-and-suspenders: blank margin bands too
    audit_log: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.audit_log.append(msg)


def _in_margin(box: TextBox, shape: Tuple[int, int], frac: float) -> bool:
    x, y, w, h = box[0], box[1], box[2], box[3]
    H, W = shape[:2]
    mx, my = int(W * frac), int(H * frac)
    cx, cy = x + w / 2.0, y + h / 2.0
    return (cx < mx) or (cx > W - mx) or (cy < my) or (cy > H - my)


def _blackout(region: np.ndarray) -> np.ndarray:
    return np.zeros_like(region)


def _blur(region: np.ndarray) -> np.ndarray:
    radius = max(4, min(region.shape[1], region.shape[0]) // 3)
    blurred = Image.fromarray(region).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred)


def _pixelate(region: np.ndarray) -> np.ndarray:
    h, w = region.shape[:2]
    small = Image.fromarray(region).resize(
        (max(1, w // PIXELATE_BLOCK), max(1, h // PIXELATE_BLOCK)), Image.NEAREST)
    return np.asarray(small.resize((w, h), Image.NEAREST))


# method name -> function mapping a region to its redacted replacement
REDACTORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "blackout": _blackout,
    "blur": _blur,
    "pixelate": _pixelate,
}


def _redact_region(arr: np.ndarray, x: int, y: int, w: int, h: int,
                   method: str) -> None:
    H, W = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    try:
        redactor = REDACTORS[method]
    except KeyError:
        raise ValueError(f"unknown redaction method: {method}") from None
    arr[y0:y1, x0:x1] = redactor(arr[y0:y1, x0:x1])


def _fallback_band_boxes(shape: Tuple[int, int], frac: float) -> List[TextBox]:
    """Top and bottom overlay bands, used when OCR is unavailable."""
    H, W = shape[:2]
    band = int(H * frac)
    return [
        (0, 0, W, band, "<top-band>", 1.0),
        (0, H - band, W, band, "<bottom-band>", 1.0),
    ]


def _as_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Copy ``arr`` as an HxWx3 uint8 image (grayscale stacked, floats rescaled)."""
    work = np.array(arr, copy=True)
    if work.ndim == 2:
        work = np.stack([work] * 3, axis=-1)
    if work.dtype != np.uint8:
        a = work.astype(np.float64)
        a -= a.min()
        work = (a / (a.max() or 1.0) * 255).astype(np.uint8)
    return work


def _detect_scaled(backend, work: np.ndarray, opts: RedactOptions) -> List[TextBox]:
    """Run OCR (optionally on an upscaled copy); boxes are in ``work`` coordinates."""
    # Upscale before OCR to improve recall on small overlay fonts, then map
    # the detected boxes back to original-resolution coordinates.
    scale = max(1.0, float(opts.upscale))
    if scale > 1.0:
        H, W = work.shape[:2]
        big = np.asarray(
            Image.fromarray(work).resize(
                (int(W * scale), int(H * scale)), Image.LANCZOS)
        )
        opts.log(f"ocr upscale: {scale:g}x")
    else:
        big = work
    boxes = []
    for x, y, w, h, text, conf in backend.detect(big, min_conf=opts.min_conf):
        if scale > 1.0:
            x, y, w, h = (int(x / scale), int(y / scale),
                          int(w / scale), int(h / scale))
        boxes.append((x, y, w, h, text, conf))
    return boxes


def _redact_bands(work: np.ndarray, opts: RedactOptions) -> List[TextBox]:
    """Blank the standard top/bottom overlay bands; returns the boxes redacted."""
    bands = _fallback_band_boxes(work.shape, opts.margin_frac)
    for bx, by, bw, bh, _, _ in bands:
        _redact_region(work, bx, by, bw, bh, opts.method)
    return bands


def _redact_with_ocr(backend, work: np.ndarray, opts: RedactOptions) -> List[TextBox]:
    opts.log(f"ocr backend: {backend.name}")
    redacted: List[TextBox] = []
    for obox in _detect_scaled(backend, work, opts):
        x, y, w, h, text, conf = obox
        if opts.margin_only and not _in_margin(obox, work.shape, opts.margin_frac):
            opts.log(f"kept (central): {text!r}")
            continue
        p = opts.pad
        _redact_region(work, x - p, y - p, w + 2 * p, h + 2 * p, opts.method)
        redacted.append(obox)
        opts.log(f"redacted {opts.method}: {text!r} ({conf:.2f})")
    # Belt-and-suspenders: OCR recall is imperfect, so optionally blank the
    # standard overlay bands as well to guarantee no margin PHI survives.
    if opts.also_redact_bands:
        redacted.extend(_redact_bands(work, opts))
        opts.log("also redacted margin bands (belt-and-suspenders)")
    return redacted


def redact_array(arr: np.ndarray,
                 opts: Optional[RedactOptions] = None) -> Tuple[np.ndarray, List[TextBox]]:
    """Redact identifying text in an image array.

    Returns ``(redacted_array, boxes_redacted)``. The input is not mutated.
    """
    opts = opts or RedactOptions()
    work = _as_rgb_uint8(arr)
    backend = ocr_backends.get_backend(opts.ocr_backend, opts.languages)

    if backend is not None:
        return work, _redact_with_ocr(backend, work, opts)
    if opts.fallback_bands:
        opts.log("no OCR backend; using margin-band fallback")
        return work, _redact_bands(work, opts)
    opts.log("no OCR backend and fallback disabled; image unchanged")
    return work, []


def redact_image_file(in_path: str,
                      out_path: str,
                      opts: Optional[RedactOptions] = None) -> RedactOptions:
    """Redact identifying text in a standard image file (PNG/JPG/...)."""
    opts = opts or RedactOptions()
    img = Image.open(in_path).convert("RGB")
    arr = np.asarray(img)
    out, _ = redact_array(arr, opts)
    Image.fromarray(out).save(out_path)
    return opts


def redact_dicom_frame(in_path: str,
                       out_image_path: str,
                       frame: int = 0,
                       opts: Optional[RedactOptions] = None) -> RedactOptions:
    """Extract a frame from a DICOM, redact burned-in text, save as an image.

    Saving the shareable artifact as a plain image avoids the complexity of
    re-encoding compressed DICOM pixel data. Pair this with
    :func:`pctk.deid.dicom_scrub.deidentify_file` for the header.
    """
    import pydicom

    opts = opts or RedactOptions()
    ds = pydicom.dcmread(in_path)
    px = ds.pixel_array
    if px.ndim == 4:           # (frames, H, W, ch)
        px = px[frame]
    elif px.ndim == 3 and px.shape[0] not in (1, 3) and px.shape[-1] not in (3, 4):
        px = px[frame]         # (frames, H, W) grayscale multi-frame
    out, _ = redact_array(px, opts)
    Image.fromarray(out).save(out_image_path)
    return opts
