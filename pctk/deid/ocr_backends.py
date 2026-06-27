"""Pluggable OCR backends for burned-in text detection.

A backend takes a HxW or HxWx3 ``numpy`` array (uint8) and returns a list of
``(x, y, w, h, text, confidence)`` tuples in pixel coordinates.

Backends, in order of preference:
  * ``easyocr``    - pip-installable, no system binary, decent on overlays
  * ``pytesseract``- needs the Tesseract binary on PATH
  * ``none``       - no OCR available; callers fall back to margin redaction

Nothing here looks at anatomy; it only finds text.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

Box = Tuple[int, int, int, int, str, float]  # x, y, w, h, text, confidence


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        # Normalise to 0..255 for OCR engines.
        a = arr.astype(np.float64)
        a -= a.min()
        m = a.max() or 1.0
        arr = (a / m * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


class _EasyOCRBackend:
    name = "easyocr"

    def __init__(self, languages: Optional[List[str]] = None):
        import easyocr  # noqa: F401  (import error handled by caller)

        self._reader = easyocr.Reader(languages or ["en"], gpu=False, verbose=False)

    def detect(self, img: np.ndarray, min_conf: float = 0.3) -> List[Box]:
        rgb = _to_rgb_uint8(img)
        out: List[Box] = []
        for bbox, text, conf in self._reader.readtext(rgb):
            if conf < min_conf or not str(text).strip():
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            out.append((x, y, w, h, str(text), float(conf)))
        return out


class _TesseractBackend:
    name = "pytesseract"

    def __init__(self, languages: Optional[List[str]] = None):
        import pytesseract  # noqa: F401

        self._pt = pytesseract
        self._lang = "+".join(languages or ["eng"])

    def detect(self, img: np.ndarray, min_conf: float = 0.3) -> List[Box]:
        from PIL import Image

        rgb = _to_rgb_uint8(img)
        data = self._pt.image_to_data(
            Image.fromarray(rgb),
            lang=self._lang,
            output_type=self._pt.Output.DICT,
        )
        out: List[Box] = []
        n = len(data["text"])
        for i in range(n):
            text = str(data["text"][i]).strip()
            try:
                conf = float(data["conf"][i]) / 100.0
            except (TypeError, ValueError):
                conf = 0.0
            if not text or conf < min_conf:
                continue
            out.append(
                (
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                    text,
                    conf,
                )
            )
        return out


def get_backend(prefer: Optional[str] = None,
                languages: Optional[List[str]] = None):
    """Return the first available OCR backend, or ``None``.

    ``prefer`` can be ``"easyocr"`` or ``"pytesseract"`` to force a choice.
    """
    order = [prefer] if prefer else ["easyocr", "pytesseract"]
    order = [b for b in order if b]
    if not prefer:
        order = ["easyocr", "pytesseract"]
    for name in order:
        try:
            if name == "easyocr":
                return _EasyOCRBackend(languages)
            if name == "pytesseract":
                return _TesseractBackend(languages)
        except Exception:
            continue
    return None


def image_to_lines(img: np.ndarray,
                   backend=None,
                   prefer: Optional[str] = None,
                   min_conf: float = 0.2) -> str:
    """OCR an image and reconstruct reading-order text (line by line).

    Groups detected word-boxes into lines by their vertical position, orders
    each line left-to-right, and joins with spaces / newlines. Backend-agnostic,
    so it works with whichever engine is installed.
    """
    backend = backend or get_backend(prefer)
    if backend is None:
        return ""
    boxes = backend.detect(_to_rgb_uint8(img), min_conf=min_conf)
    if not boxes:
        return ""
    # Sort by vertical centre, then cluster into lines.
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3] / 2.0))
    heights = [b[3] for b in boxes if b[3] > 0] or [10]
    tol = max(6, int(sorted(heights)[len(heights) // 2] * 0.6))
    lines: List[List[Box]] = []
    for b in boxes:
        cy = b[1] + b[3] / 2.0
        if lines and abs((lines[-1][0][1] + lines[-1][0][3] / 2.0) - cy) <= tol:
            lines[-1].append(b)
        else:
            lines.append([b])
    out_lines = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda b: b[0])
        out_lines.append(" ".join(str(b[4]).strip() for b in ln_sorted))
    return "\n".join(out_lines)


def available_backends() -> List[str]:
    found = []
    for name, mod in (("easyocr", "easyocr"), ("pytesseract", "pytesseract")):
        try:
            __import__(mod)
            found.append(name)
        except Exception:
            pass
    return found
