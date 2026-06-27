"""Generate synthetic PCPNDT Form F samples for ingestion testing.

Writes, under ``data/``:
  * ``form_f_sample.pdf`` - a *digital* PDF (text extractable, no OCR needed),
  * ``form_f_sample.png`` - an *image* of the same form (OCR path).

Field values are fictional. The pregnant woman's name is included only so the
ingestion pipeline can demonstrate converting it to a pseudonymous hash.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

LINES = [
    "FORM F  (PCPNDT Act, 1994)",
    "",
    "Name of pregnant woman: PRIYA SHARMA",
    "Date of test: 27-06-2026",
    "Referred by: Dr A Mehta",
    "Indication for ultrasound: Anomaly scan",
    "Period of gestation: 19 weeks",
    "Performed by: Dr S Rao",
    "USG Machine No: US-100",
    "Centre Registration No: FAC-A",
]


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_image() -> str:
    W, H = 720, 460
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = _font(22)
    y = 24
    for line in LINES:
        draw.text((36, y), line, fill="black", font=font)
        y += 40
    path = os.path.join(DATA, "form_f_sample.png")
    img.save(path)
    return path


def make_pdf() -> str:
    path = os.path.join(DATA, "form_f_sample.pdf")
    try:
        import fitz  # PyMuPDF
    except Exception:
        # Fall back: save the image as a PDF (will be image-only / needs OCR).
        Image.open(make_image()).convert("RGB").save(path)
        return path
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    y = 60
    for line in LINES:
        page.insert_text((50, y), line, fontsize=14)
        y += 28
    doc.save(path)
    doc.close()
    return path


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    print("wrote image:", make_image())
    print("wrote pdf:  ", make_pdf())


if __name__ == "__main__":
    main()
