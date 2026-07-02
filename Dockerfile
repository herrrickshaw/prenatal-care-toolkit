# prenatal-care-toolkit container image.
#
# Includes the Tesseract OCR engine so burned-in pixel-text redaction
# (Problem 1) and Form-F image ingestion (Problem 3) work out of the box.
# easyocr/torch are intentionally NOT installed to keep the image small;
# the pytesseract backend is wired in via the [ocr-tesseract] extra.
FROM python:3.11-slim AS base

# System dependency: Tesseract OCR.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the package with the OCR + PDF extras. Copy only what the build
# needs first so the layer caches well.
COPY pyproject.toml README.md ./
COPY pctk ./pctk
RUN pip install --no-cache-dir ".[ocr-tesseract,pdf,ml]"

# Bring in the examples (sample-data + fetch helpers) for convenience.
COPY examples ./examples

# Run as a non-root user.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default to the CLI; override the args at `docker run`.
ENTRYPOINT ["pctk"]
CMD ["--help"]
