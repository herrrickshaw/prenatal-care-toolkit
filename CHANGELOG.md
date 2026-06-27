# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-27

Initial release. A defensive, analytical toolkit for prenatal imaging and data
that deliberately contains **no fetal-sex / fetal-genitalia detector**.

### Added
- **`pctk.deid`** (Problem 1) — DICOM header de-identification (PS3.15 basic
  subset, UID remapping, date policy) and burned-in pixel-text redaction with a
  pluggable OCR backend (easyocr / pytesseract), pre-OCR upscaling for recall,
  margin-only redaction, and a belt-and-suspenders band pass.
- **`pctk.srb`** (Problem 2) — sex-ratio-at-birth analytics with a one-sided
  binomial test, plus adapters for real public datasets: `crs`/`hmis`,
  `child06` (Census PCA 0–6, with significance testing), `nfhs`, `census`,
  `ratio`. `examples/fetch_data.py` pulls live NFHS-5 and Census PCA data.
- **`pctk.compliance`** (Problem 3) — PCPNDT Form-F + machine-registration audit
  rule engine, and Form-F ingestion from PDFs / scanned images (name is stored
  only as a one-way pseudonymous hash).
- **`pctk.biometry`** (Problem 4) — sex-neutral Hadlock gestational-age and
  estimated-fetal-weight calculations.
- Unified `pctk` CLI, Docker image (Tesseract bundled), CI + GHCR publish
  workflows, and a smoke-test suite (10 tests).

[0.1.0]: https://github.com/herrrickshaw/prenatal-care-toolkit/releases/tag/v0.1.0
