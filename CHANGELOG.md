# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-07-02

### Fixed
- `cli.py` missing top-level `import os`, which crashed `health-demo` (and any
  command using a temp-dir default path) with `NameError: name 'os' is not
  defined`. Added a CLI smoke test path without `--data` to catch it.

## [0.3.0] - 2026-06-27

### Added
- **`pctk.health`** — sex-neutral fetal-wellbeing (CTG/cardiotocography)
  classifier (normal / suspect / pathological) on the 21 SisPorto features;
  StandardScaler → GradientBoosting, with feature importances. CLI:
  `health-demo`, `health-train`, `health-predict`.
- **Optional PyTorch CNN backend** for `pctk.planes` (`CNNPlaneClassifier`),
  behind the same `planes-train/eval/predict` interface via `--backend torch`.
  Torch is imported lazily (package still works without it); a dedicated
  `cnn-backend` CI job exercises the path on CPU wheels.

### Notes
- Neither new component has a fetal-sex target — health status and anatomical
  planes only.

## [0.2.0] - 2026-06-27

### Added
- **`pctk.planes`** — sex-neutral fetal **anatomical-plane classifier**
  (FETAL_PLANES_DB: brain / abdomen / femur / thorax / maternal cervix / other).
  Dependency-light scikit-learn pipeline (numpy HOG-style features →
  RandomForest); trains, evaluates, persists (joblib), and predicts on CPU.
- CLI: `planes-demo` (synthetic end-to-end, no download), `planes-train`,
  `planes-eval`, `planes-predict`.
- Kaggle / Zenodo download helpers (`pctk.planes.download`) and **DATASETS.md**
  cataloguing the public datasets backing each module.
- `ml` extra (scikit-learn, joblib); image + CI now install it.

### Notes
- The plane classifier has **no fetal-sex target** — labels are anatomical
  planes only, consistent with the project's design stance.

## [0.1.1] - 2026-06-27

### Added
- README status badges (CI, Docker publish, GHCR, license, Python) and a
  **Docker usage** section (pull/run/build, tag scheme, image-visibility note).

### Notes
- GHCR container packages default to private even on a public repo and cannot be
  made public via the workflow token; documented the one-time manual toggle.

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

[0.3.1]: https://github.com/herrrickshaw/prenatal-care-toolkit/releases/tag/v0.3.1
[0.3.0]: https://github.com/herrrickshaw/prenatal-care-toolkit/releases/tag/v0.3.0
[0.2.0]: https://github.com/herrrickshaw/prenatal-care-toolkit/releases/tag/v0.2.0
[0.1.1]: https://github.com/herrrickshaw/prenatal-care-toolkit/releases/tag/v0.1.1
[0.1.0]: https://github.com/herrrickshaw/prenatal-care-toolkit/releases/tag/v0.1.0
