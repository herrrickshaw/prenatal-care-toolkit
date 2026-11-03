# prenatal-care-toolkit (`pctk`)

[![CI](https://github.com/herrrickshaw/prenatal-care-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/herrrickshaw/prenatal-care-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/herrrickshaw/prenatal-care-toolkit/blob/main/notebooks/colab_test.ipynb)

A Python package and `pctk` command-line tool of **defensive** utilities for prenatal imaging and
data, aimed at preventing sex-selective abortion in India (PCPNDT Act, 1994) and protecting patient
privacy. It de-identifies DICOM headers and burned-in image text, computes sex-ratio-at-birth (SRB)
anomaly reports from public datasets, audits PCPNDT Form-F records, and provides sex-neutral fetal
biometry plus two sex-neutral classifiers (anatomical plane, CTG wellbeing). Outputs are DICOM/PNG
files, CSV reports and console summaries.

> Status: active. Research, education and authorised public-health / regulatory use only; not a
> medical device. Tests and examples use **synthetic data only**.

## No fetal-sex detector, by design

Female foeticide is driven by prenatal sex determination. A natural-looking idea is an image model
that "finds fetal genitalia so it can blur them". This toolkit refuses that on purpose:

- A genitalia / sex detector **is** the harmful tool. To blur a region you must first locate it, and
  that locator is exactly the capability the PCPNDT Act criminalises, trivially reused to reveal sex.
  So `pctk` contains **no fetal-sex or fetal-genitalia detector, and none should be added**.
- It would not help anyway: sex determination happens live, through the sonographer's eyes, not by
  running saved images through a model afterwards.

None of the modules reads, infers or exposes fetal sex. The plane and CTG classifiers have no sex
label, and the de-identifier blanks the mother's `PatientSex` by default.

## What it does
- Removes PHI from DICOM headers (PS3.15 basic-profile subset, UID remapping, date shift/blank/keep).
- OCR-detects and redacts burned-in identifying text in ultrasound pixels (blackout / blur / pixelate).
- Flags districts whose female birth share is significantly below the natural baseline (~0.488).
- Audits PCPNDT Form-F records against machine registrations and retention rules; ingests Form F
  from PDF/image with the woman's name replaced by a one-way `patient_ref` hash.
- Estimates gestational age and fetal weight from BPD/HC/AC/FL (Hadlock-style, educational).
- Trains and runs a fetal anatomical-plane classifier and a CTG fetal-health classifier.

## Quick start
```bash
git clone https://github.com/herrrickshaw/prenatal-care-toolkit.git
cd prenatal-care-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # or: pip install -r requirements.txt
python examples/make_sample_data.py   # synthetic DICOM, PNG, births, forms CSVs (into data/)
python examples/make_sample_formf.py  # synthetic Form-F PDF/PNG (for ingest-formf)
pctk deid-dicom data/sample_ultrasound.dcm data/clean.dcm -v
```

More commands:

```bash
pctk deid-image data/sample_ultrasound.png data/clean.png --method blur --also-redact-bands
pctk deid-frame data/sample_ultrasound.dcm data/clean_frame.png
pctk srb data/births.csv --group district --out data/srb_report.csv
pctk srb data/census_pca06.csv --source child06 --group "State Name"
pctk ingest-formf data/form_f_sample.pdf --append-csv data/forms_ingested.csv
pctk compliance data/forms.csv data/registrations.csv -v
pctk biometry --bpd 46 --hc 170 --ac 150 --fl 30
pctk planes-demo        # synthetic end-to-end plane classifier (needs the [ml] extra)
pctk health-demo        # synthetic end-to-end CTG classifier (needs the [ml] extra)
pctk backends           # which OCR engines are installed
```

Optional extras: `.[ocr-easy]` (easyocr, pulls torch), `.[ocr-tesseract]` (needs the `tesseract`
binary), `.[pdf]` (pypdf + pymupdf), `.[ml]` (scikit-learn + joblib). Without an OCR engine, pixel
redaction falls back to blanking the standard top/bottom overlay bands. A prebuilt image with
Tesseract is published as `ghcr.io/herrrickshaw/prenatal-care-toolkit` (entrypoint is `pctk`;
`docker run --rm ghcr.io/herrrickshaw/prenatal-care-toolkit:latest --version`). GHCR packages start
private; make the package public once under repo Packages > Package settings.

## How it works

| Stage | Module | Output |
|---|---|---|
| Header de-id | `pctk.deid.dicom_scrub` | cleaned DICOM + audit log |
| Pixel-text redaction | `pctk.deid.pixel_text`, `pctk.deid.ocr_backends` | redacted image + boxes |
| SRB analytics | `pctk.srb.analyze`, `pctk.srb.adapters` | per-unit SRB table, binomial p-values |
| Form-F ingest | `pctk.compliance.ingest` | `FormF` record (`patient_ref` hash only) |
| Compliance audit | `pctk.compliance.formf`, `pctk.compliance.io` | findings + per-facility summary |
| Biometry | `pctk.biometry.gestation` | gestational age, estimated fetal weight |
| Plane / CTG models | `pctk.planes`, `pctk.health` | saved sklearn / torch models, reports |

Design notes:
- **Header de-id** uses a curated PHI action table (`X` remove, `Z` blank, `D` dummy, `U` remap UID,
  date policy), a Person-Name (VR `PN`) catch-all, private-tag removal and deterministic UID remapping
  so references inside a study survive. Clinically useful tags (modality, age, size, weight,
  ultrasound region sequence) are retained.
- **Pixel text** is OCR'd after a 2x upscale for recall; `margin_only=True` keeps central anatomy
  untouched; `also_redact_bands=True` blanks the overlay bands regardless of OCR recall. Redaction
  methods live in a `REDACTORS` registry. OCR backends (`easyocr`, `pytesseract`) are pluggable.
- **SRB** adapters map public schemas to one table: `crs`/`hmis` and `child06` carry counts (binomial
  test); `nfhs`, `census`, `ratio` are ratio-only (no test, flagged `counts_available=False`).
- **Compliance** rules are pure functions (`validate_form_f`, `check_registration`, `check_retention`);
  CSV reading/writing is isolated in `pctk.compliance.io`.

## Repository layout
| Path | Purpose |
|---|---|
| `pctk/deid/` | DICOM header scrub, pixel-text redaction, OCR backends |
| `pctk/srb/` | SRB computation, anomaly flagging, public-dataset adapters |
| `pctk/compliance/` | Form-F model, audit rules, ingestion, CSV I/O |
| `pctk/biometry/`, `pctk/planes/`, `pctk/health/` | biometry and the two sex-neutral classifiers |
| `pctk/cli.py` | `pctk` command-line entry point |
| `examples/` | synthetic-data generators and `fetch_data.py` (public datasets) |
| `tests/` | smoke, unit and characterisation tests (`tests/golden/baseline.json`) |
| `data/` | generated, gitignored |

## Data sources
| Source | What is taken | Access route | Licence / caveat |
|---|---|---|---|
| NFHS-5 district factsheets | SRB (females per 1,000 males), 340 districts | `examples/fetch_data.py` from [pratapvardhan/NFHS-5](https://github.com/pratapvardhan/NFHS-5), originally [rchiips.org/nfhs](http://rchiips.org/nfhs/) | Third-party mirror; check upstream terms |
| Census 2011 PCA | District totals incl. `M_06`/`F_06`, 640 districts | `examples/fetch_data.py` from [pigshell/india-census-2011](https://github.com/pigshell/india-census-2011) | Third-party mirror; check upstream terms |
| FETAL_PLANES_DB | Plane-labelled ultrasound images | [Zenodo 3904280](https://zenodo.org/record/3904280), see [DATASETS.md](DATASETS.md) | Dataset licence applies |
| Fetal Health Classification (CTG) | 21 SisPorto features | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) | Needs a Kaggle token |

On the real PCA 0-6 file the pipeline flags 386 of 640 districts; the national 0-6 female share
(~0.479) reproduces the published 2011 child sex ratio (~919 girls per 1,000 boys).

## Configuration
No environment variables and no secrets. Behaviour is set through CLI flags and the option
dataclasses `DeidOptions`, `RedactOptions` and `SRBConfig`. Replace the example UID root
`dicom_scrub.PCTK_UID_ROOT` with your own registered root before production use.

## Testing
```bash
pytest -q          # 30 tests, synthetic data only
ruff check .
```

## Limitations and known issues
- Run the SRB significance test on **birth** counts (CRS/HMIS) or the Census 0-6 child ratio, not
  total population, which conflates foeticide with migration and differential mortality.
- SRB and compliance outputs are decision support for authorised auditors; a flag is a prompt to
  investigate, not proof of wrongdoing.
- Run de-identification before any data leaves a clinical environment; header de-id does not touch
  burned-in pixels, and OCR recall is imperfect (use `--also-redact-bands`).
- The retention check raises `ValueError` for a 29 February scan date when the target year is not a
  leap year (pre-existing).
- Biometry is an educational sanity check, not a clinical report.

## Licence
MIT, see [LICENSE](LICENSE). More: [DATASETS.md](DATASETS.md), [CHANGELOG.md](CHANGELOG.md),
[HOW_TO_USE_WITH_AI.md](HOW_TO_USE_WITH_AI.md).
