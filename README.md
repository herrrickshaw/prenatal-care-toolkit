# prenatal-care-toolkit (`pctk`)

A toolkit of **defensive** and **analytical** utilities for prenatal imaging and
data, built to help prevent sex-selective abortion (female foeticide) and to
protect patient privacy.

## What this is — and what it deliberately is *not*

Female foeticide is driven by **prenatal sex determination**, which India's
**PCPNDT Act, 1994** criminalises. A natural temptation is to build an image
model that "detects fetal genitalia so it can blur them." This toolkit refuses
that approach, on purpose:

- **A genitalia/sex detector *is* the harmful tool.** To blur a region you must
  first locate it; that locator is exactly the sex-determination capability the
  law bans, trivially reused to *reveal* sex. So **this toolkit contains no
  fetal-sex or fetal-genitalia detector.**
- **It also wouldn't help.** Sex determination happens live, through a trained
  sonographer's eyes during the scan — not by running saved images through a
  model later. Blurring published images stops no one.

Instead, `pctk` attacks the problem where it is actually fightable, across four
modules:

| # | Module | Purpose |
|---|--------|---------|
| 1 | `pctk.deid` | Remove patient-identifying info (PHI) from DICOM headers **and** burned-in pixel text |
| 2 | `pctk.srb` | Surface anomalous **sex-ratio-at-birth** signals for PCPNDT auditors |
| 3 | `pctk.compliance` | **PCPNDT Form-F** and machine-registration audit rule engine |
| 4 | `pctk.biometry` | Standard, **sex-neutral** fetal biometry & gestational age |

None of the modules look at, infer, or expose fetal sex.

## Install

```bash
pip install -e .          # from this directory
# or just the deps:
pip install -r requirements.txt
```

Optional OCR engine for burned-in text (Problem 1) and Form-F image ingestion
(Problem 3) — install **one**:

```bash
pip install easyocr                       # pure pip (large: torch), no binary
pip install pytesseract                   # small; needs the tesseract binary:
#   macOS:  brew install tesseract        #   (Linux: apt install tesseract-ocr)
```

Optional, to ingest Form F from PDFs (Problem 3):

```bash
pip install pypdf pymupdf                 # digital + scanned PDF support
```

Without an OCR engine, pixel redaction falls back to blanking the standard
top/bottom overlay bands.

## Quick start

```bash
# generate synthetic sample data (no real patient data needed)
python examples/make_sample_data.py

# Problem 1 — de-identify a DICOM header, then redact burned-in pixel text
pctk deid-dicom data/sample_ultrasound.dcm data/clean.dcm -v
pctk deid-image data/sample_ultrasound.png data/clean.png --method blur \
                --ocr-backend pytesseract --also-redact-bands
pctk deid-frame data/sample_ultrasound.dcm data/clean_frame.png

# Problem 2 — sex-ratio-at-birth anomaly report
pctk srb data/births.csv --group district --out data/srb_report.csv
pctk srb data/crs_births.csv --source crs            # real CRS/HMIS schema
pctk srb data/census_csr.csv --source census         # ratio-only source

# Problem 3 — ingest a Form F from PDF/image, then audit
pctk ingest-formf data/form_f_sample.pdf --append-csv data/forms_ingested.csv
pctk compliance   data/forms.csv data/registrations.csv -v

# Problem 4 — gestational age & estimated fetal weight
pctk biometry --bpd 46 --hc 170 --ac 150 --fl 30

pctk backends      # show which OCR engines are available
```

## Module details

### 1. `pctk.deid` — de-identification

- **Header** (`dicom_scrub`): a practical subset of the DICOM PS3.15 *Basic
  Application Level Confidentiality Profile* — curated PHI action table, a
  Person-Name (VR `PN`) catch-all, private-tag removal, deterministic UID
  remapping (referential integrity preserved), and a configurable date policy
  (`keep` / `blank` / `shift`-by-consistent-offset). Clinically useful tags
  (modality, patient age/size/weight, ultrasound region sequence) are retained.
  Every action is recorded in an audit log.
- **Pixel text** (`pixel_text`): OCR-detects identifying overlays and redacts
  them (`blackout` / `blur` / `pixelate`). `margin_only=True` (default) confines
  redaction to the image margins so the central anatomy is never altered. Images
  are **upscaled before OCR** (`upscale=2.0`) to boost recall on small overlay
  fonts, and `also_redact_bands=True` adds a belt-and-suspenders pass that blanks
  the overlay bands so imperfect OCR recall can't leak margin PHI. OCR backend is
  pluggable (`easyocr` / `pytesseract`); with none installed it falls back to
  band redaction.

```python
from pctk.deid import deidentify_file, DeidOptions, redact_image_file, RedactOptions
deidentify_file("in.dcm", "out.dcm", DeidOptions(date_policy="shift"))
redact_image_file("in.png", "out.png",
                  RedactOptions(method="blur", also_redact_bands=True))
```

### 2. `pctk.srb` — sex-ratio-at-birth analytics

Computes SRB by any grouping dimension and runs a one-sided binomial test
against the natural baseline (~48.8% female, ≈105♂:100♀), ranking units that are
significantly skewed toward males.

**Input schema (per-birth):** a `sex` column (`M`/`F`, `1`/`2`, etc.), one or
more group columns (`district`, `clinic`, `year` …), optional `count` column.

**Real public datasets** rarely ship one row per birth — they ship aggregated
counts or a ready-made ratio. `pctk.srb.adapters` maps the published schemas:

| `--source` | Source | Test? |
|------------|--------|-------|
| `crs` / `hmis` | Civil Registration System / HMIS — male & female birth counts by district/year | ✅ binomial (has counts) |
| `child06` | Census PCA **0–6 child** population by sex (`M_06`/`F_06`) — the canonical foeticide proxy | ✅ binomial (has counts) |
| `nfhs` | NFHS-4/5 district factsheets — "Sex ratio at birth, last 5 yrs (females/1,000 males)" | ratio-only |
| `census` | Census of India child sex ratio (0–6), girls per 1,000 boys | ratio-only |
| `ratio` | data.gov.in "Sex Ratio at Birth" resources | ratio-only |

```bash
pctk srb nfhs5_districts.csv  --source nfhs       # real NFHS-5 factsheet CSV
pctk srb census_pca_06.csv    --source child06    # 0-6 counts -> significance
```

**Fetch the real data with one command** — `examples/fetch_data.py` pulls live
government-derived datasets into `data/`:

```bash
python examples/fetch_data.py                      # both sources
pctk srb data/nfhs5_districts.csv --source nfhs
pctk srb data/census_pca06.csv    --source child06 --group State Name
```

| File | Source | Provenance |
|------|--------|-----------|
| `nfhs5_districts.csv` | 21 per-state NFHS-5 district factsheets, concatenated (340 districts) | [pratapvardhan/NFHS-5](https://github.com/pratapvardhan/NFHS-5) ← [rchiips.org/nfhs](http://rchiips.org/nfhs/) |
| `census_pca06.csv` | Census 2011 PCA district totals incl. `M_06`/`F_06` (640 districts) | [pigshell/india-census-2011](https://github.com/pigshell/india-census-2011) ← Census of India |

On the real PCA-0-6 file the pipeline flags 386/640 districts; the worst child
sex ratios (Surat ~835, Jaipur ~861, Ahmedabad, Pune, Agra) match documented
patterns, and the national 0–6 female share (~0.479) reproduces the published
2011 child sex ratio (~919 girls per 1,000 boys).

```python
import pandas as pd
from pctk.srb import from_crs_hmis, flag_anomalies, SRBConfig
df = pd.read_csv("india_census_2011.csv")            # 640 real districts
table = from_crs_hmis(df, group_cols=["State name", "District name"])
flagged = flag_anomalies(table, SRBConfig(group_cols=["State name","District name"]))
```

> **Interpretation caveat:** the 0.488 female baseline is for *births*. Run the
> significance test on **birth** counts (CRS/HMIS) or the Census **0–6 child**
> ratio. Applying it to *total* population conflates foeticide with labour
> migration and differential mortality — use SRB/CSR, not headcount.

### 3. `pctk.compliance` — PCPNDT audit + Form-F ingestion

A lightweight data model (`FormF`, `MachineRegistration`) plus a rule engine
flagging: incomplete Form F, vague/non-clinical indication, implausible
gestational age, scans on unregistered/lapsed machines, and records past their
retention window. Returns findings plus a per-facility completeness summary.

**Ingestion** (`ingest`): turn a Form F **PDF or scanned image** into a record.
Digital PDFs are read directly (`pypdf`); scanned PDFs are rasterised
(`pymupdf`) and OCR'd; images are OCR'd. Fields are pulled with label-aliased
regexes — and the pregnant woman's **name is never stored**, only a one-way
pseudonymous `patient_ref` hash. Extracted records drop straight into the
auditor:

```bash
pctk ingest-formf scan.pdf --append-csv forms.csv
pctk compliance forms.csv registrations.csv
```

### 4. `pctk.biometry` — sex-neutral biometry

Hadlock-style gestational-age regressions from BPD/HC/AC/FL and Hadlock-1985
estimated fetal weight. Educational sanity-check, not a clinical report.

## Testing

```bash
python tests/test_smoke.py     # or: pytest -q
```

## Responsible-use notes

- Run de-identification **before** any data leaves a clinical environment.
- The SRB and compliance modules are **decision-support for authorised
  auditors**; a statistical flag is a prompt to investigate, not proof of
  wrongdoing.
- Replace the example UID root in `dicom_scrub.PCTK_UID_ROOT` with your own
  registered root before production use.
- This software is provided for research, education, and authorised public-health
  / regulatory use. It is not a medical device.

## License

MIT
