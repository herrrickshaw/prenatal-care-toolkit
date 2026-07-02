# Datasets

Public datasets that back the toolkit's models and analytics. **None of these
carry a fetal-sex label, and this project builds no model that predicts fetal
sex.** They support de-identification, anatomical-plane recognition, biometry,
and sex-ratio surveillance.

## Imaging — fetal-plane classification (`pctk.planes`)

| Dataset | What | Host | Notes |
|---------|------|------|-------|
| **FETAL_PLANES_DB** | 12,400 maternal-fetal US images, 1,792 patients, 6 anatomical planes (brain, abdomen, femur, thorax, maternal cervix, other) | [Zenodo 3904280](https://zenodo.org/record/3904280) · [Kaggle mirror](https://www.kaggle.com/datasets/fatemehsarhaddi/fetal-planes-db) | Labels are **anatomical planes, not sex**. Primary training set for `pctk.planes`. |
| HC18 | Fetal head-circumference segmentation (transventricular plane) | [grand-challenge HC18](https://hc18.grand-challenge.org/) | For biometry (Problem 4): HC → gestational age. |

Download with the CLI-adjacent helpers:

```python
from pctk.planes.download import download_kaggle, download_zenodo
download_kaggle("data/fetal_planes")      # needs ~/.kaggle/kaggle.json
download_zenodo("data/fetal_planes")      # ~2 GB direct
```

Then train:

```bash
pctk planes-train data/fetal_planes --out plane_model.joblib
pctk planes-eval  data/fetal_planes --model plane_model.joblib
pctk planes-predict some_frame.png  --model plane_model.joblib
```

## Tabular — fetal wellbeing (`pctk.health`)

| Dataset | What | Host |
|---------|------|------|
| Fetal Health Classification | 2,126 CTG records, 21 features, 3-class (normal/suspect/pathological) | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) |

```bash
pctk health-demo                         # synthetic end-to-end, no download
pctk health-train fetal_health.csv --out health_model.joblib
```

## Demographics — sex-ratio analytics (`pctk.srb`)

| Dataset | What | Host |
|---------|------|------|
| Census 2011 PCA (0–6 `M_06`/`F_06`) | District child counts by sex → `--source child06` | [pigshell mirror](https://github.com/pigshell/india-census-2011) · [Kaggle India Census](https://www.kaggle.com/datasets/danofer/india-census) |
| NFHS-5 district factsheets | Sex ratio at birth (last 5 yrs) → `--source nfhs` | [pratapvardhan/NFHS-5](https://github.com/pratapvardhan/NFHS-5) |
| District-wise Sex Ratio at Birth | Year/state/district SRB | [dataful.in](https://dataful.in/datasets/5920/) |

`python examples/fetch_data.py` pulls the NFHS-5 and Census PCA files
automatically (no account needed for those mirrors).

## Licensing / ethics

Respect each dataset's license and the consent under which medical images were
collected. Use imaging models for de-identification, plane recognition, and
biometry — never to infer sex. See the README's "What this is — and what it
deliberately is *not*" section.
