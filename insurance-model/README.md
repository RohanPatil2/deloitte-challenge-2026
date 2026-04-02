# Insurance Premium Forecasting — Task 2
## Deloitte Quantum Sustainability Challenge 2026

**Owner:** Person 3 — Insurance + Submission Lead  
**Goal:** Predict 2021 insurance premiums by California ZIP code using 2018–2020 training data.

---

## Experiments

| Experiment | Wildfire Risk Input | Status |
|---|---|---|
| A | None | ⬜ |
| B | Provided fire risk score (dataset) | ⬜ |
| C | Classical model risk (Logistic Regression) | ⬜ |
| D | Quantum model risk (QKernel ZZ-6q) | ⬜ |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Folder Structure

```
insurance-model/
├── data/
│   ├── raw/          ← original dataset
│   ├── processed/    ← cleaned files
│   └── external/     ← wildfire risk scores from Shreyas
├── notebooks/        ← one notebook per experiment
├── src/              ← reusable code (config, preprocessing, models)
├── results/
│   ├── metrics/      ← CSV result tables
│   ├── predictions/  ← model predictions per ZIP
│   └── figures/      ← charts for report
└── reports/          ← final PDF
```

---

## Run Order

```
01_data_audit.ipynb
02_feature_engineering.ipynb
03_experiment_A.ipynb
04_experiment_B.ipynb
05_experiment_C.ipynb
06_experiment_D.ipynb
07_results_comparison.ipynb
```
