# Insurance Premium Forecasting - Task 2
## Deloitte Quantum Sustainability Challenge 2026

Goal: predict 2021 insurance premiums by California ZIP code using 2018-2020 historical data, then compare whether adding wildfire-risk features improves downstream premium modeling.

## Experiments

| Experiment | Wildfire Risk Input | Current Status |
|---|---|---|
| A | None | Complete |
| B | Provided fire risk score from dataset | Complete |
| C | Classical wildfire risk score | Complete |
| D | Quantum wildfire risk score | Complete |

Combined metrics are stored in `results/metrics/all_experiments_combined.csv`.

## Current Result Snapshot

Based on the checked-in metrics:

- Best `R2` is `0.9794` from Gradient Boosting in Experiment A.
- Experiments B, C, and D remain close to that baseline, which means wildfire-risk features did not materially change the best premium result in the current committed runs.

## Setup

```bash
pip install -r requirements.txt
```

## Folder Structure

```text
insurance-model/
├── data/
│   ├── raw/          # original dataset
│   ├── processed/    # cleaned files
│   └── external/     # imported wildfire risk scores
├── notebooks/        # one notebook per experiment
├── src/              # config, preprocessing, features, models, utils
└── results/
    ├── metrics/      # CSV result tables
    └── figures/      # charts for report/deck
```

## Notebook Run Order

```text
01_data_audit.ipynb
02_feature_engineering.ipynb
03_experiment_A.ipynb
04_experiment_B.ipynb
05_experiment_C.ipynb
06_experiment_D.ipynb
07_results_comparison.ipynb
```

## Key Files

- `src/config.py`: central experiment settings, paths, and feature lists
- `src/preprocessing.py`: data loading, cleaning, optional wildfire-risk merge
- `src/models.py`: model training and evaluation
- `results/metrics/all_experiments_combined.csv`: consolidated experiment comparison

## Reporting Notes

- Keep Task 2 framed as a 2021 premium-forecasting exercise using historical data through 2020.
- When referencing Experiments C and D, state clearly that the wildfire-risk feature is imported from Task 1 outputs rather than learned inside the insurance pipeline itself.
