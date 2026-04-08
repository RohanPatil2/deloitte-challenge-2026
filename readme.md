# Deloitte Quantum Sustainability Challenge 2026

This repository contains the team's working code and generated artifacts for the Deloitte wildfire prediction and insurance premium modeling challenge.

## Repository Scope

The repo currently contains two linked workstreams:

- `src/`: wildfire event prediction experiments, including classical baselines, quantum-kernel models, VQC experiments, and comparison-table builders.
- `insurance-model/`: Task 2 insurance premium forecasting for 2021, with experiments that optionally consume wildfire risk scores exported from the wildfire pipeline.

## Important Scope Note

The official challenge brief asks for wildfire risk prediction for 2023 using historical data through 2022. The checked-in wildfire experiments in this repo are currently structured as a validation study using 2018-2020 training data and 2021 validation data. The insurance pipeline is aligned to Task 2 and predicts 2021 premiums from 2018-2020 history.

That distinction should be stated clearly in any final report or submission deck. The current repository demonstrates methodology and comparative experiments; it is not yet a full end-to-end 2023 forecasting submission.

## Current Artifacts

Wildfire outputs:

- `reports/tables/model_comparison.csv`
- `reports/tables/quantum_resources.csv`
- `reports/tables/wildfire_risk_predictions_2021.csv`
- `reports/figures/quantum_vs_classical.png`

Insurance outputs:

- `insurance-model/results/metrics/all_experiments_combined.csv`
- `insurance-model/results/figures/experiment_comparison.png`

Submission package:

- `reports/final_submission/Deloitte_Quantum_Sustainability_Challenge_2026_Submission.md`
- `reports/final_submission/Deloitte_Quantum_Sustainability_Challenge_2026_Submission.pdf`
- `reports/final_submission/build_submission_package.py`
- `reports/final_submission/check_submission.py`

## Checked-In Results Snapshot

Based on the committed CSV artifacts:

- Wildfire classification: best classical ROC-AUC is `0.8647` from Logistic Regression on `7,779` training samples.
- Wildfire classification: best quantum-kernel ROC-AUC is `0.6222` from the `6`-qubit ZZ feature-map run on `800` training samples.
- Wildfire classification: VQC ROC-AUC is `0.4775` on `800` training samples.
- Insurance premium modeling: best checked-in `R2` is `0.9794` from Gradient Boosting in Experiment A.

## Layout

```text
.
├── data/                         # wildfire raw data
├── reports/                      # wildfire figures, tables, and submission package
├── results/                      # wildfire JSON logs and metrics
├── src/
│   ├── evaluation/               # comparison/resource table builders, baselines
│   ├── models/quantum/           # quantum kernel, VQC, risk generation scripts
│   └── visualization/            # wildfire plots
└── insurance-model/
    ├── data/                     # insurance raw/processed/external data
    ├── notebooks/                # experiment notebooks
    ├── results/                  # insurance metrics and figures
    └── src/                      # config, preprocessing, features, models, utils
```

## Reproducing The Current Experiments

Wildfire environment:

```bash
pip install -r requirements.txt
```

Insurance environment:

```bash
cd insurance-model
pip install -r requirements.txt
```

The repo includes generated result files already, so documentation and report work can proceed without rerunning everything.

## Recommended Next Steps

- Use `reports/REPORT_OUTLINE.md` as the starting structure for the written submission.
- Build the final package with `./.venv-report/bin/python reports/final_submission/build_submission_package.py`.
- Verify the package with `./.venv-report/bin/python reports/final_submission/check_submission.py`.
- Keep the report explicit about which results are validation experiments versus final challenge deliverables.
- If final submission requires 2023 wildfire predictions, rerun Task 1 with the challenge-aligned train window before packaging the final deck.
