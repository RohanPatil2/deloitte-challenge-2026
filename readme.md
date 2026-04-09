# Deloitte Quantum Sustainability Challenge 2026

[![Python](https://img.shields.io/badge/Python-Project-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum_ML-6929C4.svg)](https://qiskit.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classical_Models-F7931E.svg)](https://scikit-learn.org/)
[![Submission](https://img.shields.io/badge/Status-Phase_1_Submission-2E8B57.svg)](#submission-assets)

> A hybrid climate-finance modeling workflow that links wildfire-risk prediction to downstream insurance premium forecasting.

This repository is our official working codebase for the Deloitte Quantum Sustainability Challenge 2026. It brings together two connected problem settings:

1. `Task 1A / 1B`: predict wildfire occurrence risk at the California ZIP-year level and compare classical and quantum machine learning approaches.
2. `Task 2`: test whether those wildfire-risk signals improve downstream insurance premium forecasting.

The result is not just a collection of isolated models. It is a layered pipeline in which climate-risk estimation feeds a financial forecasting workflow.

## At A Glance

- Wildfire source data processed from `125,476` raw rows into `10,372` annual ZIP-year observations.
- Insurance modeling built from `47,033` ZIP-level records spanning `2018` to `2021`.
- Best checked-in wildfire ROC-AUC: `0.8647` from Logistic Regression.
- Best checked-in quantum-kernel wildfire ROC-AUC: `0.6222` from the `6`-qubit ZZ feature-map run.
- Best checked-in insurance result: `R2 = 0.9794` from Gradient Boosting in Experiment A.
- Phase 1 submission package included in [`reports/final_submission/`](reports/final_submission).

## Why This Repository Exists

Wildfire risk is not just an environmental problem. It is also a financial one. Insurance pricing, capital planning, and portfolio stress testing all become harder when localized climate risk shifts faster than historical averages can explain.

Our approach treats that as a two-layer modeling problem:

### Layer 1: Climate Risk

We transform monthly weather behavior and historical fire activity into a ZIP-level wildfire-risk estimate using both classical baselines and quantum machine learning workflows.

### Layer 2: Financial Risk

We inject those wildfire-risk signals into an insurance premium forecasting pipeline and test whether they add predictive value beyond strong historical insurance features.

This is the key design choice of the project: quantum machine learning is not presented as a standalone novelty experiment. It is evaluated inside a practical pipeline that connects climate signal extraction to an insurance use case.

## Repository Scope

The codebase is organized into two linked workstreams.

| Workstream | Purpose | Location |
| --- | --- | --- |
| Wildfire Modeling | Classical baselines, quantum kernel SVMs, VQC experiments, comparison tables, and resource tracking | [`src/`](src) |
| Insurance Modeling | Feature engineering, experiment A-D evaluation, metrics, and figures for 2021 premium forecasting | [`insurance-model/`](insurance-model) |

## Hybrid Architecture

### Part 1: Wildfire Risk Prediction

The wildfire workflow:

- separates monthly weather rows from fire-event rows
- aggregates weather to annual ZIP-year features
- defines a binary wildfire-event target
- engineers historical fire lags and cumulative fire features
- benchmarks classical and quantum models on a strict temporal split

Model families included:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Classical SVM with RBF kernel
- Quantum Kernel SVM with ZZFeatureMap
- Quantum Kernel SVM with PauliFeatureMap
- Variational Quantum Classifier using `ZZFeatureMap + RealAmplitudes`

### Part 2: Insurance Premium Forecasting

The insurance workflow:

- cleans and standardizes the ZIP-level insurance panel
- clips negative premium rows
- constructs premium lag, trend, loss-ratio, and exposure-normalized features
- evaluates four experiment settings:

| Experiment | Wildfire Input |
| --- | --- |
| A | None |
| B | Provided wildfire score already present in dataset |
| C | Imported classical wildfire score from Task 1 |
| D | Imported quantum wildfire score from Task 1 |

## Evidence Snapshot

The following numbers come directly from the checked-in repository artifacts.

### Wildfire Model Comparison

Classical wildfire rows use the full `7,779` training set. The checked-in quantum runs use `800` training samples to keep exact-kernel simulation tractable.

| Model | Type | Qubits | ROC-AUC | F1 | Recall |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | Classical | - | **0.8647** | 0.3546 | 0.8341 |
| Gradient Boosting | Classical | - | 0.8471 | 0.2494 | 0.9193 |
| Classical SVM (RBF) | Classical | - | 0.8372 | 0.3363 | 0.8341 |
| Quantum Kernel SVM (6q) | Quantum Kernel | 6 | **0.6222** | 0.0552 | 0.0673 |
| Quantum Kernel SVM (Pauli-4q) | Quantum Kernel | 4 | 0.5436 | 0.1690 | 0.4798 |
| VQC (4q) | Quantum VQC | 4 | 0.4775 | 0.1356 | 0.5471 |

Reference artifacts:

- [`reports/tables/model_comparison.csv`](reports/tables/model_comparison.csv)
- [`reports/figures/quantum_vs_classical.png`](reports/figures/quantum_vs_classical.png)

### Quantum Resource Tracking

We explicitly track runtime, qubit count, and circuit depth because resource transparency matters as much as model accuracy in early-stage quantum workflows.

| Model | Qubits | Circuit Depth | Runtime (s) |
| --- | --- | --- | --- |
| Quantum Kernel SVM (4q ZZ) | 4 | 31 | 6.8515 |
| Quantum Kernel SVM (Pauli-4q) | 4 | 31 | 7.3550 |
| Quantum Kernel SVM (6q ZZ) | 6 | 49 | 8.3727 |
| VQC (4q) | 4 | 27 | 267.13 |

Reference artifact:

- [`reports/tables/quantum_resources.csv`](reports/tables/quantum_resources.csv)

### Insurance Experiment Summary

Across all checked-in insurance experiments, Gradient Boosting is the strongest model family.

| Experiment | Wildfire Input | Best Model | MAE | R2 |
| --- | --- | --- | --- | --- |
| A | None | Gradient Boosting | 79,010.32 | **0.9794** |
| B | Provided dataset score | Gradient Boosting | 79,590.69 | 0.9788 |
| C | Imported classical wildfire score | Gradient Boosting | 78,974.23 | 0.9788 |
| D | Imported quantum wildfire score | Gradient Boosting | 79,056.11 | 0.9782 |

Reference artifacts:

- [`insurance-model/results/metrics/all_experiments_combined.csv`](insurance-model/results/metrics/all_experiments_combined.csv)
- [`insurance-model/results/figures/experiment_comparison.png`](insurance-model/results/figures/experiment_comparison.png)

## Scope Note

This repository contains a strong prototype and submission-ready documentation, but the checked-in wildfire experiments are still validation-stage runs.

- The current wildfire raw file contains weather coverage through `2021`.
- The checked-in Task 1 experiments therefore train on `2018–2020` and validate on `2021`.
- The official challenge framing asks for wildfire prediction in `2023` using historical data through `2022`.

That difference is important. The repository demonstrates the modeling approach, comparative evaluation, and submission narrative clearly, but a fully challenge-aligned `2023` production run would require the corresponding sponsor-aligned weather horizon or an updated input release.

## Repository Map

```text
deloitte-challenge-2026/
├── data/
│   ├── raw/                               # wildfire raw inputs
│   └── processed/
├── src/
│   ├── evaluation/                        # comparison tables and baselines
│   ├── models/
│   │   └── quantum/
│   │       ├── quantum_kernel_pipeline.py
│   │       ├── vqc_pipeline.py
│   │       └── generate_risk_predictions.py
│   └── visualization/
├── results/
│   ├── quantum_metrics/                   # classical baseline JSON metrics
│   └── quantum_runs/                      # quantum run logs
├── reports/
│   ├── figures/                           # wildfire figures
│   ├── tables/                            # canonical CSV outputs
│   └── final_submission/                  # PDF, Markdown, LaTeX, builder scripts
├── insurance-model/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── external/                      # imported wildfire-risk scores
│   ├── notebooks/                         # experiment notebooks
│   ├── src/                               # insurance feature engineering and models
│   └── results/
│       ├── figures/
│       └── metrics/
└── README.md
```

## Submission Assets

The final submission package lives in [`reports/final_submission/`](reports/final_submission).

Included files:

- [`Deloitte_Quantum_Sustainability_Challenge_2026_Submission.pdf`](reports/final_submission/Deloitte_Quantum_Sustainability_Challenge_2026_Submission.pdf)
- [`Deloitte_Quantum_Sustainability_Challenge_2026_Submission.md`](reports/final_submission/Deloitte_Quantum_Sustainability_Challenge_2026_Submission.md)
- [`Deloitte_Quantum_Sustainability_Challenge_2026_Submission.tex`](reports/final_submission/Deloitte_Quantum_Sustainability_Challenge_2026_Submission.tex)
- [`build_submission_package.py`](reports/final_submission/build_submission_package.py)
- [`check_submission.py`](reports/final_submission/check_submission.py)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RohanPatil2/deloitte-challenge-2026.git
cd deloitte-challenge-2026
```

### 2. Install dependencies

For the wildfire workstream:

```bash
pip install -r requirements.txt
```

For the insurance workstream:

```bash
cd insurance-model
pip install -r requirements.txt
cd ..
```

### 3. Run the wildfire workflows

```bash
python src/evaluation/classical_baselines.py
python src/models/quantum/quantum_kernel_pipeline.py
python src/models/quantum/vqc_pipeline.py
python src/evaluation/build_comparison_table.py
python src/evaluation/build_resource_table.py
```

### 4. Run the insurance workflow

The insurance side is notebook-driven in the current repository snapshot. The intended order is:

```text
01_data_audit.ipynb
02_feature_engineering.ipynb
03_experiment_A.ipynb
04_experiment_B.ipynb
05_experiment_C.ipynb
06_experiment_D.ipynb
07_results_comparison.ipynb
```

## Reproducibility Notes

- The repository already includes generated result artifacts, so reviewers can inspect outputs without rerunning every pipeline.
- The final submission package can be regenerated with:

```bash
./.venv-report/bin/python reports/final_submission/build_submission_package.py
./.venv-report/bin/python reports/final_submission/check_submission.py
```

- The report checker validates abstract length, required sections, and the presence of clickable GitHub links.

## What This Project Shows

This repository makes three things clear:

1. Strong classical baselines still lead predictive performance on the checked-in wildfire validation artifacts.
2. The quantum workflows are operational, measurable, and informative from a resource-tracking perspective.
3. Wildfire-risk scores can be injected into an insurance forecasting pipeline without breaking downstream performance, even if they do not yet outperform the strongest purely financial baseline.

That is a useful result. It gives the project a realistic, defensible story: classical models currently win on raw accuracy, while quantum methods contribute innovation through hybrid design, transparent benchmarking, and a credible route for future scaling.

## Team

| Name | Focus Area |
| --- | --- |
| Rohan Patil | Repository integration, reproducibility, packaging, delivery review |
| Shreyas Khandale | Quantum machine learning, wildfire baselines, comparative evaluation |
| Aishwarya Das | Insurance analytics, feature engineering, report editing |

Affiliation: Binghamton University

---

Built for the Deloitte Quantum Sustainability Challenge 2026 using competition-provided data and repository-backed experiment artifacts.
