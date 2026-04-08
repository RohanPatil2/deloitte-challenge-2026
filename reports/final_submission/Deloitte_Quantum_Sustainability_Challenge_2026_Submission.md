# Deloitte Quantum Sustainability Challenge 2026

## Phase 1 Submission Draft: Tasks 1A, 1B, and 2

Prepared April 08, 2026

## Overview of the Individual or Team and Backgrounds

| Field | Value |
| --- | --- |
| Registered Team Name | Insert the exact registered team name before submission |
| Primary Contact | Rohan Patil |
| Submission Scope | Phase 1 entry covering Tasks 1A, 1B, and 2 in one PDF |
| College / University Affiliation | Verify and match the exact affiliation wording used during registration |

| Name | Background | E-mail | College / University Affiliation |
| --- | --- | --- | --- |
| Rohan Patil | Repository integration, reproducibility, document packaging, and delivery review | rohanpatil0212@gmail.com | Affiliation to verify before submission |
| Shreyas Khandale | Quantum machine learning, wildfire baselines, and comparative evaluation | skhandale@binghamton.edu | Binghamton University (to verify) |
| Aishwarya Das | Insurance analytics, feature engineering, experiment synthesis, and report editing | adas21@binghamton.edu | Binghamton University |

Before uploading the Phase 1 PDF, replace any placeholder language in the team name or affiliation fields with the exact registration values used on the Deloitte competition website.

## One-Page Summary / Abstract

This Phase 1 submission presents a hybrid wildfire-risk and insurance-premium modeling framework for Deloitte's Quantum Sustainability Challenge 2026. Task 1A is addressed through a wildfire occurrence prediction workflow at California ZIP-year granularity, Task 1B through explicit classical-versus-quantum evaluation, and Task 2 through a downstream insurance premium forecast that tests whether wildfire-risk signals improve financial modeling. The wildfire source file in the repository contains 125,476 rows and 30 columns, including 123,258 weather rows and 2,218 fire-event rows. These records are converted into 10,372 annual ZIP-year observations by aggregating weather, defining a binary fire-event target, and engineering lagged fire-history features. The insurance dataset contains 47,033 rows across 76 columns and is modeled using a strict 2018-2020 training window with 2021 held out for evaluation.

For wildfire prediction, we benchmark Logistic Regression, Random Forest, Gradient Boosting, and an RBF-kernel SVM against quantum-kernel SVMs and a 4-qubit variational quantum classifier. The quantum-kernel models use mutual-information feature selection, exact-statevector FidelityStatevectorKernel evaluation, ZZFeatureMap or PauliFeatureMap circuits, and balanced SVC classification. The VQC uses a ZZFeatureMap encoder, a RealAmplitudes ansatz, COBYLA optimization, and a class-balanced weighted cross-entropy objective. In the committed results, Logistic Regression is the strongest wildfire baseline with ROC-AUC 0.8647, while the best quantum-kernel model reaches 0.6222 and the VQC reaches 0.4775. The resource profile is also explicit: the checked-in quantum runs operate at 4-6 qubits, circuit depths between 27 and 49, and total simulator runtimes between 6.85 seconds and 267.13 seconds.

For insurance forecasting, we compare four experiments: no wildfire input, the provided dataset wildfire score, an imported classical wildfire score, and an imported quantum wildfire score. Gradient Boosting in Experiment A produces the best checked-in insurance result with R2 = 0.9794. The wildfire-augmented experiments remain close to that baseline, indicating that wildfire-risk features are directionally useful but not yet the dominant source of predictive lift when strong lagged insurance features are already present. No external predictive datasets beyond the challenge materials were introduced. Overall, the submission demonstrates a feasible and transparent prototype: classical models currently lead predictive accuracy, while the quantum workflows contribute innovation through reproducible hybrid benchmarking, resource-aware experimentation, and a clear path toward a future 2023 wildfire-risk forecasting run.

## Detailed Description of the Participant's Algorithm

<b>Challenge coverage.</b> This document addresses the three required work items in a single PDF. <b>Task 1A</b> is handled by a wildfire-risk classification pipeline at ZIP-year level. <b>Task 1B</b> is handled by comparative evaluation between classical baselines and quantum models, including performance metrics and resource evidence. <b>Task 2</b> is handled by a time-series-style insurance premium forecasting workflow that tests whether wildfire-risk scores improve downstream premium prediction.

<b>Data and additional data used.</b> The wildfire workstream uses the challenge-provided mixed file containing monthly weather observations and fire event rows: 125,476 rows, 30 columns, 123,258 weather rows, and 2,218 fire rows. The insurance workstream uses the challenge-provided ZIP-level insurance panel with 47,033 rows and 76 columns. No external predictive datasets were added. Supplemental non-predictive references were limited to the challenge feature-description material and the CAL FIRE coded-value notes referenced in the challenge guide. All remaining inputs are internally derived features such as lags, ratios, aggregated weather summaries, and imported wildfire-risk scores generated from Task 1 outputs.

<b>Concept.</b> The central concept is a hybrid risk-modeling stack. The first layer turns wildfire history and weather behavior into a per-ZIP wildfire-risk estimate. The second layer tests whether those risk signals can improve an insurance premium model. This design is useful for the challenge because it links climate risk estimation to a concrete financial application instead of treating the quantum model as an isolated benchmark.

<b>General composition.</b> The pipeline has four layers: data cleaning, feature construction, model training, and cross-model evaluation. Data cleaning handles ZIP normalization, row-type separation, negative-premium clipping, and boolean encoding. Feature construction creates annual weather summaries, wildfire-history lags, premium lags, trend features, and exposure-normalized insurance indicators. Model training then splits into classical wildfire baselines, quantum wildfire models, and insurance regression experiments. Finally, the evaluation layer consolidates performance tables, figures, and explicit quantum resource summaries.

<b>Task 1A wildfire algorithm.</b> Wildfire modeling begins by aggregating monthly weather rows into annual ZIP-year features: mean maximum temperature, mean minimum temperature, total precipitation, and temperature range. Fire-event rows are reduced to a binary target indicating whether at least one wildfire occurred in a ZIP-year. The algorithm then constructs lag-1 fire count, lag-2 fire count, cumulative fire count, and an ever-had-fire flag. Classical baselines include Logistic Regression, Random Forest, Gradient Boosting, and an RBF-kernel SVM. Quantum runs use mutual-information feature selection, exact-statevector quantum kernels built from ZZFeatureMap or PauliFeatureMap circuits, and a balanced SVC on the resulting kernel matrix. A 4-qubit VQC with ZZFeatureMap plus RealAmplitudes serves as the variational quantum alternative.

<b>Task 1B evaluation.</b> The wildfire models are evaluated with a strict temporal split rather than random shuffling. In the checked-in artifact package, training uses 2018-2020 ZIP-year rows (7,779) and validation uses 2021 (2,593). Primary metrics are ROC-AUC and PR-AUC, supported by F1, precision, and recall because the wildfire-event rate is low. Task 1B also includes a resource perspective: number of qubits, circuit depth, runtime, backend choice, and trainable parameter count where relevant.

<b>Task 2 insurance algorithm.</b> The insurance model uses the challenge ZIP-level panel with 33,689 training rows and 13,344 test rows. Preprocessing removes leakage-prone post-event fire fields and unusable weather columns, clips negative premiums, and encodes boolean columns numerically. Feature engineering then adds premium lag-1, premium lag-2, year-over-year premium change, premium percent change, lagged loss ratio, and premium-per-exposure features. Four experiments are run: Experiment A with no added wildfire-risk score, Experiment B with the provided wildfire-risk score in the dataset, Experiment C with an imported classical wildfire-risk score, and Experiment D with an imported quantum wildfire-risk score. Each experiment benchmarks a naive lag-1 baseline, Ridge, Random Forest, and Gradient Boosting regression.

<b>Underlying assumptions.</b> The solution assumes that annual ZIP-level weather aggregation preserves enough signal for wildfire-risk classification; that prior fire activity contains meaningful predictive value; and that imported wildfire-risk scores can be treated as exogenous features inside the insurance model. It also assumes that temporal splits are more appropriate than random splits for both tasks, because the submission is intended to emulate forward prediction rather than retrospective curve fit.

<b>Scope note for challenge alignment.</b> The repository-backed evidence in this submission validates the wildfire pipeline on the latest weather horizon available in the checked-in raw file, which ends at 2021. The algorithm design is compatible with the challenge's requested 2018-2022 to 2023 framing, but a final 2023 production run would require the corresponding sponsor-aligned weather horizon or an updated wildfire input release.

## Description of Results

<b>Task 1A and Task 1B wildfire results.</b> Table 1 and Figure 1 summarize the wildfire experiments. Logistic Regression is the strongest checked-in classical model with ROC-AUC = 0.8647. The best quantum-kernel run is Quantum Kernel SVM (6q) with ROC-AUC = 0.6222, while the VQC reaches 0.4775. The Pauli 4-qubit kernel performs better than the ZZ 4-qubit kernel, which indicates that feature-map choice matters even in a compact qubit regime.

<b>Evaluation interpretation.</b> The wildfire comparison should be read with its tractability constraints in mind. The strongest classical baseline uses the full 7,779-example training set, while the quantum models are capped at 800 examples because exact kernel construction scales quadratically. Even so, the Task 1B evidence is still useful: it shows that the quantum workflows are operational, measurable, and sensitive to qubit count, circuit family, and runtime budget.

<b>Task 2 results.</b> Table 2 and Figure 2 summarize the insurance experiments. Across all four experiments, Gradient Boosting is the strongest model family. The best overall checked-in result is Experiment A, which excludes external wildfire-risk input, with R2 = 0.9794. Experiments B, C, and D remain close to that baseline, suggesting that wildfire-risk scores are plausible auxiliary signals but do not yet dominate the premium forecast once strong lagged insurance features are available.

<b>Resource evidence.</b> Table 3 should be read as part of the Task 1B result set. The committed quantum runs use a statevector simulator, 4-6 qubits, circuit depths between 27 and 49, and total runtimes between 6.85 seconds and 267.13 seconds. This resource framing strengthens the feasibility argument because it documents not just accuracy, but also cost, depth, and optimization burden.

<b>Innovation, feasibility, and community impact.</b> The submission's innovation lies in connecting wildfire-risk prediction to an insurance use case, while still providing transparent classical baselines. Its feasibility is supported by reproducible scripts, tractable simulator runtimes for the kernel models, and direct artifact links. Its quantum community impact comes from providing an open benchmark narrative for where quantum methods are currently promising, where they are still limited, and how hybrid workflows can be evaluated responsibly.

### Table 1. Wildfire model comparison

| Model | Type | Qubits | ROC-AUC | PR-AUC | F1 | Recall | Train Samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Classical | - | 0.8647 | 0.4481 | 0.3546 | 0.8341 | 7779 |
| Random Forest | Classical | - | 0.6406 | 0.2140 | 0.1765 | 0.7130 | 7779 |
| Gradient Boosting | Classical | - | 0.8471 | 0.4690 | 0.2494 | 0.9193 | 7779 |
| Classical SVM (RBF) | Classical | - | 0.8372 | 0.2583 | 0.3363 | 0.8341 | 7779 |
| Quantum Kernel SVM (4q) | Quantum Kernel | 4 | 0.4701 | 0.0844 | 0.1316 | 0.3498 | 800 |
| Quantum Kernel SVM (6q) | Quantum Kernel | 6 | 0.6222 | 0.1105 | 0.0552 | 0.0673 | 800 |
| Quantum Kernel SVM (Pauli-4q) | Quantum Kernel | 4 | 0.5436 | 0.1182 | 0.1690 | 0.4798 | 800 |
| VQC (4q) | Quantum VQC | 4 | 0.4775 | 0.1418 | 0.1356 | 0.5471 | 800 |

Table note: classical wildfire rows use the full 7,779-example training set, while the checked-in quantum rows use 800 training examples because of exact-kernel tractability.

![Wildfire comparison figure](reports/figures/quantum_vs_classical.png)

Figure 1 note: the chart visually reinforces that the current prototype's strongest classical baseline outperforms the checked-in quantum runs on ROC-AUC, while the quantum artifacts still provide meaningful resource and architecture comparisons.

### Table 2. Insurance Experiment A-D summary

| Experiment | Wildfire Input | Best Model | MAE | RMSE | R2 | MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| A | None | Gradient Boosting | 79,010.32 | 245,963.11 | 0.9794 | 145,505.94 |
| B | Provided score in dataset | Gradient Boosting | 79,590.69 | 249,453.45 | 0.9788 | 148,944.22 |
| C | Imported classical wildfire score | Gradient Boosting | 78,974.23 | 249,769.66 | 0.9788 | 141,841.39 |
| D | Imported quantum wildfire score | Gradient Boosting | 79,056.11 | 253,132.12 | 0.9782 | 143,532.48 |

Table note: each row reports the best-performing model family inside that experiment, not every model that was tested.

![Insurance comparison figure](insurance-model/results/figures/experiment_comparison.png)

Figure 2 note: the insurance figure shows that all wildfire-augmented experiments remain close to the no-wildfire baseline, with Gradient Boosting dominating the checked-in runs.

### Table 3. Quantum resource summary

| Model | Qubits | Depth | Backend | Train Samples | Runtime (s) | Best ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Quantum Kernel SVM (4q) | 4 | 31 | statevector_simulator | 800 | 6.8515 | 0.4701 |
| Quantum Kernel SVM (6q) | 6 | 49 | statevector_simulator | 800 | 8.3727 | 0.6222 |
| Quantum Kernel SVM (Pauli-4q) | 4 | 31 | statevector_simulator | 800 | 7.355 | 0.5436 |
| VQC (4q) | 4 | 27 | statevector_simulator | 800 | 267.13 | 0.4775 |

Table note: Table 3 is part of the Task 1B evaluation evidence because it captures runtime, qubit count, backend, and circuit-depth implications.

## Description of the Envisioned Algorithm

<b>Envisioned next-stage algorithm.</b> The natural next step is a fully challenge-aligned 2023 wildfire-risk forecasting workflow trained on 2018-2022 history. That production version should preserve the strongest aspects of the current prototype: row-type separation, annual weather aggregation, lagged fire-history features, strict temporal validation, and comparative benchmarking against strong classical baselines.

<b>Expected benefits.</b> A mature version of this solution could improve decision-making in two ways. First, better wildfire-risk estimates can support more targeted geographic risk monitoring. Second, calibrated wildfire-risk scores can be fed into downstream insurance models, scenario analysis, or portfolio stress testing. The quantum component is most compelling in a hybrid architecture, where it complements rather than replaces the strongest classical screen.

<b>Requirements for the envisioned solution.</b> A stronger follow-on version would need the sponsor-aligned 2022-2023 wildfire-weather horizon, calibrated probability output, hardware-aware or shot-based quantum experiments in addition to statevector simulations, and a clear strategy for scaling beyond the 800-sample quantum training cap. It would also benefit from shallower circuit exploration, probability calibration, and explicit uncertainty communication so that the output is more useful to insurance and climate-risk stakeholders.

<b>Why this matters.</b> Even in its current form, the repository provides a credible prototype foundation for the quantum community: reproducible comparison tables, explicit resource reporting, and an application that links QML to a real sustainability and finance problem. The envisioned algorithm extends that foundation into a more operational hybrid forecasting system with clearer practical value.

## Clickable Repo Links

- [Repository root](https://github.com/RohanPatil2/deloitte-challenge-2026) - Main codebase and checked-in artifacts.
- [Wildfire quantum kernel pipeline](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/src/models/quantum/quantum_kernel_pipeline.py) - Primary Task 1 quantum-kernel implementation.
- [Wildfire VQC pipeline](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/src/models/quantum/vqc_pipeline.py) - Variational quantum classifier implementation.
- [Wildfire comparison table](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/reports/tables/model_comparison.csv) - Canonical cross-model comparison used in this report.
- [Quantum resource table](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/reports/tables/quantum_resources.csv) - Circuit depth, runtime, backend, and sample counts.
- [Insurance module](https://github.com/RohanPatil2/deloitte-challenge-2026/tree/main/insurance-model) - Task 2 source code, notebooks, and results.
- [Insurance combined metrics](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/insurance-model/results/metrics/all_experiments_combined.csv) - Committed Experiment A-D result table.

Note: this report is a repository-backed submission draft built from the current checked-in artifacts on the `main` branch.
