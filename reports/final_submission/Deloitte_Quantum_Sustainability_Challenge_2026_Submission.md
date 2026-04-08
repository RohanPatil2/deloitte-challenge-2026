# Deloitte Quantum Sustainability Challenge 2026

## Final Submission Draft: Wildfire Prediction and Insurance Premium Modeling

Prepared April 08, 2026

## Team Overview and Contact Details

| Name | Role | Contact |
| --- | --- | --- |
| Rohan Patil | Repository integration, reproducibility, and final packaging | rohanpatil0212@gmail.com |
| Shreyas Khandale | Wildfire modeling, quantum experiments, and classical baselines | Verify preferred submission email before sending |
| Aishwarya Das | Insurance modeling, experiment analysis, and report assembly | adas21@binghamton.edu |

Primary submission contact can be listed as Rohan Patil. Names and emails above were inferred from repository metadata and should be verified once before external submission.

## Abstract

This submission presents a two-stage modeling workflow for Deloitte's Quantum Sustainability Challenge 2026. The first stage predicts wildfire occurrence risk at the California ZIP-year level, and the second stage studies whether wildfire-risk signals improve a downstream insurance premium forecast. In the checked-in wildfire data, the raw file contains 125,476 rows and 30 columns, combining 123,258 weather rows with 2,218 fire-event rows. We aggregate monthly weather observations into annual ZIP summaries, build a binary fire-event label, and engineer lagged fire-history features that capture recent activity, cumulative history, and a prior-fire indicator. We then compare four classical baselines against two quantum learning approaches on a strict temporal split using 2018-2020 training data and 2021 validation data. The classical baseline suite includes Logistic Regression, Random Forest, Gradient Boosting, and an RBF-kernel SVM. The quantum suite includes exact-statevector quantum kernel SVMs built from ZZFeatureMap and PauliFeatureMap circuits, plus a 4-qubit variational quantum classifier using COBYLA and a custom class-balanced loss. Because kernel evaluation scales quadratically, the quantum runs are capped at 800 stratified training examples, while the best classical baseline uses the full 7779 training examples. In the committed artifacts, Logistic Regression achieves the strongest wildfire ROC-AUC at 0.8647, while the best quantum kernel model reaches 0.6222 and the VQC reaches 0.4775. These results do not yet demonstrate a quantum accuracy advantage, but they do provide a grounded benchmark for circuit depth, simulator runtime, feature compression, and hybrid-integration tradeoffs. Resource analysis is therefore part of the contribution: the kernel models remain comparatively fast on the simulator, while the variational model is meaningfully slower even at four qubits, clarifying which quantum path is more practical under current constraints. For Task 2, we forecast 2021 insurance premiums from 2018-2020 history using lagged premium signals, loss-based features, exposure measures, fire-risk bands, product-category flags, and census variables. We compare four experiments: a no-wildfire baseline, a provided wildfire score, a classical wildfire score, and a quantum wildfire score. In the checked-in results, Gradient Boosting in A - No Wildfire Risk delivers the best overall insurance fit with R2 = 0.9794. The wildfire-augmented experiments remain close to that baseline, which suggests that imported wildfire risk is directionally useful but not yet the dominant driver of premium accuracy when strong lagged insurance features are already present. Overall, the repository currently supports a scientifically useful prototype: classical models lead predictive performance, while the quantum workflows quantify how a constrained hybrid pipeline could evolve toward the full 2023 wildfire-forecasting objective described in the challenge.

## Data

The wildfire workstream uses a mixed raw dataset that combines monthly weather observations with historical fire-event rows. In the checked-in snapshot, the raw file contains 125,476 rows across 30 columns. The file includes 123,258 weather rows and 2,218 fire rows. The modeling pipeline normalizes ZIP codes, aggregates monthly weather into annual ZIP summaries, marks whether any fire occurred in a given ZIP-year, and engineers fire-history features such as lag-1 count, lag-2 count, cumulative fire count, and an ever-had-fire indicator. After filtering to the current evaluation window, the working wildfire dataset has 10,372 ZIP-year rows covering 2,593 ZIP codes, with 875 positive wildfire events.

The current wildfire experiments in this repository are validation-stage studies rather than the final challenge horizon. Training uses 2018-2020 ZIP-year observations (7,779 rows, fire rate 8.38%), and evaluation uses 2021 (2,593 rows, fire rate 8.60%). This distinction should be stated clearly in the final submission because the challenge brief ultimately asks for wildfire-risk prediction in 2023 using history through 2022.

The insurance workstream uses a separate ZIP-level panel containing 47,033 rows and 76 columns. The checked-in file covers 2018-2021 and spans 2,251 unique ZIP codes. The model uses a strict temporal split with 2018-2020 as training history and 2021 as the holdout year (33,689 training rows and 13,344 test rows). Data cleaning includes clipping 7 negative premium rows to zero and careful handling of wide insurance column names that contain embedded double spaces (16 such columns in the checked-in file).

The challenge briefing also calls out data-quality issues such as malformed date fields, categorical-code quirks, and naming inconsistencies. Those notes informed the cleaning strategy even where the committed CSV snapshot is already comparatively clean. In particular, the project treats row-type separation, column-name stability, and temporal partitioning as first-order requirements for both model validity and reproducibility.

## Methodology

The wildfire modeling pipeline shares a common preprocessing backbone across classical and quantum runs. Monthly weather rows are aggregated to annual ZIP-year features using mean maximum temperature, mean minimum temperature, total precipitation, and derived temperature range. Fire rows are collapsed into a binary target that marks whether a wildfire event occurred in the same ZIP-year. Historical fire activity is then encoded via lagged counts, an exclusive cumulative fire count, and a prior-fire flag. This produces a compact feature set designed to balance predictive signal with quantum tractability.

The classical wildfire benchmarks consist of Logistic Regression, Random Forest, Gradient Boosting, and an RBF-kernel SVM. All models are evaluated on the same 2021 holdout period, and class imbalance is handled through balanced class weights or equivalent sample weighting. Logistic Regression and the classical SVM use standardized features, while the tree-based models operate directly on the engineered inputs. These baselines establish an accuracy reference before any quantum modeling claims are made.

The quantum-kernel workflow compresses the candidate feature set through mutual-information-based selection, then maps the resulting 4- or 6-feature vectors into parameterized quantum feature maps. The committed experiments use exact-statevector FidelityStatevectorKernel evaluation, balanced SVC classification on a precomputed kernel matrix, and a stratified cap of 800 training samples to keep O(n^2) kernel construction tractable. Two ZZ feature-map runs (4 qubits and 6 qubits) are paired with a 4-qubit Pauli feature-map ablation to test whether circuit design changes the resulting similarity structure.

The VQC workflow further compresses the wildfire task into a 4-qubit variational model composed of a ZZFeatureMap encoder and a RealAmplitudes ansatz. Training uses the COBYLA optimizer for 100 iterations and a custom weighted cross-entropy loss so the minority wildfire class receives more attention during optimization. This setup prioritizes proof-of-concept comparability over hardware realism and therefore remains a simulator-based prototype.

The insurance methodology is intentionally modular. The preprocessing layer loads the raw ZIP-level insurance panel, removes leakage-prone post-event fire columns and unusable weather columns, clips negative premiums, and encodes boolean fields numerically. Feature engineering then builds premium lag features, year-over-year premium trend features, lagged loss ratios, and premium-per-exposure measures. The four insurance experiments differ only in which wildfire-risk input is appended: no additional wildfire score (Experiment A), the provided dataset wildfire-risk score (Experiment B), an imported classical wildfire-risk score (Experiment C), or an imported quantum wildfire-risk score (Experiment D).

Task 2 modeling compares a naive lag-1 baseline against Ridge regression, Random Forest regression, and Gradient Boosting regression. Ridge is trained on standardized inputs, while the tree-based models operate on the raw feature matrix. All experiments respect the same temporal split: 2018-2020 for training and 2021 for testing. This design makes it possible to ask a narrow question cleanly: do wildfire-risk signals materially improve premium prediction once strong lagged insurance features are already available?

## Results

Wildfire classification results are summarized in Table 1. In the current repository snapshot, Logistic Regression is the strongest classical model with ROC-AUC = 0.8647. The best quantum-kernel result is Quantum Kernel SVM (6q) with ROC-AUC = 0.6222, and the VQC reaches ROC-AUC = 0.4775. The Pauli 4-qubit kernel ablation performs better than the ZZ 4-qubit kernel, which suggests that feature-map choice matters even under a small-qubit regime.

These wildfire results should be interpreted carefully. The strongest classical baseline is trained on the full 7,779-example training set, while the quantum models are capped at 800 examples for tractability on a classical simulator. The comparison is still useful for benchmarking current pipeline behavior, but it should not be overstated as a like-for-like statement about quantum versus classical learning capacity. What the experiments do show is that the current quantum workflows are operational, measurable, and meaningfully sensitive to qubit count, circuit family, and runtime budget.

Insurance results are summarized in Table 2 and Figure 2. Across all four experiments, Gradient Boosting is the best-performing model family in the checked-in metrics. The strongest overall result comes from Experiment A (no external wildfire-risk feature) with R2 = 0.9794. Experiments B, C, and D remain close to that baseline, which suggests that the imported wildfire risk signals are directionally plausible but do not materially improve the best premium forecast under the current feature set and model choices. The most defensible conclusion is therefore that strong lagged insurance and exposure features dominate Task 2 performance in this snapshot.

Taken together, the two tasks point to a coherent storyline for the final submission: the classical baselines currently deliver the best predictive accuracy, while the quantum workflows add value as controlled prototype pipelines that quantify resource tradeoffs, establish a benchmark for future work, and create a pathway for hybrid wildfire-to-insurance feature transfer.

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

![Wildfire comparison figure](reports/figures/quantum_vs_classical.png)

### Table 2. Insurance Experiment A-D summary

| Experiment | Wildfire Input | Best Model | MAE | RMSE | R2 | MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| A | None | Gradient Boosting | 79,010.32 | 245,963.11 | 0.9794 | 145,505.94 |
| B | Provided score in dataset | Gradient Boosting | 79,590.69 | 249,453.45 | 0.9788 | 148,944.22 |
| C | Imported classical wildfire score | Gradient Boosting | 78,974.23 | 249,769.66 | 0.9788 | 141,841.39 |
| D | Imported quantum wildfire score | Gradient Boosting | 79,056.11 | 253,132.12 | 0.9782 | 143,532.48 |

![Insurance comparison figure](insurance-model/results/figures/experiment_comparison.png)

## Resource Requirements

Quantum resource requirements are summarized in Table 3. All committed quantum runs use a statevector simulator rather than a real quantum device. The quantum-kernel experiments operate at 4 or 6 qubits with decomposed feature-map depths between 31 and 49 and total runtimes between 6.85 seconds and 8.37 seconds for the exact kernel build plus SVM training. The VQC uses 4 qubits, a decomposed circuit depth of 27, 16 trainable parameters, and a substantially longer total runtime of 267.13 seconds.

Two resource implications matter for the challenge narrative. First, exact statevector kernel evaluation is manageable only because the training set is subsampled to 800 examples. Second, variational training is materially more expensive than the kernel runs in this snapshot despite using the same small qubit regime. Those tradeoffs should be highlighted explicitly because the challenge requests resource requirements, not just predictive metrics.

### Table 3. Quantum resource summary

| Model | Qubits | Depth | Backend | Train Samples | Runtime (s) | Best ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Quantum Kernel SVM (4q) | 4 | 31 | statevector_simulator | 800 | 6.8515 | 0.4701 |
| Quantum Kernel SVM (6q) | 6 | 49 | statevector_simulator | 800 | 8.3727 | 0.6222 |
| Quantum Kernel SVM (Pauli-4q) | 4 | 31 | statevector_simulator | 800 | 7.355 | 0.5436 |
| VQC (4q) | 4 | 27 | statevector_simulator | 800 | 267.13 | 0.4775 |

## Envisioned Algorithm

The envisioned challenge-aligned algorithm extends the current prototype into a full 2023 wildfire-risk forecasting workflow. The first step is to retrain Task 1 on the requested 2018-2022 history window and generate calibrated 2023 ZIP-level wildfire probabilities. That production run should preserve the strongest parts of the current preprocessing logic: row-type separation, annual weather aggregation, lagged fire-history features, and strict temporal evaluation discipline.

From a modeling standpoint, the most credible near-term architecture is a hybrid tiered system rather than a pure quantum replacement for the best classical baseline. A practical design would use the strongest classical model to screen the entire ZIP universe, then apply a quantum-kernel reranker or hybrid ensemble only to the most uncertain or highest-risk subset where a nonlinear similarity model might add value. This keeps the quantum workload resource-aware while still preserving a meaningful role for quantum learning.

For the insurance task, the 2023 wildfire probabilities can be injected as forward-looking exogenous signals rather than retrospective descriptors. That would make the wildfire model more useful to downstream premium forecasting, stress testing, and scenario analysis. A final submission could also extend the quantum side by evaluating shallower circuits, shot-based kernels, hardware-compatible transpilation constraints, and probability calibration. In short, the current repository provides a sound prototype foundation, while the envisioned algorithm turns it into a challenge-aligned hybrid forecasting system with clearer operational value.

## Clickable Repo Links

- [Repository root](https://github.com/RohanPatil2/deloitte-challenge-2026) - Main codebase and checked-in artifacts.
- [Wildfire quantum kernel pipeline](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/src/models/quantum/quantum_kernel_pipeline.py) - Primary Task 1 quantum-kernel implementation.
- [Wildfire VQC pipeline](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/src/models/quantum/vqc_pipeline.py) - Variational quantum classifier implementation.
- [Wildfire comparison table](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/reports/tables/model_comparison.csv) - Canonical cross-model comparison used in this report.
- [Quantum resource table](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/reports/tables/quantum_resources.csv) - Circuit depth, runtime, backend, and sample counts.
- [Insurance module](https://github.com/RohanPatil2/deloitte-challenge-2026/tree/main/insurance-model) - Task 2 source code, notebooks, and results.
- [Insurance combined metrics](https://github.com/RohanPatil2/deloitte-challenge-2026/blob/main/insurance-model/results/metrics/all_experiments_combined.csv) - Committed Experiment A-D result table.

Note: this report is a repository-backed submission draft built from the current checked-in artifacts on the `main` branch.
