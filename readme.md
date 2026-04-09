# 🔥 Deloitte Quantum Sustainability Challenge 2026: Hybrid Wildfire & Insurance Risk Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum_ML-purple.svg)](https://qiskit.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classical_Baselines-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Phase_1_Submission-success.svg)]()

> **Bridging the gap between localized climate risk and downstream financial modeling using hybrid Quantum-Classical Machine Learning.**

Welcome to our official Phase 1 submission repository for the **Deloitte Quantum Sustainability Challenge 2026**. This project tackles the complex intersection of climate change and finance by predicting California wildfire occurrences at the ZIP-year level (Tasks 1A & 1B) and evaluating how those climate risk signals improve downstream insurance premium forecasting (Task 2).

---

## 🧠 The Vision: Why Hybrid?
Current climate-finance models rely heavily on classical historical financial lags, which often fail to account for emerging, localized climate shifts. We propose a **two-layer hybrid risk-modeling stack**:
1. **The Climate Layer:** A Quantum/Classical machine learning pipeline that translates monthly weather behaviors and historical fire events into a calibrated Wildfire-Risk Score.
2. **The Financial Layer:** An actuarial forecasting model that tests whether embedding these newly generated climate risk scores provides measurable predictive lift for insurance premiums.

Rather than treating Quantum Machine Learning (QML) as an isolated benchmark, our architecture integrates it directly into a tangible sustainability and finance pipeline.

---

## 🏗️ Architecture & Approach

### Part 1: Wildfire Risk Prediction (Tasks 1A & 1B)
* **Data Processing:** Aggregated 125,476 raw weather/fire records into 10,372 annual ZIP-year observations. Extracted temperature ranges, precipitation, and historical fire lags.
* **Classical Baselines:** Logistic Regression, Random Forest, Gradient Boosting, and RBF-kernel SVMs.
* **Quantum Models:** * **Quantum Kernel SVMs:** Utilizing exact-statevector simulations, mutual-information feature selection, and benchmarking ZZ vs. Pauli Feature Maps across 4 to 6 qubits.
  * **Variational Quantum Classifier (VQC):** A 4-qubit VQC utilizing a RealAmplitudes ansatz and COBYLA optimization.
* **Validation:** Strict temporal split (Train: 2018–2020 | Validate: 2021) to emulate real-world forward-looking prediction and prevent data leakage.

### Part 2: Insurance Premium Forecasting (Task 2)
* **Data Processing:** Preprocessed 47,033 ZIP-level insurance records, enforcing negative-premium clipping (guided by our EDA) and constructing robust financial lag features.
* **Experimentation Framework:** We trained Gradient Boosting, Random Forest, and Ridge regressors across four distinct scenarios:
  * **Exp A:** No external wildfire input (Financial baseline).
  * **Exp B:** Using the provided dataset wildfire score.
  * **Exp C:** Using our imported *Classical* wildfire score (from Task 1).
  * **Exp D:** Using our imported *Quantum* wildfire score (from Task 1).

---

## 📊 Key Results

### Wildfire Classification Performance (Validation Set: 2021)
*Classical models leveraged the full 7,779 training set, while quantum models were capped at 800 samples to maintain exact-kernel tractability.*

| Model | Type | Qubits | ROC-AUC | F1 Score | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Classical | - | **0.8647** | 0.3546 | 0.8341 |
| Gradient Boosting | Classical | - | 0.8471 | 0.2494 | 0.9193 |
| **Quantum Kernel SVM (6q)** | Quantum | 6 | **0.6222** | 0.0552 | 0.0673 |
| Quantum Kernel SVM (Pauli-4q)| Quantum | 4 | 0.5436 | 0.1690 | 0.4798 |
| VQC (4q) | Quantum | 4 | 0.4775 | 0.1356 | 0.5471 |

### Quantum Resource Tracking
We believe transparent resource tracking is vital for the quantum community's growth. Below is the footprint of our QML workflows (Statevector Simulator):

| Model | Qubits | Circuit Depth | Runtime (s) | 
| :--- | :---: | :---: | :---: | 
| Quantum Kernel (4q ZZ) | 4 | 31 | 6.85 | 
| Quantum Kernel (Pauli-4q) | 4 | 31 | 7.35 | 
| Quantum Kernel (6q ZZ) | 6 | 49 | 8.37 |
| VQC (4q) | 4 | 27 | 267.13 | 

### Insurance Premium Forecasting (Gradient Boosting)
Our experiments revealed that while historical financial lags dominate the predictive power ($R^2$ = 0.9794), wildfire-risk features remain a directionally valid auxiliary signal that can be injected into the actuarial model without degrading accuracy.

| Experiment | Wildfire Input | MAE | $R^2$ |
| :--- | :--- | :---: | :---: |
| **A** | None | 79,010.32 | **0.9794** |
| **B** | Provided Dataset Score | 79,590.69 | 0.9788 |
| **C** | Imported Classical Score | 78,974.23 | 0.9788 |
| **D** | Imported Quantum Score | 79,056.11 | 0.9782 |

*(For full visual analysis, including Actual vs. Predicted variance mapping and premium distributions, refer to the `/figures` directory).*

---

## 📂 Repository Structure

Navigate our codebase using the links below. The repository is modularized to separate the quantum/classical climate models from the downstream financial insurance models.

deloitte-challenge-2026/
├── src/
│   └── models/
│       └── quantum/
│           ├── quantum_kernel_pipeline.py  # Primary Task 1 Q-Kernel implementation
│           └── vqc_pipeline.py             # Variational Quantum Classifier implementation
├── insurance-model/                        # Task 2 Insurance Module
│   ├── notebooks/                          # EDA and experiment formulation
│   ├── src/                                # Insurance feature engineering and regression
│   └── results/                            
│       ├── metrics/all_experiments_combined.csv
│       └── figures/experiment_comparison.png
├── reports/
│   └── tables/                             # Canonical cross-model comparison CSVs
├── figures/                                # Report graphics and EDA visual evidence
│   ├── quantum_vs_classical.png
│   ├── premium_distribution.png
│   ├── lag_feature_correlations.png
│   └── exp_A_actual_vs_predicted.png
├── README.md
└── requirements.txt
🚀 Getting Started & Reproducibility
To ensure maximum transparency, our pipeline is fully reproducible.

1. Clone the repository:

Bash
git clone [https://github.com/RohanPatil2/deloitte-challenge-2026.git](https://github.com/RohanPatil2/deloitte-challenge-2026.git)
cd deloitte-challenge-2026
2. Set up the environment:
(We recommend using a virtual environment like conda or venv)

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
3. Run the Quantum Wildfire Pipeline:

Bash
python src/models/quantum/quantum_kernel_pipeline.py
4. Run the Insurance Evaluation:
Navigate to the insurance-model directory and execute the main experiment runner to reproduce the metrics for Experiments A-D.

🔭 The Envisioned Future (Scaling for Phase 2 & Beyond)
This repository represents a robust prototype. The natural evolution of this framework—a fully challenge-aligned 2023 production model—will require:

Hardware Execution: Transitioning from statevector simulations to shot-based real quantum hardware execution to map practical noise profiles.

Scalability Strategies: Implementing PCA-based dimension reduction or data re-uploading circuits to shatter the current 800-sample processing cap for quantum kernels.

Probability Calibration: Ensuring QML outputs produce calibrated probabilities suitable for direct ingestion by enterprise actuarial models.

👥 The Team
Rohan Patil | Repository Integration, Document Packaging, Delivery Review

Shreyas Khandale | Quantum Machine Learning, Baselines, Comparative Evaluation

Aishwarya Das | Insurance Analytics, Feature Engineering, Report Editing

Affiliation: Binghamton University ---
Created for the Deloitte Quantum Sustainability Challenge 2026. Data strictly utilized per competition rules.


### A few quick tips before you commit this:
1. **Ensure `requirements.txt` exists:** If you don't have one yet, make sure to generate it (`pip freeze > requirements.txt`) so the judges can easily install your dependencies (Qiskit, Scikit-learn, Pandas, etc.).
2. **Images rendering:** Since you have a `figures/` folder, the relative links in your README will automatically display if you ever choose to embed them using the `![Alt Text](figures/image.png)` Markdown syntax, but the tree structure above is usually enough to let reviewers know where to look.
