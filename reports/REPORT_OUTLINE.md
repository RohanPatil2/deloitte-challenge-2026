# Report Outline

Use this as the working structure for the written submission and pitch deck narrative.

## 1. Executive Summary

- One paragraph on the challenge goal.
- One paragraph on the team's approach: wildfire prediction plus insurance premium modeling.
- One paragraph on the main takeaway from the current checked-in experiments.

## 2. Challenge Scope And Alignment

- State the official Task 1 and Task 2 objectives from the Deloitte brief.
- State clearly what the current repository implements today.
- Do not blur validation experiments with final forecasting deliverables.

Suggested wording:

> The current repository demonstrates comparative modeling pipelines on held-out validation periods. It should be presented as a prototype and evaluation framework unless final challenge-aligned forecasting runs are generated separately.

## 3. Data Sources And Preprocessing

Cover:

- wildfire data sources and row-type separation
- ZIP-year aggregation logic
- lag-feature construction
- insurance data cleaning, feature engineering, and temporal split
- known data-quality issues from the challenge brief

Call out the challenge notes explicitly:

- malformed `year_month` values
- `OBJECTIVE` processing artifacts
- `AGENCY_ID` encoding caveat
- double-space insurance column names

## 4. Task 1: Wildfire Prediction Approach

Describe:

- classical baselines
- quantum kernel SVM workflow
- VQC workflow
- feature selection and scaling
- simulator/resource assumptions

Recommended figures/tables:

- `reports/tables/model_comparison.csv`
- `reports/tables/quantum_resources.csv`
- `reports/figures/quantum_vs_classical.png`

## 5. Task 1: Results And Interpretation

Minimum points to cover:

- best classical ROC-AUC in checked-in artifacts: `0.8647`
- best quantum-kernel ROC-AUC in checked-in artifacts: `0.6222`
- VQC ROC-AUC in checked-in artifacts: `0.4775`
- train-sample counts differ across the committed comparison artifacts and should be stated explicitly

Keep the interpretation disciplined:

- explain what the quantum experiments demonstrate
- explain what they do not yet demonstrate
- distinguish resource/feasibility insight from outright performance superiority

## 6. Task 2: Insurance Premium Modeling

Describe:

- experiment design for A/B/C/D
- temporal split: 2018-2020 train, 2021 test
- feature families used in the insurance model
- how wildfire-risk outputs are injected into Experiments C and D

Recommended result source:

- `insurance-model/results/metrics/all_experiments_combined.csv`

## 7. Task 2: Results And Interpretation

Minimum points to cover:

- best checked-in `R2`: `0.9794` from Gradient Boosting in Experiment A
- experiments with wildfire-risk features are close to the baseline in current committed runs
- explain whether wildfire-risk inputs improved robustness, interpretability, or only marginally changed fit

## 8. Quantum Resource Discussion

Include:

- qubit counts
- circuit depth
- simulator backend
- runtime tradeoffs
- why subsampling was used in the quantum experiments

This section matters because the challenge explicitly asks for resource requirements, not just accuracy.

## 9. Limitations

Suggested bullets:

- current repo appears to be a validation-stage implementation rather than the final challenge forecast horizon
- quantum experiments are compute-constrained
- model comparisons should be contextualized by sample counts and feature constraints
- insurance improvements from imported wildfire-risk features are modest in current artifacts

## 10. Conclusion

Close with:

- what the prototype proves
- what needs to happen for a final challenge submission
- where quantum methods added value in exploration, workflow, or resource analysis

## 11. Submission Checklist

- confirm challenge-aligned forecast horizon is stated accurately
- confirm tables and figures match committed artifacts
- confirm terminology is consistent: prototype, validation, simulator, resource estimate
- confirm no result is described more strongly than the evidence supports
