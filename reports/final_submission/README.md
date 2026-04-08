# Final Submission Package

This directory contains the assembled final submission draft for the Deloitte Quantum Sustainability Challenge 2026.

## Files

- `build_submission_package.py`: generates the Markdown report and the PDF.
- `check_submission.py`: verifies that required inputs and generated outputs exist.
- `Deloitte_Quantum_Sustainability_Challenge_2026_Submission.md`: editable generated report source.
- `Deloitte_Quantum_Sustainability_Challenge_2026_Submission.pdf`: generated submission PDF.

## Build

```bash
./.venv-report/bin/python reports/final_submission/build_submission_package.py
./.venv-report/bin/python reports/final_submission/check_submission.py
```

## Notes

- The report is built from the currently checked-in repository artifacts on `main`.
- The repository remote currently exposes only `origin/main`; the branch names `quantum-model`, `data-baselines`, and `insurance` are not available to merge from this checkout.
