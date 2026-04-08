#!/usr/bin/env python3
"""
Build the final submission package in both Markdown and PDF form.

Outputs:
  - Deloitte_Quantum_Sustainability_Challenge_2026_Submission.md
  - Deloitte_Quantum_Sustainability_Challenge_2026_Submission.pdf
"""

from __future__ import annotations

import csv
import math
import textwrap
from datetime import date
from pathlib import Path

try:
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError as exc:  # pragma: no cover - build-time dependency guard
    raise SystemExit(
        "Missing PDF dependencies. Use .venv-report/bin/python to run this script."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "Deloitte_Quantum_Sustainability_Challenge_2026_Submission"
OUTPUT_MD = REPORT_DIR / f"{OUTPUT_STEM}.md"
OUTPUT_PDF = REPORT_DIR / f"{OUTPUT_STEM}.pdf"

REMOTE_ROOT = "https://github.com/RohanPatil2/deloitte-challenge-2026"

MODEL_COMPARISON_CSV = REPO_ROOT / "reports" / "tables" / "model_comparison.csv"
QUANTUM_RESOURCES_CSV = REPO_ROOT / "reports" / "tables" / "quantum_resources.csv"
INSURANCE_RESULTS_CSV = (
    REPO_ROOT / "insurance-model" / "results" / "metrics" / "all_experiments_combined.csv"
)

WILDFIRE_FIGURE = REPO_ROOT / "reports" / "figures" / "quantum_vs_classical.png"
INSURANCE_FIGURE = (
    REPO_ROOT / "insurance-model" / "results" / "figures" / "experiment_comparison.png"
)


TEAM_ROWS = [
    {
        "name": "Rohan Patil",
        "role": "Repository integration, reproducibility, and final packaging",
        "contact": "rohanpatil0212@gmail.com",
    },
    {
        "name": "Shreyas Khandale",
        "role": "Wildfire modeling, quantum experiments, and classical baselines",
        "contact": "Verify preferred submission email before sending",
    },
    {
        "name": "Aishwarya Das",
        "role": "Insurance modeling, experiment analysis, and report assembly",
        "contact": "adas21@binghamton.edu",
    },
]


DATA_POINTS = {
    "wildfire_raw_rows": "125,476",
    "wildfire_raw_cols": "30",
    "wildfire_weather_rows": "123,258",
    "wildfire_fire_rows": "2,218",
    "wildfire_dataset_rows": "10,372",
    "wildfire_unique_zips": "2,593",
    "wildfire_positive_events": "875",
    "wildfire_train_rows": "7,779",
    "wildfire_val_rows": "2,593",
    "wildfire_train_fire_rate": "8.38%",
    "wildfire_val_fire_rate": "8.60%",
    "insurance_raw_rows": "47,033",
    "insurance_raw_cols": "76",
    "insurance_unique_zips": "2,251",
    "insurance_train_rows": "33,689",
    "insurance_test_rows": "13,344",
    "insurance_negative_premiums": "7",
    "insurance_double_space_cols": "16",
}


REPO_LINKS = [
    (
        "Repository root",
        REMOTE_ROOT,
        "Main codebase and checked-in artifacts.",
    ),
    (
        "Wildfire quantum kernel pipeline",
        f"{REMOTE_ROOT}/blob/main/src/models/quantum/quantum_kernel_pipeline.py",
        "Primary Task 1 quantum-kernel implementation.",
    ),
    (
        "Wildfire VQC pipeline",
        f"{REMOTE_ROOT}/blob/main/src/models/quantum/vqc_pipeline.py",
        "Variational quantum classifier implementation.",
    ),
    (
        "Wildfire comparison table",
        f"{REMOTE_ROOT}/blob/main/reports/tables/model_comparison.csv",
        "Canonical cross-model comparison used in this report.",
    ),
    (
        "Quantum resource table",
        f"{REMOTE_ROOT}/blob/main/reports/tables/quantum_resources.csv",
        "Circuit depth, runtime, backend, and sample counts.",
    ),
    (
        "Insurance module",
        f"{REMOTE_ROOT}/tree/main/insurance-model",
        "Task 2 source code, notebooks, and results.",
    ),
    (
        "Insurance combined metrics",
        f"{REMOTE_ROOT}/blob/main/insurance-model/results/metrics/all_experiments_combined.csv",
        "Committed Experiment A-D result table.",
    ),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt_float(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def load_wildfire_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(MODEL_COMPARISON_CSV)
    trimmed = []
    for row in rows:
        trimmed.append(
            {
                "Model": row["Model"],
                "Type": row["Type"],
                "Qubits": row["Qubits"],
                "ROC-AUC": fmt_float(as_float(row["ROC-AUC"])),
                "PR-AUC": fmt_float(as_float(row["PR-AUC"])),
                "F1": fmt_float(as_float(row["F1"])),
                "Recall": fmt_float(as_float(row["Recall"])),
                "Train Samples": row["Train_samples"],
            }
        )
    return trimmed


def load_insurance_best_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(INSURANCE_RESULTS_CSV)
    experiment_order = [
        "A - No Wildfire Risk",
        "B - Provided Fire Risk Score",
        "C - Classical Wildfire Risk",
        "D - Quantum Wildfire Risk",
    ]
    wildfire_inputs = {
        "A - No Wildfire Risk": "None",
        "B - Provided Fire Risk Score": "Provided score in dataset",
        "C - Classical Wildfire Risk": "Imported classical wildfire score",
        "D - Quantum Wildfire Risk": "Imported quantum wildfire score",
    }

    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in experiment_order}
    for row in rows:
        grouped[row["Experiment"]].append(row)

    summary = []
    for experiment in experiment_order:
        best = max(grouped[experiment], key=lambda item: as_float(item["R2"]))
        summary.append(
            {
                "Experiment": experiment.split(" - ", 1)[0],
                "Wildfire Input": wildfire_inputs[experiment],
                "Best Model": best["Model"],
                "MAE": f"{as_float(best['MAE']):,.2f}",
                "RMSE": f"{as_float(best['RMSE']):,.2f}",
                "R2": fmt_float(as_float(best["R2"])),
                "MAPE": f"{as_float(best['MAPE']):,.2f}",
            }
        )
    return summary


def load_quantum_resource_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(QUANTUM_RESOURCES_CSV)
    trimmed = []
    for row in rows:
        trimmed.append(
            {
                "Model": row["Model"],
                "Qubits": row["Qubits"],
                "Depth": row["Circuit Depth"],
                "Backend": row["Backend"],
                "Train Samples": row["Train Samples"],
                "Runtime (s)": row["Total Runtime (s)"],
                "Best ROC-AUC": row["Best ROC-AUC"],
            }
        )
    return trimmed


def build_summary_stats() -> dict[str, str]:
    wildfire_rows = read_csv_rows(MODEL_COMPARISON_CSV)
    insurance_rows = read_csv_rows(INSURANCE_RESULTS_CSV)

    classical_best = max(
        (row for row in wildfire_rows if row["Type"] == "Classical"),
        key=lambda item: as_float(item["ROC-AUC"]),
    )
    quantum_kernel_best = max(
        (row for row in wildfire_rows if row["Type"] == "Quantum Kernel"),
        key=lambda item: as_float(item["ROC-AUC"]),
    )
    vqc_row = next(row for row in wildfire_rows if row["Model"] == "VQC (4q)")
    insurance_best = max(insurance_rows, key=lambda item: as_float(item["R2"]))

    return {
        "classical_best_model": classical_best["Model"],
        "classical_best_roc": classical_best["ROC-AUC"],
        "classical_best_samples": classical_best["Train_samples"],
        "quantum_best_model": quantum_kernel_best["Model"],
        "quantum_best_roc": quantum_kernel_best["ROC-AUC"],
        "quantum_best_samples": quantum_kernel_best["Train_samples"],
        "vqc_roc": vqc_row["ROC-AUC"],
        "insurance_best_experiment": insurance_best["Experiment"],
        "insurance_best_model": insurance_best["Model"],
        "insurance_best_r2": insurance_best["R2"],
    }


def word_count(text: str) -> int:
    return len([token for token in text.split() if token.strip()])


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, divider, *body])


def get_report_text() -> dict[str, object]:
    summary = build_summary_stats()

    abstract = textwrap.dedent(
        f"""
        This submission presents a two-stage modeling workflow for Deloitte's Quantum Sustainability Challenge 2026. The first stage predicts wildfire occurrence risk at the California ZIP-year level, and the second stage studies whether wildfire-risk signals improve a downstream insurance premium forecast. In the checked-in wildfire data, the raw file contains {DATA_POINTS['wildfire_raw_rows']} rows and {DATA_POINTS['wildfire_raw_cols']} columns, combining {DATA_POINTS['wildfire_weather_rows']} weather rows with {DATA_POINTS['wildfire_fire_rows']} fire-event rows. We aggregate monthly weather observations into annual ZIP summaries, build a binary fire-event label, and engineer lagged fire-history features that capture recent activity, cumulative history, and a prior-fire indicator. We then compare four classical baselines against two quantum learning approaches on a strict temporal split using 2018-2020 training data and 2021 validation data. The classical baseline suite includes Logistic Regression, Random Forest, Gradient Boosting, and an RBF-kernel SVM. The quantum suite includes exact-statevector quantum kernel SVMs built from ZZFeatureMap and PauliFeatureMap circuits, plus a 4-qubit variational quantum classifier using COBYLA and a custom class-balanced loss. Because kernel evaluation scales quadratically, the quantum runs are capped at {summary['quantum_best_samples']} stratified training examples, while the best classical baseline uses the full {summary['classical_best_samples']} training examples. In the committed artifacts, {summary['classical_best_model']} achieves the strongest wildfire ROC-AUC at {summary['classical_best_roc']}, while the best quantum kernel model reaches {summary['quantum_best_roc']} and the VQC reaches {summary['vqc_roc']}. These results do not yet demonstrate a quantum accuracy advantage, but they do provide a grounded benchmark for circuit depth, simulator runtime, feature compression, and hybrid-integration tradeoffs. Resource analysis is therefore part of the contribution: the kernel models remain comparatively fast on the simulator, while the variational model is meaningfully slower even at four qubits, clarifying which quantum path is more practical under current constraints. For Task 2, we forecast 2021 insurance premiums from 2018-2020 history using lagged premium signals, loss-based features, exposure measures, fire-risk bands, product-category flags, and census variables. We compare four experiments: a no-wildfire baseline, a provided wildfire score, a classical wildfire score, and a quantum wildfire score. In the checked-in results, {summary['insurance_best_model']} in {summary['insurance_best_experiment']} delivers the best overall insurance fit with R2 = {summary['insurance_best_r2']}. The wildfire-augmented experiments remain close to that baseline, which suggests that imported wildfire risk is directionally useful but not yet the dominant driver of premium accuracy when strong lagged insurance features are already present. Overall, the repository currently supports a scientifically useful prototype: classical models lead predictive performance, while the quantum workflows quantify how a constrained hybrid pipeline could evolve toward the full 2023 wildfire-forecasting objective described in the challenge.
        """
    ).strip()

    if not 350 <= word_count(abstract) <= 450:
        raise SystemExit(f"Abstract word count is {word_count(abstract)}, expected about 400.")

    team_note = (
        "Primary submission contact can be listed as Rohan Patil. "
        "Names and emails above were inferred from repository metadata and should be "
        "verified once before external submission."
    )

    data_paragraphs = [
        (
            "The wildfire workstream uses a mixed raw dataset that combines monthly "
            "weather observations with historical fire-event rows. In the checked-in "
            f"snapshot, the raw file contains {DATA_POINTS['wildfire_raw_rows']} rows across "
            f"{DATA_POINTS['wildfire_raw_cols']} columns. The file includes "
            f"{DATA_POINTS['wildfire_weather_rows']} weather rows and {DATA_POINTS['wildfire_fire_rows']} "
            "fire rows. The modeling pipeline normalizes ZIP codes, aggregates monthly "
            "weather into annual ZIP summaries, marks whether any fire occurred in a "
            "given ZIP-year, and engineers fire-history features such as lag-1 count, "
            "lag-2 count, cumulative fire count, and an ever-had-fire indicator. After "
            f"filtering to the current evaluation window, the working wildfire dataset has "
            f"{DATA_POINTS['wildfire_dataset_rows']} ZIP-year rows covering "
            f"{DATA_POINTS['wildfire_unique_zips']} ZIP codes, with "
            f"{DATA_POINTS['wildfire_positive_events']} positive wildfire events."
        ),
        (
            "The current wildfire experiments in this repository are validation-stage "
            "studies rather than the final challenge horizon. Training uses 2018-2020 "
            f"ZIP-year observations ({DATA_POINTS['wildfire_train_rows']} rows, fire rate "
            f"{DATA_POINTS['wildfire_train_fire_rate']}), and evaluation uses 2021 "
            f"({DATA_POINTS['wildfire_val_rows']} rows, fire rate "
            f"{DATA_POINTS['wildfire_val_fire_rate']}). This distinction should be stated "
            "clearly in the final submission because the challenge brief ultimately asks "
            "for wildfire-risk prediction in 2023 using history through 2022."
        ),
        (
            "The insurance workstream uses a separate ZIP-level panel containing "
            f"{DATA_POINTS['insurance_raw_rows']} rows and {DATA_POINTS['insurance_raw_cols']} "
            "columns. The checked-in file covers 2018-2021 and spans "
            f"{DATA_POINTS['insurance_unique_zips']} unique ZIP codes. The model uses a "
            "strict temporal split with 2018-2020 as training history and 2021 as the "
            f"holdout year ({DATA_POINTS['insurance_train_rows']} training rows and "
            f"{DATA_POINTS['insurance_test_rows']} test rows). Data cleaning includes "
            f"clipping {DATA_POINTS['insurance_negative_premiums']} negative premium rows "
            "to zero and careful handling of wide insurance column names that contain "
            f"embedded double spaces ({DATA_POINTS['insurance_double_space_cols']} such "
            "columns in the checked-in file)."
        ),
        (
            "The challenge briefing also calls out data-quality issues such as malformed "
            "date fields, categorical-code quirks, and naming inconsistencies. Those "
            "notes informed the cleaning strategy even where the committed CSV snapshot is "
            "already comparatively clean. In particular, the project treats row-type "
            "separation, column-name stability, and temporal partitioning as first-order "
            "requirements for both model validity and reproducibility."
        ),
    ]

    methodology_paragraphs = [
        (
            "The wildfire modeling pipeline shares a common preprocessing backbone across "
            "classical and quantum runs. Monthly weather rows are aggregated to annual "
            "ZIP-year features using mean maximum temperature, mean minimum temperature, "
            "total precipitation, and derived temperature range. Fire rows are collapsed "
            "into a binary target that marks whether a wildfire event occurred in the same "
            "ZIP-year. Historical fire activity is then encoded via lagged counts, an "
            "exclusive cumulative fire count, and a prior-fire flag. This produces a "
            "compact feature set designed to balance predictive signal with quantum "
            "tractability."
        ),
        (
            "The classical wildfire benchmarks consist of Logistic Regression, Random "
            "Forest, Gradient Boosting, and an RBF-kernel SVM. All models are evaluated "
            "on the same 2021 holdout period, and class imbalance is handled through "
            "balanced class weights or equivalent sample weighting. Logistic Regression "
            "and the classical SVM use standardized features, while the tree-based models "
            "operate directly on the engineered inputs. These baselines establish an "
            "accuracy reference before any quantum modeling claims are made."
        ),
        (
            "The quantum-kernel workflow compresses the candidate feature set through "
            "mutual-information-based selection, then maps the resulting 4- or 6-feature "
            "vectors into parameterized quantum feature maps. The committed experiments "
            "use exact-statevector FidelityStatevectorKernel evaluation, balanced SVC "
            "classification on a precomputed kernel matrix, and a stratified cap of 800 "
            "training samples to keep O(n^2) kernel construction tractable. Two ZZ "
            "feature-map runs (4 qubits and 6 qubits) are paired with a 4-qubit Pauli "
            "feature-map ablation to test whether circuit design changes the resulting "
            "similarity structure."
        ),
        (
            "The VQC workflow further compresses the wildfire task into a 4-qubit "
            "variational model composed of a ZZFeatureMap encoder and a RealAmplitudes "
            "ansatz. Training uses the COBYLA optimizer for 100 iterations and a custom "
            "weighted cross-entropy loss so the minority wildfire class receives more "
            "attention during optimization. This setup prioritizes proof-of-concept "
            "comparability over hardware realism and therefore remains a simulator-based "
            "prototype."
        ),
        (
            "The insurance methodology is intentionally modular. The preprocessing layer "
            "loads the raw ZIP-level insurance panel, removes leakage-prone post-event "
            "fire columns and unusable weather columns, clips negative premiums, and "
            "encodes boolean fields numerically. Feature engineering then builds premium "
            "lag features, year-over-year premium trend features, lagged loss ratios, and "
            "premium-per-exposure measures. The four insurance experiments differ only in "
            "which wildfire-risk input is appended: no additional wildfire score "
            "(Experiment A), the provided dataset wildfire-risk score (Experiment B), an "
            "imported classical wildfire-risk score (Experiment C), or an imported "
            "quantum wildfire-risk score (Experiment D)."
        ),
        (
            "Task 2 modeling compares a naive lag-1 baseline against Ridge regression, "
            "Random Forest regression, and Gradient Boosting regression. Ridge is trained "
            "on standardized inputs, while the tree-based models operate on the raw "
            "feature matrix. All experiments respect the same temporal split: 2018-2020 "
            "for training and 2021 for testing. This design makes it possible to ask a "
            "narrow question cleanly: do wildfire-risk signals materially improve premium "
            "prediction once strong lagged insurance features are already available?"
        ),
    ]

    results_paragraphs = [
        (
            f"Wildfire classification results are summarized in Table 1. In the current "
            f"repository snapshot, {summary['classical_best_model']} is the strongest "
            f"classical model with ROC-AUC = {summary['classical_best_roc']}. The best "
            f"quantum-kernel result is {summary['quantum_best_model']} with ROC-AUC = "
            f"{summary['quantum_best_roc']}, and the VQC reaches ROC-AUC = "
            f"{summary['vqc_roc']}. The Pauli 4-qubit kernel ablation performs better than "
            "the ZZ 4-qubit kernel, which suggests that feature-map choice matters even "
            "under a small-qubit regime."
        ),
        (
            "These wildfire results should be interpreted carefully. The strongest "
            "classical baseline is trained on the full 7,779-example training set, while "
            "the quantum models are capped at 800 examples for tractability on a classical "
            "simulator. The comparison is still useful for benchmarking current pipeline "
            "behavior, but it should not be overstated as a like-for-like statement about "
            "quantum versus classical learning capacity. What the experiments do show is "
            "that the current quantum workflows are operational, measurable, and "
            "meaningfully sensitive to qubit count, circuit family, and runtime budget."
        ),
        (
            "Insurance results are summarized in Table 2 and Figure 2. Across all four "
            "experiments, Gradient Boosting is the best-performing model family in the "
            "checked-in metrics. The strongest overall result comes from Experiment A "
            "(no external wildfire-risk feature) with R2 = 0.9794. Experiments B, C, and "
            "D remain close to that baseline, which suggests that the imported wildfire "
            "risk signals are directionally plausible but do not materially improve the "
            "best premium forecast under the current feature set and model choices. The "
            "most defensible conclusion is therefore that strong lagged insurance and "
            "exposure features dominate Task 2 performance in this snapshot."
        ),
        (
            "Taken together, the two tasks point to a coherent storyline for the final "
            "submission: the classical baselines currently deliver the best predictive "
            "accuracy, while the quantum workflows add value as controlled prototype "
            "pipelines that quantify resource tradeoffs, establish a benchmark for "
            "future work, and create a pathway for hybrid wildfire-to-insurance feature "
            "transfer."
        ),
    ]

    resource_paragraphs = [
        (
            "Quantum resource requirements are summarized in Table 3. All committed "
            "quantum runs use a statevector simulator rather than a real quantum device. "
            "The quantum-kernel experiments operate at 4 or 6 qubits with decomposed "
            "feature-map depths between 31 and 49 and total runtimes between 6.85 seconds "
            "and 8.37 seconds for the exact kernel build plus SVM training. The VQC uses "
            "4 qubits, a decomposed circuit depth of 27, 16 trainable parameters, and a "
            "substantially longer total runtime of 267.13 seconds."
        ),
        (
            "Two resource implications matter for the challenge narrative. First, exact "
            "statevector kernel evaluation is manageable only because the training set is "
            "subsampled to 800 examples. Second, variational training is materially more "
            "expensive than the kernel runs in this snapshot despite using the same small "
            "qubit regime. Those tradeoffs should be highlighted explicitly because the "
            "challenge requests resource requirements, not just predictive metrics."
        ),
    ]

    envisioned_paragraphs = [
        (
            "The envisioned challenge-aligned algorithm extends the current prototype into "
            "a full 2023 wildfire-risk forecasting workflow. The first step is to retrain "
            "Task 1 on the requested 2018-2022 history window and generate calibrated 2023 "
            "ZIP-level wildfire probabilities. That production run should preserve the "
            "strongest parts of the current preprocessing logic: row-type separation, "
            "annual weather aggregation, lagged fire-history features, and strict temporal "
            "evaluation discipline."
        ),
        (
            "From a modeling standpoint, the most credible near-term architecture is a "
            "hybrid tiered system rather than a pure quantum replacement for the best "
            "classical baseline. A practical design would use the strongest classical model "
            "to screen the entire ZIP universe, then apply a quantum-kernel reranker or "
            "hybrid ensemble only to the most uncertain or highest-risk subset where a "
            "nonlinear similarity model might add value. This keeps the quantum workload "
            "resource-aware while still preserving a meaningful role for quantum learning."
        ),
        (
            "For the insurance task, the 2023 wildfire probabilities can be injected as "
            "forward-looking exogenous signals rather than retrospective descriptors. That "
            "would make the wildfire model more useful to downstream premium forecasting, "
            "stress testing, and scenario analysis. A final submission could also extend "
            "the quantum side by evaluating shallower circuits, shot-based kernels, "
            "hardware-compatible transpilation constraints, and probability calibration. "
            "In short, the current repository provides a sound prototype foundation, while "
            "the envisioned algorithm turns it into a challenge-aligned hybrid forecasting "
            "system with clearer operational value."
        ),
    ]

    return {
        "title": "Deloitte Quantum Sustainability Challenge 2026",
        "subtitle": "Final Submission Draft: Wildfire Prediction and Insurance Premium Modeling",
        "prepared": f"Prepared {date.today().strftime('%B %d, %Y')}",
        "team_note": team_note,
        "abstract": abstract,
        "sections": [
            ("Team Overview and Contact Details", []),
            ("Data", data_paragraphs),
            ("Methodology", methodology_paragraphs),
            ("Results", results_paragraphs),
            ("Resource Requirements", resource_paragraphs),
            ("Envisioned Algorithm", envisioned_paragraphs),
            ("Clickable Repo Links", []),
        ],
    }


def scale_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def make_table(headers: list[str], rows: list[list[str]], col_widths: list[float]) -> Table:
    table_data = [headers, *rows]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa9b8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fb")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_markdown(report: dict[str, object]) -> str:
    wildfire_rows = load_wildfire_rows()
    insurance_rows = load_insurance_best_rows()
    resource_rows = load_quantum_resource_rows()

    lines: list[str] = []
    lines.append(f"# {report['title']}")
    lines.append("")
    lines.append(f"## {report['subtitle']}")
    lines.append("")
    lines.append(report["prepared"])
    lines.append("")
    lines.append("## Team Overview and Contact Details")
    lines.append("")
    lines.append(
        markdown_table(
            ["Name", "Role", "Contact"],
            [[row["name"], row["role"], row["contact"]] for row in TEAM_ROWS],
        )
    )
    lines.append("")
    lines.append(report["team_note"])
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(report["abstract"])
    lines.append("")
    lines.append("## Data")
    lines.append("")
    for paragraph in report["sections"][1][1]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Methodology")
    lines.append("")
    for paragraph in report["sections"][2][1]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Results")
    lines.append("")
    for paragraph in report["sections"][3][1]:
        lines.append(paragraph)
        lines.append("")
    lines.append("### Table 1. Wildfire model comparison")
    lines.append("")
    lines.append(
        markdown_table(
            list(wildfire_rows[0].keys()),
            [list(row.values()) for row in wildfire_rows],
        )
    )
    lines.append("")
    lines.append(f"![Wildfire comparison figure]({WILDFIRE_FIGURE.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append("### Table 2. Insurance Experiment A-D summary")
    lines.append("")
    lines.append(
        markdown_table(
            list(insurance_rows[0].keys()),
            [list(row.values()) for row in insurance_rows],
        )
    )
    lines.append("")
    lines.append(f"![Insurance comparison figure]({INSURANCE_FIGURE.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append("## Resource Requirements")
    lines.append("")
    for paragraph in report["sections"][4][1]:
        lines.append(paragraph)
        lines.append("")
    lines.append("### Table 3. Quantum resource summary")
    lines.append("")
    lines.append(
        markdown_table(
            list(resource_rows[0].keys()),
            [list(row.values()) for row in resource_rows],
        )
    )
    lines.append("")
    lines.append("## Envisioned Algorithm")
    lines.append("")
    for paragraph in report["sections"][5][1]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Clickable Repo Links")
    lines.append("")
    for label, url, note in REPO_LINKS:
        lines.append(f"- [{label}]({url}) - {note}")
    lines.append("")
    lines.append(
        "Note: this report is a repository-backed submission draft built from the "
        "current checked-in artifacts on the `main` branch."
    )
    lines.append("")
    return "\n".join(lines)


def add_section_header(story: list, text: str, styles) -> None:
    story.append(Paragraph(text, styles["Heading1Custom"]))
    story.append(Spacer(1, 0.12 * inch))


def add_paragraphs(story: list, paragraphs: list[str], styles) -> None:
    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, styles["BodyCustom"]))
        story.append(Spacer(1, 0.12 * inch))


def build_pdf(report: dict[str, object]) -> None:
    wildfire_rows = load_wildfire_rows()
    insurance_rows = load_insurance_best_rows()
    resource_rows = load_quantum_resource_rows()

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#16324f"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubtitleCustom",
            parent=styles["Heading2"],
            fontName="Helvetica",
            fontSize=13,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#36566f"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1Custom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#16324f"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionCustom",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#4f6475"),
            alignment=TA_CENTER,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.55 * inch,
        title=report["subtitle"],
        author="Rohan Patil, Shreyas Khandale, Aishwarya Das",
    )

    story: list = []
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(report["title"], styles["TitleCustom"]))
    story.append(Paragraph(report["subtitle"], styles["SubtitleCustom"]))
    story.append(Paragraph(report["prepared"], styles["BodyCustom"]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Submission package built from the current checked-in repository artifacts.",
            styles["BodyCustom"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    add_section_header(story, "Team Overview and Contact Details", styles)
    team_table = make_table(
        ["Name", "Role", "Contact"],
        [[row["name"], row["role"], row["contact"]] for row in TEAM_ROWS],
        [1.35 * inch, 3.65 * inch, 1.35 * inch],
    )
    story.append(team_table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(report["team_note"], styles["BodyCustom"]))
    story.append(Spacer(1, 0.2 * inch))

    add_section_header(story, "Abstract", styles)
    story.append(Paragraph(report["abstract"], styles["BodyCustom"]))
    story.append(PageBreak())

    add_section_header(story, "Data", styles)
    add_paragraphs(story, report["sections"][1][1], styles)

    add_section_header(story, "Methodology", styles)
    add_paragraphs(story, report["sections"][2][1], styles)

    add_section_header(story, "Results", styles)
    add_paragraphs(story, report["sections"][3][1], styles)

    story.append(Paragraph("Table 1. Wildfire model comparison", styles["Heading1Custom"]))
    wildfire_table = make_table(
        list(wildfire_rows[0].keys()),
        [list(row.values()) for row in wildfire_rows],
        [1.75 * inch, 1.1 * inch, 0.45 * inch, 0.6 * inch, 0.6 * inch, 0.45 * inch, 0.55 * inch, 0.75 * inch],
    )
    story.append(wildfire_table)
    story.append(Spacer(1, 0.18 * inch))

    if WILDFIRE_FIGURE.exists():
        wildfire_image = scale_image(WILDFIRE_FIGURE, max_width=6.7 * inch, max_height=3.4 * inch)
        story.append(
            KeepTogether(
                [
                    wildfire_image,
                    Spacer(1, 0.08 * inch),
                    Paragraph(
                        "Figure 1. Checked-in wildfire comparison figure generated from the repository artifacts.",
                        styles["CaptionCustom"],
                    ),
                ]
            )
        )
        story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Table 2. Insurance Experiment A-D summary", styles["Heading1Custom"]))
    insurance_table = make_table(
        list(insurance_rows[0].keys()),
        [list(row.values()) for row in insurance_rows],
        [0.6 * inch, 2.05 * inch, 1.15 * inch, 0.8 * inch, 0.82 * inch, 0.5 * inch, 0.8 * inch],
    )
    story.append(insurance_table)
    story.append(Spacer(1, 0.18 * inch))

    if INSURANCE_FIGURE.exists():
        insurance_image = scale_image(INSURANCE_FIGURE, max_width=6.7 * inch, max_height=3.5 * inch)
        story.append(
            KeepTogether(
                [
                    insurance_image,
                    Spacer(1, 0.08 * inch),
                    Paragraph(
                        "Figure 2. Checked-in insurance experiment comparison figure generated from the repository artifacts.",
                        styles["CaptionCustom"],
                    ),
                ]
            )
        )
        story.append(PageBreak())

    add_section_header(story, "Resource Requirements", styles)
    add_paragraphs(story, report["sections"][4][1], styles)
    story.append(Paragraph("Table 3. Quantum resource summary", styles["Heading1Custom"]))
    resource_table = make_table(
        list(resource_rows[0].keys()),
        [list(row.values()) for row in resource_rows],
        [1.9 * inch, 0.5 * inch, 0.55 * inch, 1.2 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch],
    )
    story.append(resource_table)
    story.append(Spacer(1, 0.2 * inch))

    add_section_header(story, "Envisioned Algorithm", styles)
    add_paragraphs(story, report["sections"][5][1], styles)

    add_section_header(story, "Clickable Repo Links", styles)
    for label, url, note in REPO_LINKS:
        link_text = (
            f"<link href='{url}' color='blue'>{label}</link>: {note}"
        )
        story.append(Paragraph(link_text, styles["BodyCustom"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            f"<link href='{REMOTE_ROOT}' color='blue'>{REMOTE_ROOT}</link>",
            styles["BodyCustom"],
        )
    )

    def add_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#4f6475"))
        canvas.drawString(doc_obj.leftMargin, 0.35 * inch, OUTPUT_STEM.replace("_", " "))
        canvas.drawRightString(
            doc_obj.pagesize[0] - doc_obj.rightMargin,
            0.35 * inch,
            f"Page {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)


def main() -> None:
    report = get_report_text()
    markdown = build_markdown(report)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    build_pdf(report)
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
