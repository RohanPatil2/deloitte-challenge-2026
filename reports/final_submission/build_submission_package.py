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


TEAM_METADATA = [
    ("Registered Team Name", "Insert the exact registered team name before submission"),
    ("Primary Contact", "Rohan Patil"),
    ("Submission Scope", "Phase 1 entry covering Tasks 1A, 1B, and 2 in one PDF"),
    (
        "College / University Affiliation",
        "Verify and match the exact affiliation wording used during registration",
    ),
]

TEAM_ROWS = [
    {
        "name": "Rohan Patil",
        "background": "Repository integration, reproducibility, document packaging, and delivery review",
        "contact": "rohanpatil0212@gmail.com",
        "affiliation": "Affiliation to verify before submission",
    },
    {
        "name": "Shreyas Khandale",
        "background": "Quantum machine learning, wildfire baselines, and comparative evaluation",
        "contact": "skhandale@binghamton.edu",
        "affiliation": "Binghamton University (to verify)",
    },
    {
        "name": "Aishwarya Das",
        "background": "Insurance analytics, feature engineering, experiment synthesis, and report editing",
        "contact": "adas21@binghamton.edu",
        "affiliation": "Binghamton University",
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
        This Phase 1 submission presents a hybrid wildfire-risk and insurance-premium modeling framework for Deloitte's Quantum Sustainability Challenge 2026. Task 1A is addressed through a wildfire occurrence prediction workflow at California ZIP-year granularity, Task 1B through explicit classical-versus-quantum evaluation, and Task 2 through a downstream insurance premium forecast that tests whether wildfire-risk signals improve financial modeling. The wildfire source file in the repository contains {DATA_POINTS['wildfire_raw_rows']} rows and {DATA_POINTS['wildfire_raw_cols']} columns, including {DATA_POINTS['wildfire_weather_rows']} weather rows and {DATA_POINTS['wildfire_fire_rows']} fire-event rows. These records are converted into {DATA_POINTS['wildfire_dataset_rows']} annual ZIP-year observations by aggregating weather, defining a binary fire-event target, and engineering lagged fire-history features. The insurance dataset contains {DATA_POINTS['insurance_raw_rows']} rows across {DATA_POINTS['insurance_raw_cols']} columns and is modeled using a strict 2018-2020 training window with 2021 held out for evaluation.

        For wildfire prediction, we benchmark Logistic Regression, Random Forest, Gradient Boosting, and an RBF-kernel SVM against quantum-kernel SVMs and a 4-qubit variational quantum classifier. The quantum-kernel models use mutual-information feature selection, exact-statevector FidelityStatevectorKernel evaluation, ZZFeatureMap or PauliFeatureMap circuits, and balanced SVC classification. The VQC uses a ZZFeatureMap encoder, a RealAmplitudes ansatz, COBYLA optimization, and a class-balanced weighted cross-entropy objective. In the committed results, {summary['classical_best_model']} is the strongest wildfire baseline with ROC-AUC {summary['classical_best_roc']}, while the best quantum-kernel model reaches {summary['quantum_best_roc']} and the VQC reaches {summary['vqc_roc']}. The resource profile is also explicit: the checked-in quantum runs operate at 4-6 qubits, circuit depths between 27 and 49, and total simulator runtimes between 6.85 seconds and 267.13 seconds.

        For insurance forecasting, we compare four experiments: no wildfire input, the provided dataset wildfire score, an imported classical wildfire score, and an imported quantum wildfire score. Gradient Boosting in Experiment A produces the best checked-in insurance result with R2 = {summary['insurance_best_r2']}. The wildfire-augmented experiments remain close to that baseline, indicating that wildfire-risk features are directionally useful but not yet the dominant source of predictive lift when strong lagged insurance features are already present. No external predictive datasets beyond the challenge materials were introduced. Overall, the submission demonstrates a feasible and transparent prototype: classical models currently lead predictive accuracy, while the quantum workflows contribute innovation through reproducible hybrid benchmarking, resource-aware experimentation, and a clear path toward a future 2023 wildfire-risk forecasting run.
        """
    ).strip()

    if word_count(abstract) > 400:
        raise SystemExit(
            f"Abstract word count is {word_count(abstract)}, which exceeds the 400-word cap."
        )

    team_note = (
        "Before uploading the Phase 1 PDF, replace any placeholder language in the team "
        "name or affiliation fields with the exact registration values used on the Deloitte "
        "competition website."
    )

    algorithm_paragraphs = [
        (
            "<b>Challenge coverage.</b> This document addresses the three required work items "
            "in a single PDF. <b>Task 1A</b> is handled by a wildfire-risk classification "
            "pipeline at ZIP-year level. <b>Task 1B</b> is handled by comparative evaluation "
            "between classical baselines and quantum models, including performance metrics "
            "and resource evidence. <b>Task 2</b> is handled by a time-series-style insurance "
            "premium forecasting workflow that tests whether wildfire-risk scores improve "
            "downstream premium prediction."
        ),
        (
            "<b>Data and additional data used.</b> The wildfire workstream uses the "
            "challenge-provided mixed file containing monthly weather observations and fire "
            f"event rows: {DATA_POINTS['wildfire_raw_rows']} rows, {DATA_POINTS['wildfire_raw_cols']} "
            f"columns, {DATA_POINTS['wildfire_weather_rows']} weather rows, and "
            f"{DATA_POINTS['wildfire_fire_rows']} fire rows. The insurance workstream uses "
            f"the challenge-provided ZIP-level insurance panel with "
            f"{DATA_POINTS['insurance_raw_rows']} rows and {DATA_POINTS['insurance_raw_cols']} "
            "columns. No external predictive datasets were added. Supplemental non-predictive "
            "references were limited to the challenge feature-description material and the "
            "CAL FIRE coded-value notes referenced in the challenge guide. All remaining inputs "
            "are internally derived features such as lags, ratios, aggregated weather summaries, "
            "and imported wildfire-risk scores generated from Task 1 outputs."
        ),
        (
            "<b>Concept.</b> The central concept is a hybrid risk-modeling stack. The first "
            "layer turns wildfire history and weather behavior into a per-ZIP wildfire-risk "
            "estimate. The second layer tests whether those risk signals can improve an "
            "insurance premium model. This design is useful for the challenge because it links "
            "climate risk estimation to a concrete financial application instead of treating the "
            "quantum model as an isolated benchmark."
        ),
        (
            "<b>General composition.</b> The pipeline has four layers: data cleaning, feature "
            "construction, model training, and cross-model evaluation. Data cleaning handles ZIP "
            "normalization, row-type separation, negative-premium clipping, and boolean encoding. "
            "Feature construction creates annual weather summaries, wildfire-history lags, premium "
            "lags, trend features, and exposure-normalized insurance indicators. Model training then "
            "splits into classical wildfire baselines, quantum wildfire models, and insurance "
            "regression experiments. Finally, the evaluation layer consolidates performance tables, "
            "figures, and explicit quantum resource summaries."
        ),
        (
            "<b>Task 1A wildfire algorithm.</b> Wildfire modeling begins by aggregating monthly "
            "weather rows into annual ZIP-year features: mean maximum temperature, mean minimum "
            "temperature, total precipitation, and temperature range. Fire-event rows are reduced "
            "to a binary target indicating whether at least one wildfire occurred in a ZIP-year. "
            "The algorithm then constructs lag-1 fire count, lag-2 fire count, cumulative fire "
            "count, and an ever-had-fire flag. Classical baselines include Logistic Regression, "
            "Random Forest, Gradient Boosting, and an RBF-kernel SVM. Quantum runs use "
            "mutual-information feature selection, exact-statevector quantum kernels built from "
            "ZZFeatureMap or PauliFeatureMap circuits, and a balanced SVC on the resulting kernel "
            "matrix. A 4-qubit VQC with ZZFeatureMap plus RealAmplitudes serves as the variational "
            "quantum alternative."
        ),
        (
            "<b>Task 1B evaluation.</b> The wildfire models are evaluated with a strict temporal "
            "split rather than random shuffling. In the checked-in artifact package, training uses "
            f"2018-2020 ZIP-year rows ({DATA_POINTS['wildfire_train_rows']}) and validation uses "
            f"2021 ({DATA_POINTS['wildfire_val_rows']}). Primary metrics are ROC-AUC and PR-AUC, "
            "supported by F1, precision, and recall because the wildfire-event rate is low. "
            "Task 1B also includes a resource perspective: number of qubits, circuit depth, runtime, "
            "backend choice, and trainable parameter count where relevant."
        ),
        (
            "<b>Task 2 insurance algorithm.</b> The insurance model uses the challenge ZIP-level "
            f"panel with {DATA_POINTS['insurance_train_rows']} training rows and "
            f"{DATA_POINTS['insurance_test_rows']} test rows. Preprocessing removes leakage-prone "
            "post-event fire fields and unusable weather columns, clips negative premiums, and "
            "encodes boolean columns numerically. Feature engineering then adds premium lag-1, "
            "premium lag-2, year-over-year premium change, premium percent change, lagged loss "
            "ratio, and premium-per-exposure features. Four experiments are run: Experiment A with "
            "no added wildfire-risk score, Experiment B with the provided wildfire-risk score in the "
            "dataset, Experiment C with an imported classical wildfire-risk score, and Experiment D "
            "with an imported quantum wildfire-risk score. Each experiment benchmarks a naive lag-1 "
            "baseline, Ridge, Random Forest, and Gradient Boosting regression."
        ),
        (
            "<b>Underlying assumptions.</b> The solution assumes that annual ZIP-level weather "
            "aggregation preserves enough signal for wildfire-risk classification; that prior fire "
            "activity contains meaningful predictive value; and that imported wildfire-risk scores "
            "can be treated as exogenous features inside the insurance model. It also assumes that "
            "temporal splits are more appropriate than random splits for both tasks, because the "
            "submission is intended to emulate forward prediction rather than retrospective curve fit."
        ),
        (
            "<b>Scope note for challenge alignment.</b> The repository-backed evidence in this "
            "submission validates the wildfire pipeline on the latest weather horizon available in "
            "the checked-in raw file, which ends at 2021. The algorithm design is compatible with "
            "the challenge's requested 2018-2022 to 2023 framing, but a final 2023 production run "
            "would require the corresponding sponsor-aligned weather horizon or an updated wildfire "
            "input release."
        ),
    ]

    results_paragraphs = [
        (
            f"<b>Task 1A and Task 1B wildfire results.</b> Table 1 and Figure 1 summarize the "
            f"wildfire experiments. {summary['classical_best_model']} is the strongest checked-in "
            f"classical model with ROC-AUC = {summary['classical_best_roc']}. The best "
            f"quantum-kernel run is {summary['quantum_best_model']} with ROC-AUC = "
            f"{summary['quantum_best_roc']}, while the VQC reaches {summary['vqc_roc']}. The "
            "Pauli 4-qubit kernel performs better than the ZZ 4-qubit kernel, which indicates that "
            "feature-map choice matters even in a compact qubit regime."
        ),
        (
            "<b>Evaluation interpretation.</b> The wildfire comparison should be read with its "
            "tractability constraints in mind. The strongest classical baseline uses the full "
            "7,779-example training set, while the quantum models are capped at 800 examples because "
            "exact kernel construction scales quadratically. Even so, the Task 1B evidence is still "
            "useful: it shows that the quantum workflows are operational, measurable, and sensitive "
            "to qubit count, circuit family, and runtime budget."
        ),
        (
            "<b>Task 2 results.</b> Table 2 and Figure 2 summarize the insurance experiments. "
            "Across all four experiments, Gradient Boosting is the strongest model family. The best "
            "overall checked-in result is Experiment A, which excludes external wildfire-risk input, "
            "with R2 = 0.9794. Experiments B, C, and D remain close to that baseline, suggesting "
            "that wildfire-risk scores are plausible auxiliary signals but do not yet dominate the "
            "premium forecast once strong lagged insurance features are available."
        ),
        (
            "<b>Resource evidence.</b> Table 3 should be read as part of the Task 1B result set. "
            "The committed quantum runs use a statevector simulator, 4-6 qubits, circuit depths "
            "between 27 and 49, and total runtimes between 6.85 seconds and 267.13 seconds. This "
            "resource framing strengthens the feasibility argument because it documents not just "
            "accuracy, but also cost, depth, and optimization burden."
        ),
        (
            "<b>Innovation, feasibility, and community impact.</b> The submission's innovation "
            "lies in connecting wildfire-risk prediction to an insurance use case, while still "
            "providing transparent classical baselines. Its feasibility is supported by reproducible "
            "scripts, tractable simulator runtimes for the kernel models, and direct artifact links. "
            "Its quantum community impact comes from providing an open benchmark narrative for where "
            "quantum methods are currently promising, where they are still limited, and how hybrid "
            "workflows can be evaluated responsibly."
        ),
    ]

    envisioned_paragraphs = [
        (
            "<b>Envisioned next-stage algorithm.</b> The natural next step is a fully challenge-"
            "aligned 2023 wildfire-risk forecasting workflow trained on 2018-2022 history. That "
            "production version should preserve the strongest aspects of the current prototype: "
            "row-type separation, annual weather aggregation, lagged fire-history features, strict "
            "temporal validation, and comparative benchmarking against strong classical baselines."
        ),
        (
            "<b>Expected benefits.</b> A mature version of this solution could improve decision-making "
            "in two ways. First, better wildfire-risk estimates can support more targeted geographic "
            "risk monitoring. Second, calibrated wildfire-risk scores can be fed into downstream "
            "insurance models, scenario analysis, or portfolio stress testing. The quantum component "
            "is most compelling in a hybrid architecture, where it complements rather than replaces "
            "the strongest classical screen."
        ),
        (
            "<b>Requirements for the envisioned solution.</b> A stronger follow-on version would need "
            "the sponsor-aligned 2022-2023 wildfire-weather horizon, calibrated probability output, "
            "hardware-aware or shot-based quantum experiments in addition to statevector simulations, "
            "and a clear strategy for scaling beyond the 800-sample quantum training cap. It would also "
            "benefit from shallower circuit exploration, probability calibration, and explicit uncertainty "
            "communication so that the output is more useful to insurance and climate-risk stakeholders."
        ),
        (
            "<b>Why this matters.</b> Even in its current form, the repository provides a credible "
            "prototype foundation for the quantum community: reproducible comparison tables, explicit "
            "resource reporting, and an application that links QML to a real sustainability and finance "
            "problem. The envisioned algorithm extends that foundation into a more operational hybrid "
            "forecasting system with clearer practical value."
        ),
    ]

    return {
        "title": "Deloitte Quantum Sustainability Challenge 2026",
        "subtitle": "Phase 1 Submission Draft: Tasks 1A, 1B, and 2",
        "prepared": f"Prepared {date.today().strftime('%B %d, %Y')}",
        "team_note": team_note,
        "abstract": abstract,
        "sections": [
            ("Overview of the Individual or Team and Backgrounds", []),
            ("Detailed Description of the Participant's Algorithm", algorithm_paragraphs),
            ("Description of Results", results_paragraphs),
            ("Description of the Envisioned Algorithm", envisioned_paragraphs),
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
    lines.append("## Overview of the Individual or Team and Backgrounds")
    lines.append("")
    lines.append(
        markdown_table(
            ["Field", "Value"],
            [[field, value] for field, value in TEAM_METADATA],
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            ["Name", "Background", "E-mail", "College / University Affiliation"],
            [
                [row["name"], row["background"], row["contact"], row["affiliation"]]
                for row in TEAM_ROWS
            ],
        )
    )
    lines.append("")
    lines.append(report["team_note"])
    lines.append("")
    lines.append("## One-Page Summary / Abstract")
    lines.append("")
    lines.append(report["abstract"])
    lines.append("")
    lines.append("## Detailed Description of the Participant's Algorithm")
    lines.append("")
    for paragraph in report["sections"][1][1]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Description of Results")
    lines.append("")
    for paragraph in report["sections"][2][1]:
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
    lines.append(
        "Table note: classical wildfire rows use the full 7,779-example training set, while "
        "the checked-in quantum rows use 800 training examples because of exact-kernel tractability."
    )
    lines.append("")
    lines.append(f"![Wildfire comparison figure]({WILDFIRE_FIGURE.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append(
        "Figure 1 note: the chart visually reinforces that the current prototype's strongest "
        "classical baseline outperforms the checked-in quantum runs on ROC-AUC, while the quantum "
        "artifacts still provide meaningful resource and architecture comparisons."
    )
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
    lines.append(
        "Table note: each row reports the best-performing model family inside that experiment, "
        "not every model that was tested."
    )
    lines.append("")
    lines.append(f"![Insurance comparison figure]({INSURANCE_FIGURE.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append(
        "Figure 2 note: the insurance figure shows that all wildfire-augmented experiments remain "
        "close to the no-wildfire baseline, with Gradient Boosting dominating the checked-in runs."
    )
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
    lines.append(
        "Table note: Table 3 is part of the Task 1B evaluation evidence because it captures "
        "runtime, qubit count, backend, and circuit-depth implications."
    )
    lines.append("")
    lines.append("## Description of the Envisioned Algorithm")
    lines.append("")
    for paragraph in report["sections"][3][1]:
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
            name="Heading2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#36566f"),
            spaceAfter=6,
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

    add_section_header(story, "Overview of the Individual or Team and Backgrounds", styles)
    team_meta_table = make_table(
        ["Field", "Value"],
        [[field, value] for field, value in TEAM_METADATA],
        [1.85 * inch, 4.85 * inch],
    )
    story.append(team_meta_table)
    story.append(Spacer(1, 0.12 * inch))
    team_table = make_table(
        ["Name", "Background", "E-mail", "College / University Affiliation"],
        [
            [row["name"], row["background"], row["contact"], row["affiliation"]]
            for row in TEAM_ROWS
        ],
        [1.1 * inch, 2.95 * inch, 1.45 * inch, 1.2 * inch],
    )
    story.append(team_table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(report["team_note"], styles["BodyCustom"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())

    add_section_header(story, "One-Page Summary / Abstract", styles)
    story.append(Paragraph(report["abstract"], styles["BodyCustom"]))
    story.append(PageBreak())

    add_section_header(story, "Detailed Description of the Participant's Algorithm", styles)
    add_paragraphs(story, report["sections"][1][1], styles)

    add_section_header(story, "Description of Results", styles)
    add_paragraphs(story, report["sections"][2][1], styles)

    story.append(Paragraph("Table 1. Wildfire model comparison", styles["Heading1Custom"]))
    wildfire_table = make_table(
        list(wildfire_rows[0].keys()),
        [list(row.values()) for row in wildfire_rows],
        [1.75 * inch, 1.1 * inch, 0.45 * inch, 0.6 * inch, 0.6 * inch, 0.45 * inch, 0.55 * inch, 0.75 * inch],
    )
    story.append(wildfire_table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Table note: classical wildfire rows use the full 7,779-example training set, "
            "while the checked-in quantum rows use 800 training examples because of exact-kernel tractability.",
            styles["BodyCustom"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))

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
        story.append(
            Paragraph(
                "Figure 1 note: the chart reinforces that the current classical baseline outperforms "
                "the checked-in quantum runs on ROC-AUC, while the quantum models still contribute "
                "useful architecture and resource comparisons.",
                styles["BodyCustom"],
            )
        )
        story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Table 2. Insurance Experiment A-D summary", styles["Heading1Custom"]))
    insurance_table = make_table(
        list(insurance_rows[0].keys()),
        [list(row.values()) for row in insurance_rows],
        [0.6 * inch, 2.05 * inch, 1.15 * inch, 0.8 * inch, 0.82 * inch, 0.5 * inch, 0.8 * inch],
    )
    story.append(insurance_table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Table note: each row reports the best-performing model family within that experiment, "
            "not every candidate model tested.",
            styles["BodyCustom"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))

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
        story.append(Spacer(1, 0.12 * inch))
        story.append(
            Paragraph(
                "Figure 2 note: the insurance chart shows that wildfire-augmented experiments stay "
                "close to the no-wildfire baseline, with Gradient Boosting dominating the checked-in runs.",
                styles["BodyCustom"],
            )
        )
        story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Table 3. Quantum resource summary", styles["Heading1Custom"]))
    resource_table = make_table(
        list(resource_rows[0].keys()),
        [list(row.values()) for row in resource_rows],
        [1.9 * inch, 0.5 * inch, 0.55 * inch, 1.2 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch],
    )
    story.append(resource_table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "Table note: Table 3 is part of the Task 1B evidence because it documents qubit count, "
            "circuit depth, backend choice, and runtime alongside predictive performance.",
            styles["BodyCustom"],
        )
    )
    story.append(PageBreak())

    add_section_header(story, "Description of the Envisioned Algorithm", styles)
    add_paragraphs(story, report["sections"][3][1], styles)

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
