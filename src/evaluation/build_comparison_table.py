"""
Model Comparison Table Builder
================================
Reads all result JSON files from:
  • results/quantum_runs/       – quantum kernel SVM runs and VQC run
  • results/quantum_metrics/    – classical baseline results

Selects the canonical record for each of the 7 models:
  1. Logistic Regression          – classical_baselines.json
  2. Random Forest                – classical_baselines.json
  3. Gradient Boosting            – classical_baselines.json
  4. Classical SVM (RBF)          – classical_baselines.json
  5. Quantum Kernel SVM (4-qubit) – most recent quantum_runs JSON that
                                    contains a quantum_4qubit key
  6. Quantum Kernel SVM (6-qubit) – same file, quantum_6qubit key
  7. VQC (4-qubit)                – most recent *_vqc.json

Output:
  reports/tables/model_comparison.csv   (columns defined below)
  Terminal table (formatted)
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
RUNS_DIR    = Path("results/quantum_runs")
METRICS_DIR = Path("results/quantum_metrics")
REPORTS_DIR = Path("reports/tables")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = REPORTS_DIR / "model_comparison.csv"

# ── Column order for the output table ─────────────────────────────────────────
COLUMNS = [
    "Model", "Type", "Qubits", "Features",
    "ROC-AUC", "F1", "Precision", "Recall",
    "Runtime_s", "Train_samples", "Notes",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_features(feat_list: list[str]) -> str:
    """Return a short human-readable feature string for the CSV cell."""
    abbrev = {
        "avg_tmax_c"           : "tmax",
        "avg_tmin_c"           : "tmin",
        "tot_prcp_mm"          : "prcp",
        "temp_range"           : "range",
        "fire_count_lag1"      : "lag1",
        "fire_count_lag2"      : "lag2",
        "cumulative_fire_count": "cum_fire",
        "ever_had_fire"        : "ever_fire",
    }
    return "; ".join(abbrev.get(f, f) for f in feat_list)


def pick_latest(files, predicate=None):
    """Return the most-recently-modified file matching an optional predicate."""
    candidates = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
    if predicate is None:
        return candidates[0] if candidates else None
    for f in candidates:
        try:
            if predicate(load_json(f)):
                return f
        except Exception:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASSICAL BASELINES
# ══════════════════════════════════════════════════════════════════════════════
print("Reading classical_baselines.json …")

cl_path = METRICS_DIR / "classical_baselines.json"
if not cl_path.exists():
    print(f"ERROR: {cl_path} not found. Run src/evaluation/classical_baselines.py first.",
          file=sys.stderr)
    sys.exit(1)

cl_data    = load_json(cl_path)
cl_samples = cl_data["data"]["train_samples"]
cl_feats   = cl_data["features"]["all_features_used"]  # list of 8

CLASSICAL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest"      : "Random Forest",
    "gradient_boosting"  : "Gradient Boosting",
    "svm_rbf"            : "Classical SVM (RBF)",
}

CLASSICAL_NOTES = {
    "logistic_regression": "max_iter=1000; class_weight=balanced",
    "random_forest"      : "n_estimators=300; class_weight=balanced",
    "gradient_boosting"  : "n_estimators=300; lr=0.05; balanced via sample_weight",
    "svm_rbf"            : "RBF kernel; class_weight=balanced",
}

rows = []

for key, display_name in CLASSICAL_DISPLAY.items():
    m = cl_data["models"][key]
    rows.append({
        "Model"         : display_name,
        "Type"          : "Classical",
        "Qubits"        : "-",
        "Features"      : fmt_features(cl_feats),
        "ROC-AUC"       : m["roc_auc"],
        "F1"            : m["f1"],
        "Precision"     : m["precision"],
        "Recall"        : m["recall"],
        "Runtime_s"     : m["train_runtime_s"],
        "Train_samples" : cl_samples,
        "Notes"         : CLASSICAL_NOTES[key],
    })

print(f"  Loaded {len(rows)} classical models "
      f"(train_samples={cl_samples:,})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM KERNEL SVM  (4-qubit and 6-qubit)
#    Canonical source: most recent non-VQC run that has quantum_4qubit +
#    quantum_6qubit keys (i.e., the dual-experiment refactor run).
# ══════════════════════════════════════════════════════════════════════════════
print("Reading quantum kernel SVM results …")

qk_files = [f for f in RUNS_DIR.glob("*.json") if "vqc" not in f.name]
qk_path  = pick_latest(
    qk_files,
    predicate=lambda d: "quantum_4qubit" in d and "quantum_6qubit" in d,
)

if qk_path is None:
    print("ERROR: no quantum kernel 4v6-qubit run found in results/quantum_runs/.",
          file=sys.stderr)
    sys.exit(1)

qk_data    = load_json(qk_path)
qk_samples = qk_data["data"]["train_samples"]
qk_cfg     = qk_data["shared_config"]

for qubit_key, display_name in [
    ("quantum_4qubit", "Quantum Kernel SVM (4q)"),
    ("quantum_6qubit", "Quantum Kernel SVM (6q)"),
]:
    q = qk_data[qubit_key]
    n_q       = q["num_qubits"]
    feats     = q["selected_features"]
    runtime   = round(q["kernel_runtime_s"] + q["svm_train_runtime_s"], 4)
    depth     = q["circuit_depth"]

    rows.append({
        "Model"         : display_name,
        "Type"          : "Quantum Kernel",
        "Qubits"        : n_q,
        "Features"      : fmt_features(feats),
        "ROC-AUC"       : q["roc_auc"],
        "F1"            : q["f1"],
        "Precision"     : q["precision"],
        "Recall"        : q["recall"],
        "Runtime_s"     : runtime,
        "Train_samples" : qk_samples,
        "Notes"         : (
            f"FidelityStatevectorKernel; ZZFeatureMap reps=2; "
            f"circuit_depth={depth}; class_weight=balanced"
        ),
    })

print(f"  Loaded Q-Kernel 4q + 6q from {qk_path.name} "
      f"(train_samples={qk_samples:,})")


# ══════════════════════════════════════════════════════════════════════════════
# 3. VQC
# ══════════════════════════════════════════════════════════════════════════════
print("Reading VQC results …")

vqc_files = list(RUNS_DIR.glob("*_vqc.json"))
vqc_path  = pick_latest(vqc_files)

if vqc_path is None:
    print("ERROR: no VQC run found in results/quantum_runs/.", file=sys.stderr)
    sys.exit(1)

vqc_data    = load_json(vqc_path)
vqc_cfg     = vqc_data["vqc_config"]
vqc_feats   = vqc_data["feature_engineering"]["selected_features"]
vqc_samples = vqc_data["data"]["train_samples"]
vqc_res     = vqc_data["vqc_results"]

rows.append({
    "Model"         : "VQC (4q)",
    "Type"          : "Quantum VQC",
    "Qubits"        : vqc_cfg["num_qubits"],
    "Features"      : fmt_features(vqc_feats),
    "ROC-AUC"       : vqc_res["roc_auc"],
    "F1"            : vqc_res["f1"],
    "Precision"     : vqc_res["precision"],
    "Recall"        : vqc_res["recall"],
    "Runtime_s"     : vqc_cfg["train_runtime_s"],
    "Train_samples" : vqc_samples,
    "Notes"         : (
        f"{vqc_cfg['feature_map']}; {vqc_cfg['ansatz']}; "
        f"depth={vqc_cfg['full_circuit_depth']}; "
        f"{vqc_cfg['num_parameters']} params; "
        f"COBYLA iters={vqc_cfg['iterations_run']}; "
        f"WeightedCrossEntropyLoss"
    ),
})

print(f"  Loaded VQC from {vqc_path.name} "
      f"(train_samples={vqc_samples:,})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD DATAFRAME  &  SAVE CSV
# ══════════════════════════════════════════════════════════════════════════════
# Enforce canonical row order
MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "Classical SVM (RBF)",
    "Quantum Kernel SVM (4q)",
    "Quantum Kernel SVM (6q)",
    "VQC (4q)",
]

df = pd.DataFrame(rows, columns=COLUMNS)
df["_order"] = df["Model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

# Round numeric columns
for col in ["ROC-AUC", "F1", "Precision", "Recall"]:
    df[col] = df[col].round(4)
df["Runtime_s"] = df["Runtime_s"].round(4)

df.to_csv(OUT_CSV, index=False)
print(f"\nCSV saved → {OUT_CSV}  ({len(df)} rows × {len(df.columns)} columns)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TERMINAL PRINT
# ══════════════════════════════════════════════════════════════════════════════
# Column widths for the display table (excludes Notes and Features — shown below)
W_MODEL   = 26
W_TYPE    = 16
W_QUBITS  =  6
W_METRIC  =  9
W_RUNTIME = 10
W_SAMPLES = 14

DIVIDER = "─" * (W_MODEL + W_TYPE + W_QUBITS + W_METRIC * 4 + W_RUNTIME + W_SAMPLES + 8)

print()
print("=" * len(DIVIDER))
print("MODEL COMPARISON — WILDFIRE FIRE-EVENT PREDICTION  (Val year: 2021)")
print("=" * len(DIVIDER))
print()

header = (
    f"{'Model':<{W_MODEL}} "
    f"{'Type':<{W_TYPE}} "
    f"{'Q':>{W_QUBITS}} "
    f"{'ROC-AUC':>{W_METRIC}} "
    f"{'F1':>{W_METRIC}} "
    f"{'Precision':>{W_METRIC}} "
    f"{'Recall':>{W_METRIC}} "
    f"{'Runtime(s)':>{W_RUNTIME}} "
    f"{'Train Samples':>{W_SAMPLES}}"
)
print(header)
print(DIVIDER)

prev_type = None
for _, row in df.iterrows():
    # Blank separator line between model families
    if prev_type is not None and row["Type"] != prev_type:
        print()
    prev_type = row["Type"]

    print(
        f"{row['Model']:<{W_MODEL}} "
        f"{row['Type']:<{W_TYPE}} "
        f"{str(row['Qubits']):>{W_QUBITS}} "
        f"{row['ROC-AUC']:>{W_METRIC}.4f} "
        f"{row['F1']:>{W_METRIC}.4f} "
        f"{row['Precision']:>{W_METRIC}.4f} "
        f"{row['Recall']:>{W_METRIC}.4f} "
        f"{row['Runtime_s']:>{W_RUNTIME}.3f} "
        f"{row['Train_samples']:>{W_SAMPLES},}"
    )

print()
print(DIVIDER)

# ── Feature sets used ─────────────────────────────────────────────────────────
print()
print("Feature sets:")
seen_feats: set[str] = set()
for _, row in df.iterrows():
    key = f"{row['Model']}: {row['Features']}"
    if row["Features"] not in seen_feats:
        print(f"  {row['Model']:<26}  {row['Features']}")
        seen_feats.add(row["Features"])

print()
print("Abbreviations: tmax=avg_tmax_c  tmin=avg_tmin_c  prcp=tot_prcp_mm  "
      "range=temp_range")
print("               lag1=fire_count_lag1  lag2=fire_count_lag2  "
      "cum_fire=cumulative_fire_count  ever_fire=ever_had_fire")

# ── Best per metric ───────────────────────────────────────────────────────────
print()
print("Best per metric:")
for metric in ["ROC-AUC", "F1", "Precision", "Recall"]:
    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]
    print(f"  {metric:<12} → {best_row['Model']:<26} ({best_row[metric]:.4f})")

print()
print(f"Full results saved to: {OUT_CSV}")
