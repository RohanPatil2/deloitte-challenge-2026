"""
Quantum Resource Table Builder
================================
Reads quantum result JSON files from results/quantum_runs/ and produces a
hardware/resource-focused summary at reports/tables/quantum_resources.csv.

Rows  : Quantum Kernel SVM (4q), Quantum Kernel SVM (6q), VQC (4q)
Source:
  • Q-Kernel rows – most recent non-VQC run containing quantum_4qubit /
                    quantum_6qubit keys
  • VQC row       – most recent *_vqc.json

Columns:
  Model, Qubits, Circuit Depth, Feature Map, Shots, Backend,
  Train Samples, Kernel Build Time (s), Total Runtime (s),
  Trainable Parameters, Best ROC-AUC
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
RUNS_DIR    = Path("results/quantum_runs")
REPORTS_DIR = Path("reports/tables")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = REPORTS_DIR / "quantum_resources.csv"

COLUMNS = [
    "Model",
    "Qubits",
    "Circuit Depth",
    "Feature Map",
    "Shots",
    "Backend",
    "Train Samples",
    "Kernel Build Time (s)",
    "Total Runtime (s)",
    "Trainable Parameters",
    "Best ROC-AUC",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def pick_latest(files, predicate=None):
    """Most-recently-modified file optionally matching a predicate."""
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
# 1. QUANTUM KERNEL SVM  (4-qubit and 6-qubit)
# ══════════════════════════════════════════════════════════════════════════════
print("Reading quantum kernel SVM results …")

qk_files = [f for f in RUNS_DIR.glob("*.json") if "vqc" not in f.name]
qk_path  = pick_latest(
    qk_files,
    predicate=lambda d: "quantum_4qubit" in d and "quantum_6qubit" in d,
)

if qk_path is None:
    print("ERROR: no 4v6-qubit kernel run found in results/quantum_runs/.",
          file=sys.stderr)
    sys.exit(1)

qk_data = load_json(qk_path)
sc      = qk_data["shared_config"]       # zz_reps, shots, backend, …
samples = qk_data["data"]["train_samples"]

rows = []

for qubit_key, display_name in [
    ("quantum_4qubit", "Quantum Kernel SVM (4q)"),
    ("quantum_6qubit", "Quantum Kernel SVM (6q)"),
]:
    q            = qk_data[qubit_key]
    n_q          = q["num_qubits"]
    depth        = q["circuit_depth"]
    k_time       = round(q["kernel_runtime_s"], 4)
    svm_time     = round(q["svm_train_runtime_s"], 4)
    total        = round(k_time + svm_time, 4)
    feature_map  = f"ZZFeatureMap (reps={sc['zz_reps']})"
    shots        = sc["shots"]
    backend      = sc["backend"]
    roc_auc      = q["roc_auc"]

    rows.append({
        "Model"                 : display_name,
        "Qubits"                : n_q,
        "Circuit Depth"         : depth,
        "Feature Map"           : feature_map,
        "Shots"                 : shots,
        "Backend"               : backend,
        "Train Samples"         : samples,
        "Kernel Build Time (s)" : k_time,
        "Total Runtime (s)"     : total,
        "Trainable Parameters"  : "-",
        "Best ROC-AUC"          : roc_auc,
    })

print(f"  Loaded Q-Kernel 4q + 6q  ← {qk_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. VQC
# ══════════════════════════════════════════════════════════════════════════════
print("Reading VQC results …")

vqc_path = pick_latest(list(RUNS_DIR.glob("*_vqc.json")))
if vqc_path is None:
    print("ERROR: no VQC run found in results/quantum_runs/.", file=sys.stderr)
    sys.exit(1)

vd      = load_json(vqc_path)
cfg     = vd["vqc_config"]
res     = vd["vqc_results"]

# Feature map label combines both encoding and variational circuits
feature_map_label = (
    f"{cfg['feature_map']} + {cfg['ansatz']}"
)

# Parse shots from the sampler description
# e.g. "StatevectorSampler (default_shots=1024)"
sampler_str = cfg["sampler"]
shots_label = sampler_str   # keep full string for transparency

rows.append({
    "Model"                 : "VQC (4q)",
    "Qubits"                : cfg["num_qubits"],
    "Circuit Depth"         : cfg["full_circuit_depth"],
    "Feature Map"           : feature_map_label,
    "Shots"                 : shots_label,
    "Backend"               : "statevector_simulator",
    "Train Samples"         : vd["data"]["train_samples"],
    "Kernel Build Time (s)" : "-",          # VQC has no kernel matrix step
    "Total Runtime (s)"     : cfg["train_runtime_s"],
    "Trainable Parameters"  : cfg["num_parameters"],
    "Best ROC-AUC"          : res["roc_auc"],
})

print(f"  Loaded VQC                ← {vqc_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD DATAFRAME  &  SAVE CSV
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(rows, columns=COLUMNS)
df.to_csv(OUT_CSV, index=False)
print(f"\nCSV saved → {OUT_CSV}  ({len(df)} rows × {len(df.columns)} columns)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TERMINAL PRINT
# ══════════════════════════════════════════════════════════════════════════════
# Split into two sub-tables for readability: circuit info and runtime info

SEP = "─" * 108

print()
print("=" * 108)
print("QUANTUM RESOURCE SUMMARY  —  Wildfire Prediction (Val year: 2021)")
print("=" * 108)

# ── Sub-table A: Circuit / encoding properties ────────────────────────────────
print()
print("  Circuit & Encoding Properties")
print()
print(f"  {'Model':<26} {'Q':>5} {'Depth':>7} {'Params':>8}  {'Feature Map / Ansatz'}")
print(f"  {SEP[:95]}")

for _, r in df.iterrows():
    print(
        f"  {r['Model']:<26} "
        f"{str(r['Qubits']):>5} "
        f"{str(r['Circuit Depth']):>7} "
        f"{str(r['Trainable Parameters']):>8}  "
        f"{r['Feature Map']}"
    )

# ── Sub-table B: Execution / runtime properties ───────────────────────────────
print()
print("  Execution & Runtime Properties")
print()
print(
    f"  {'Model':<26} "
    f"{'Train Samples':>14} "
    f"{'Kernel Build (s)':>17} "
    f"{'Total Runtime (s)':>18} "
    f"{'ROC-AUC':>9}"
)
print(f"  {SEP[:95]}")

for _, r in df.iterrows():
    kbt = (f"{r['Kernel Build Time (s)']:>17.3f}"
           if r["Kernel Build Time (s)"] != "-"
           else f"{'- (no kernel)':>17}")
    print(
        f"  {r['Model']:<26} "
        f"{r['Train Samples']:>14,} "
        f"{kbt} "
        f"{r['Total Runtime (s)']:>18.3f} "
        f"{r['Best ROC-AUC']:>9.4f}"
    )

# ── Sub-table C: Shots & backend ─────────────────────────────────────────────
print()
print("  Shots & Backend")
print()
print(f"  {'Model':<26} {'Backend':<24} {'Shots / Sampler'}")
print(f"  {SEP[:95]}")

for _, r in df.iterrows():
    print(f"  {r['Model']:<26} {r['Backend']:<24} {r['Shots']}")

# ── Notes ─────────────────────────────────────────────────────────────────────
print()
print(SEP)
print()
print("  Notes:")
print("  • Circuit depth shown after full gate decomposition.")
print("  • Quantum Kernel: runtime = kernel matrix build + SVC fit.")
print("    K-matrix shape: (800, 800) train + (2593, 800) val.")
print("  • VQC: runtime = COBYLA optimisation over 100 iterations × 800 samples.")
print("    'Kernel Build Time' is not applicable — VQC has no pre-built kernel matrix.")
print("  • Both approaches use StatevectorSampler / statevector backend (exact).")
print("  • Trainable Parameters: VQC only. RealAmplitudes (reps=3, linear) → 4×4=16.")
print(f"\n  Full resource table → {OUT_CSV}")
