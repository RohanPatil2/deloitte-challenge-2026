"""
Quantum Kernel SVM Pipeline for Wildfire Prediction
====================================================
Dataset  : wildfire_weather.csv
Target   : Binary – did a zip code have any fire event in a given year?
Grain    : zip × year (2018–2021, weather years only)

Two logical row types in the raw file:
  • Weather rows  (OBJECTID is NaN) – one row per zip per month, with
                                       avg_tmax_c / avg_tmin_c / tot_prcp_mm
  • Fire rows     (OBJECTID not NaN) – one row per fire event, with zip & Year

Pipeline steps
--------------
1. Load & separate the two row types
2. Aggregate weather to zip × year (mean temps, annual precip sum)
3. Aggregate fire events to zip × year → binary target
4. Engineer fire-history lag features (lag-1, lag-2, cumulative, ever_had_fire)
5. Train / val split  (train: 2018-2020 | val: 2021)
6. Feature selection (top 6 via mutual information → 6 qubits)
7. StandardScaler normalisation
8. Quantum Kernel SVM  – ZZFeatureMap + FidelityStatevectorKernel + SVC (balanced)
9. Classical SVM baseline – RBF kernel SVC (balanced)
10. Evaluate: ROC-AUC, F1, precision, recall
11. Log everything to results/quantum_runs/<timestamp>.json

Quantum note: FidelityStatevectorKernel computes the exact inner-product
kernel using statevector simulation, which is equivalent to
FidelityQuantumKernel evaluated with shots → ∞.  It is the recommended
high-fidelity variant in qiskit-machine-learning ≥ 0.7.

Reproducibility: RANDOM_SEED = 42 throughout.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
N_QUBITS         = 6        # features fed to the quantum kernel = number of qubits
ZZ_REPS          = 2        # ZZFeatureMap circuit repetitions
TRAIN_YEARS      = [2018, 2019, 2020]
VAL_YEARS        = [2021]
# Quantum kernel matrices are O(n²); cap training samples to keep runtime
# tractable on a classical simulator.  Set to None to use all data.
MAX_TRAIN_SAMPLES = 400

RAW_DIR     = Path("data/raw")
RESULTS_DIR = Path("results/quantum_runs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & SEPARATE ROW TYPES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("QUANTUM KERNEL SVM – WILDFIRE PREDICTION")
print("=" * 65)
print("\n[1] Loading wildfire_weather.csv …")

df = pd.read_csv(
    RAW_DIR / "wildfire_weather.csv",
    encoding="latin-1",
    low_memory=False,
)
print(f"    Raw rows loaded : {len(df):,}")

# Normalise zip to a clean integer string ("90001") in both row types
df["zip"] = pd.to_numeric(df["zip"], errors="coerce")
df = df.dropna(subset=["zip"])
df["zip"] = df["zip"].astype(int).astype(str)

# ── Weather rows: OBJECTID is NaN; year comes from year_month (YYYY-MM) ──────
weather = df[df["OBJECTID"].isna()].copy()
weather["year"] = weather["year_month"].str[:4].astype(float).astype("Int64")
weather = weather.dropna(subset=["year", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"])
weather = weather[weather["year"].isin([2018, 2019, 2020, 2021])]

# ── Fire rows: OBJECTID not NaN; year comes from the Year column ───────────
fire = df[df["OBJECTID"].notna()].copy()
fire["Year"] = pd.to_numeric(fire["Year"], errors="coerce")
fire = fire.dropna(subset=["zip", "Year"])
fire["Year"] = fire["Year"].astype(int)

print(f"    Weather rows (2018-2021): {len(weather):,}")
print(f"    Fire rows   (all years) : {len(fire):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGGREGATE WEATHER TO ZIP × YEAR
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Aggregating weather to zip × year …")

# Monthly rows → annual summary per zip
weather_agg = (
    weather
    .groupby(["zip", "year"])
    .agg(
        avg_tmax_c  = ("avg_tmax_c",  "mean"),   # mean monthly max temp (°C)
        avg_tmin_c  = ("avg_tmin_c",  "mean"),   # mean monthly min temp (°C)
        tot_prcp_mm = ("tot_prcp_mm", "sum"),     # annual total precip (mm)
    )
    .reset_index()
    .rename(columns={"year": "Year"})
)
weather_agg["Year"] = weather_agg["Year"].astype(int)
print(f"    Zip-year weather records : {len(weather_agg):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CREATE BINARY TARGET (fire event in that zip-year)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Building binary fire-event targets …")

# For each zip, which years had at least one fire?
fire_zip_year = (
    fire[fire["Year"].isin([2018, 2019, 2020, 2021])]
    .groupby(["zip", "Year"])
    .size()
    .reset_index(name="fire_count_target")
)
fire_zip_year["fire_event"] = 1

# The universe of valid zip-years is defined by weather coverage
dataset = weather_agg.merge(
    fire_zip_year[["zip", "Year", "fire_event"]],
    on=["zip", "Year"],
    how="left",
)
dataset["fire_event"] = dataset["fire_event"].fillna(0).astype(int)

print(f"    Total zip-year records : {len(dataset):,}")
print(f"    Fire event (target=1)  : {dataset['fire_event'].sum():,}  "
      f"({dataset['fire_event'].mean()*100:.1f}%)")
print(f"    No fire    (target=0)  : {(dataset['fire_event']==0).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIRE-HISTORY LAG FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Engineering fire-history lag features …")

# Use ALL years of fire data (not just 2018-2021) for richer lag signal.
# Aggregate fire counts per zip per year across the full historical record.
hist_counts = (
    fire.groupby(["zip", "Year"])
    .size()
    .reset_index(name="fire_count")
)

# Build a complete grid: every (zip in dataset) × every relevant year.
# We need 2016 to compute lag-2 for 2018, and 2017 for lag-1 of 2018.
all_years_for_lag = sorted(
    set([2016, 2017] + TRAIN_YEARS + VAL_YEARS)
)
target_zips = dataset["zip"].unique()

grid = (
    pd.MultiIndex.from_product(
        [target_zips, all_years_for_lag], names=["zip", "Year"]
    )
    .to_frame(index=False)
)
grid = grid.merge(hist_counts, on=["zip", "Year"], how="left")
grid["fire_count"] = grid["fire_count"].fillna(0)
grid = grid.sort_values(["zip", "Year"]).reset_index(drop=True)

# fire_count_lag1: fires in the immediately preceding year (t-1)
grid["fire_count_lag1"] = (
    grid.groupby("zip")["fire_count"]
    .transform(lambda s: s.shift(1).fillna(0))
)

# fire_count_lag2: fires two years prior (t-2)
grid["fire_count_lag2"] = (
    grid.groupby("zip")["fire_count"]
    .transform(lambda s: s.shift(2).fillna(0))
)

# cumulative_fire_count: total fires in ALL years BEFORE the current year
# (shift-then-cumsum gives the exclusive running total)
grid["cumulative_fire_count"] = (
    grid.groupby("zip")["fire_count"]
    .transform(lambda s: s.cumsum().shift(1).fillna(0))
)

# ever_had_fire: binary flag — did this zip have any recorded fire before this year?
grid["ever_had_fire"] = (grid["cumulative_fire_count"] > 0).astype(int)

lag_features = grid[grid["Year"].isin(TRAIN_YEARS + VAL_YEARS)][
    ["zip", "Year", "fire_count_lag1", "fire_count_lag2",
     "cumulative_fire_count", "ever_had_fire"]
]

dataset = dataset.merge(lag_features, on=["zip", "Year"], how="left")
for col in ["fire_count_lag1", "fire_count_lag2", "cumulative_fire_count", "ever_had_fire"]:
    dataset[col] = dataset[col].fillna(0)

# Derived feature: annual temperature range (max – min)
dataset["temp_range"] = dataset["avg_tmax_c"] - dataset["avg_tmin_c"]

print(f"    fire_count_lag1       – mean: {dataset['fire_count_lag1'].mean():.3f}")
print(f"    fire_count_lag2       – mean: {dataset['fire_count_lag2'].mean():.3f}")
print(f"    cumulative_fire_count – mean: {dataset['cumulative_fire_count'].mean():.3f}")
print(f"    ever_had_fire         – rate: {dataset['ever_had_fire'].mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Splitting train (2018-2020) / val (2021) …")

CANDIDATE_FEATURES = [
    "avg_tmax_c",
    "avg_tmin_c",
    "tot_prcp_mm",
    "temp_range",
    "fire_count_lag1",
    "fire_count_lag2",
    "cumulative_fire_count",
    "ever_had_fire",
]

train_df = dataset[dataset["Year"].isin(TRAIN_YEARS)].copy()
val_df   = dataset[dataset["Year"].isin(VAL_YEARS)].copy()

X_train_raw = train_df[CANDIDATE_FEATURES].values
y_train     = train_df["fire_event"].values
X_val_raw   = val_df[CANDIDATE_FEATURES].values
y_val       = val_df["fire_event"].values

print(f"    Train : {len(y_train):,} samples  "
      f"(fire rate {y_train.mean()*100:.1f}%)")
print(f"    Val   : {len(y_val):,} samples  "
      f"(fire rate {y_val.mean()*100:.1f}%)")

# ── Optional stratified subsample for quantum kernel tractability ─────────────
train_sampled = False
if MAX_TRAIN_SAMPLES and len(X_train_raw) > MAX_TRAIN_SAMPLES:
    X_train_raw, y_train = resample(
        X_train_raw, y_train,
        n_samples=MAX_TRAIN_SAMPLES,
        stratify=y_train,
        random_state=RANDOM_SEED,
    )
    train_sampled = True
    print(f"    ⚠  Subsampled to {MAX_TRAIN_SAMPLES} training examples "
          f"for quantum kernel tractability.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE SELECTION (top N_QUBITS via mutual information)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6] Selecting top {N_QUBITS} features via mutual information …")

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=N_QUBITS,
)
selector.fit(X_train_raw, y_train)

selected_features = [
    f for f, keep in zip(CANDIDATE_FEATURES, selector.get_support()) if keep
]
mi_scores = dict(zip(CANDIDATE_FEATURES, selector.scores_))
print(f"    MI scores   : { {k: round(v,4) for k,v in mi_scores.items()} }")
print(f"    Selected    : {selected_features}")

X_train_sel = selector.transform(X_train_raw)
X_val_sel   = selector.transform(X_val_raw)


# ══════════════════════════════════════════════════════════════════════════════
# 7. STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Standardising features (StandardScaler fit on train) …")

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_sel)
X_val_sc    = scaler.transform(X_val_sel)

print(f"    Train mean (post-scale): {X_train_sc.mean(axis=0).round(4)}")
print(f"    Train std  (post-scale): {X_train_sc.std(axis=0).round(4)}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. QUANTUM FEATURE MAP & KERNEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[8] Building ZZFeatureMap ({N_QUBITS} qubits, reps={ZZ_REPS}) …")

# ZZFeatureMap encodes each feature into a qubit via Hadamard + Rz + ZZ
# entanglement gates.  reps controls how many times the encoding block is
# repeated (more reps → higher expressibility, deeper circuit).
feature_map    = ZZFeatureMap(feature_dimension=N_QUBITS, reps=ZZ_REPS)
circuit_depth  = feature_map.decompose().depth()
num_params     = feature_map.num_parameters

print(f"    Num qubits             : {N_QUBITS}")
print(f"    Circuit depth (decomp) : {circuit_depth}")
print(f"    Num parameters         : {num_params}")

# FidelityStatevectorKernel: computes K(x,y) = |<φ(x)|φ(y)>|² exactly via
# statevector simulation — equivalent to FidelityQuantumKernel with shots → ∞
quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)

print("\n    Computing quantum kernel matrices (train×train, val×train) …")
t0       = time.perf_counter()
K_train  = quantum_kernel.evaluate(X_train_sc, X_train_sc)
K_val    = quantum_kernel.evaluate(X_val_sc,   X_train_sc)
q_kernel_runtime = time.perf_counter() - t0
print(f"    Kernel computation time : {q_kernel_runtime:.2f}s")
print(f"    K_train shape : {K_train.shape}  |  K_val shape : {K_val.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. QUANTUM SVM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Training Quantum SVM (precomputed kernel) …")

qsvm = SVC(kernel="precomputed", probability=True, class_weight="balanced", random_state=RANDOM_SEED)

t0 = time.perf_counter()
qsvm.fit(K_train, y_train)
q_train_runtime = time.perf_counter() - t0

y_pred_q = qsvm.predict(K_val)
y_prob_q = qsvm.predict_proba(K_val)[:, 1]

q_metrics = {
    "roc_auc"   : float(roc_auc_score(y_val, y_prob_q)),
    "f1"        : float(f1_score(y_val, y_pred_q, zero_division=0)),
    "precision" : float(precision_score(y_val, y_pred_q, zero_division=0)),
    "recall"    : float(recall_score(y_val, y_pred_q, zero_division=0)),
}
print(f"    ROC-AUC   : {q_metrics['roc_auc']:.4f}")
print(f"    F1        : {q_metrics['f1']:.4f}")
print(f"    Precision : {q_metrics['precision']:.4f}")
print(f"    Recall    : {q_metrics['recall']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CLASSICAL SVM BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Training Classical SVM baseline (RBF kernel) …")

csvm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED)

t0 = time.perf_counter()
csvm.fit(X_train_sc, y_train)
c_train_runtime = time.perf_counter() - t0

y_pred_c = csvm.predict(X_val_sc)
y_prob_c = csvm.predict_proba(X_val_sc)[:, 1]

c_metrics = {
    "roc_auc"   : float(roc_auc_score(y_val, y_prob_c)),
    "f1"        : float(f1_score(y_val, y_pred_c, zero_division=0)),
    "precision" : float(precision_score(y_val, y_pred_c, zero_division=0)),
    "recall"    : float(recall_score(y_val, y_pred_c, zero_division=0)),
}
print(f"    ROC-AUC   : {c_metrics['roc_auc']:.4f}")
print(f"    F1        : {c_metrics['f1']:.4f}")
print(f"    Precision : {c_metrics['precision']:.4f}")
print(f"    Recall    : {c_metrics['recall']:.4f}")

# ── Comparison summary ────────────────────────────────────────────────────────
delta_auc = q_metrics["roc_auc"] - c_metrics["roc_auc"]
print(f"\n    Δ ROC-AUC (Quantum – Classical) : {delta_auc:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. LOG RESULTS TO JSON
# ══════════════════════════════════════════════════════════════════════════════
run_log = {
    "experiment"   : "quantum_kernel_wildfire_svm",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "random_seed"  : RANDOM_SEED,

    "data": {
        "source_file"    : "wildfire_weather.csv",
        "train_years"    : TRAIN_YEARS,
        "val_years"      : VAL_YEARS,
        "train_samples"  : int(len(y_train)),
        "val_samples"    : int(len(y_val)),
        "subsampled"     : train_sampled,
        "max_train_cap"  : MAX_TRAIN_SAMPLES,
        "class_balance"  : {
            "train_fire_rate": round(float(y_train.mean()), 4),
            "val_fire_rate"  : round(float(y_val.mean()),   4),
        },
    },

    "feature_engineering": {
        "candidate_features": CANDIDATE_FEATURES,
        "selection_method"  : "SelectKBest (mutual_info_classif)",
        "mi_scores"         : {k: round(v, 4) for k, v in mi_scores.items()},
        "selected_features" : selected_features,
        "scaler"            : "StandardScaler (fit on train)",
    },

    "quantum_config": {
        "kernel"           : "FidelityStatevectorKernel",
        "kernel_type"      : "ZZFeatureMap inner-product fidelity |<φ(x)|φ(y)>|²",
        "feature_map"      : "ZZFeatureMap",
        "num_qubits"       : N_QUBITS,
        "reps"             : ZZ_REPS,
        "circuit_depth_decomposed": int(circuit_depth),
        "num_parameters"   : int(num_params),
        "shots"            : "statevector (exact — equivalent to shots → ∞)",
        "backend"          : "statevector_simulator",
        "K_train_shape"    : list(K_train.shape),
        "K_val_shape"      : list(K_val.shape),
        "kernel_runtime_s" : round(q_kernel_runtime, 4),
        "svm_train_runtime_s": round(q_train_runtime, 4),
    },

    "quantum_svm_results": {
        "sklearn_kernel": "precomputed",
        "class_weight"  : "balanced",
        **{k: round(v, 4) for k, v in q_metrics.items()},
    },

    "classical_svm_baseline": {
        "sklearn_kernel"  : "rbf",
        "class_weight"    : "balanced",
        "train_runtime_s" : round(c_train_runtime, 4),
        **{k: round(v, 4) for k, v in c_metrics.items()},
    },

    "comparison": {
        "delta_roc_auc"   : round(delta_auc, 4),
        "delta_f1"        : round(q_metrics["f1"] - c_metrics["f1"], 4),
        "delta_precision" : round(q_metrics["precision"] - c_metrics["precision"], 4),
        "delta_recall"    : round(q_metrics["recall"] - c_metrics["recall"], 4),
    },
}

out_path = RESULTS_DIR / f"run_{int(time.time())}.json"
out_path.write_text(json.dumps(run_log, indent=2))

print(f"\n{'='*65}")
print(f"Results saved → {out_path}")
print("=" * 65)

# ── Comparison table ──────────────────────────────────────────────────────────
print()
print("{:<20} {:>14} {:>14} {:>14}".format(
    "Metric", "Quantum SVM", "Classical SVM", "Delta (Q-C)"))
print("-" * 64)
for metric in ["roc_auc", "f1", "precision", "recall"]:
    print("{:<20} {:>14.4f} {:>14.4f} {:>+14.4f}".format(
        metric,
        q_metrics[metric],
        c_metrics[metric],
        q_metrics[metric] - c_metrics[metric],
    ))
print()
print("class_weight    : balanced (both models)")
print("Kernel runtime  : {:.4f}s  (quantum kernel matrices)".format(q_kernel_runtime))
print("Classical train : {:.4f}s".format(c_train_runtime))
print("Train samples   : {} (subsampled={})".format(len(y_train), train_sampled))
print("Val samples     : {}".format(len(y_val)))
print("Selected feats  : {}".format(selected_features))
print("Num qubits      : {}".format(N_QUBITS))
print("Circuit depth   : {}".format(circuit_depth))
