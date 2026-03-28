"""
Variational Quantum Classifier (VQC) Pipeline for Wildfire Prediction
======================================================================
Uses Qiskit's VQC with:
  • ZZFeatureMap   – data encoding (4 features → 4 qubits, reps=1)
  • RealAmplitudes – variational ansatz (4 qubits, reps=3, linear entanglement)
  • COBYLA         – gradient-free optimizer via scipy wrapper (maxiter=100)
  • WeightedCrossEntropyLoss – custom Loss subclass that applies
                               class_weight='balanced' directly in the loss,
                               since this version of qiskit-machine-learning
                               does not expose sample_weight in fit().

Same data pipeline and train/val split as quantum_kernel_pipeline.py:
  • Grain         : zip × year (2018–2021)
  • Train years   : 2018–2020  (800 stratified samples for tractability)
  • Val year      : 2021        (full 2,593 samples)
  • Top 4 features via mutual information (SelectKBest)
  • StandardScaler fit on train

Results are saved to results/quantum_runs/<timestamp>_vqc.json.
A final comparison table prints VQC alongside the quantum kernel SVM
(6-qubit, 800 samples) and the best classical baseline (Logistic Regression).

Reproducibility: RANDOM_SEED = 42 throughout.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.algorithms.trainable_model import Loss

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
N_QUBITS          = 4        # qubits = selected features fed to the circuit
ZZ_REPS           = 1        # ZZFeatureMap repetitions (reps=1 keeps depth manageable)
ANSATZ_REPS       = 3        # RealAmplitudes repetitions → 4×(3+1)=16 parameters
MAX_ITER          = 100      # COBYLA max iterations (gradient-free)
MAX_TRAIN_SAMPLES = 800

TRAIN_YEARS = [2018, 2019, 2020]
VAL_YEARS   = [2021]

RAW_DIR     = Path("data/raw")
RESULTS_DIR = Path("results/quantum_runs")
METRICS_DIR = Path("results/quantum_metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM LOSS  –  Weighted Cross-Entropy
# ══════════════════════════════════════════════════════════════════════════════

class WeightedCrossEntropyLoss(Loss):
    """
    Cross-entropy loss with per-class weights applied at the sample level.

    Equivalent to class_weight='balanced' in scikit-learn classifiers:
    minority class (fire=1) receives higher weight so the optimiser pays
    more attention to it despite the 8.4% base rate.

    evaluate() returns shape (N, 1) — per-sample weighted CE — which the
    OneHotObjectiveFunction inside VQC sums and divides by N to get the
    scalar objective value.

    gradient() returns the softmax Jacobian scaled by the same weights.
    COBYLA never calls gradient(), but the method is required by the
    abstract Loss base class.
    """

    def __init__(self, w0: float, w1: float):
        """
        Args:
            w0: weight for the majority class (fire=0).
            w1: weight for the minority class (fire=1).
        """
        self.w0 = w0
        self.w1 = w1

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        if predict.ndim == 1:
            predict = predict.reshape(1, -1)
            target  = target.reshape(1, -1)

        # Standard cross-entropy per sample: -Σ target * log2(predict)
        ce = -np.einsum(
            "ij,ij->i",
            target,
            np.log2(np.clip(predict, a_min=1e-10, a_max=None)),
        ).reshape(-1, 1)

        # Scale each sample by the weight of its true class
        cls = np.argmax(target, axis=-1)          # 0 or 1
        w   = np.where(cls == 1, self.w1, self.w0).reshape(-1, 1)
        return ce * w

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Softmax Jacobian scaled by class weights (same formula as CrossEntropyLoss)."""
        if predict.ndim == 1:
            predict = predict.reshape(1, -1)
            target  = target.reshape(1, -1)

        grad = np.einsum("ij,i->ij", predict, np.sum(target, axis=1)) - target
        cls  = np.argmax(target, axis=-1)
        w    = np.where(cls == 1, self.w1, self.w0).reshape(-1, 1)
        return grad * w


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & SEPARATE ROW TYPES  (identical to quantum_kernel_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("VQC PIPELINE – WILDFIRE PREDICTION")
print("=" * 65)
print("\n[1] Loading wildfire_weather.csv …")

df = pd.read_csv(
    RAW_DIR / "wildfire_weather.csv",
    encoding="latin-1",
    low_memory=False,
)
print(f"    Raw rows loaded : {len(df):,}")

df["zip"] = pd.to_numeric(df["zip"], errors="coerce")
df = df.dropna(subset=["zip"])
df["zip"] = df["zip"].astype(int).astype(str)

weather = df[df["OBJECTID"].isna()].copy()
weather["year"] = weather["year_month"].str[:4].astype(float).astype("Int64")
weather = weather.dropna(subset=["year", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"])
weather = weather[weather["year"].isin([2018, 2019, 2020, 2021])]

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

weather_agg = (
    weather
    .groupby(["zip", "year"])
    .agg(
        avg_tmax_c  = ("avg_tmax_c",  "mean"),
        avg_tmin_c  = ("avg_tmin_c",  "mean"),
        tot_prcp_mm = ("tot_prcp_mm", "sum"),
    )
    .reset_index()
    .rename(columns={"year": "Year"})
)
weather_agg["Year"] = weather_agg["Year"].astype(int)
print(f"    Zip-year weather records : {len(weather_agg):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BINARY TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Building binary fire-event targets …")

fire_zip_year = (
    fire[fire["Year"].isin([2018, 2019, 2020, 2021])]
    .groupby(["zip", "Year"])
    .size()
    .reset_index(name="fire_count_target")
)
fire_zip_year["fire_event"] = 1

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

hist_counts = (
    fire.groupby(["zip", "Year"])
    .size()
    .reset_index(name="fire_count")
)

all_years_for_lag = sorted(set([2016, 2017] + TRAIN_YEARS + VAL_YEARS))
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

grid["fire_count_lag1"] = (
    grid.groupby("zip")["fire_count"]
    .transform(lambda s: s.shift(1).fillna(0))
)
grid["fire_count_lag2"] = (
    grid.groupby("zip")["fire_count"]
    .transform(lambda s: s.shift(2).fillna(0))
)
grid["cumulative_fire_count"] = (
    grid.groupby("zip")["fire_count"]
    .transform(lambda s: s.cumsum().shift(1).fillna(0))
)
grid["ever_had_fire"] = (grid["cumulative_fire_count"] > 0).astype(int)

lag_features = grid[grid["Year"].isin(TRAIN_YEARS + VAL_YEARS)][
    ["zip", "Year", "fire_count_lag1", "fire_count_lag2",
     "cumulative_fire_count", "ever_had_fire"]
]
dataset = dataset.merge(lag_features, on=["zip", "Year"], how="left")
for col in ["fire_count_lag1", "fire_count_lag2", "cumulative_fire_count", "ever_had_fire"]:
    dataset[col] = dataset[col].fillna(0)

dataset["temp_range"] = dataset["avg_tmax_c"] - dataset["avg_tmin_c"]

print(f"    fire_count_lag1       – mean: {dataset['fire_count_lag1'].mean():.3f}")
print(f"    fire_count_lag2       – mean: {dataset['fire_count_lag2'].mean():.3f}")
print(f"    cumulative_fire_count – mean: {dataset['cumulative_fire_count'].mean():.3f}")
print(f"    ever_had_fire         – rate: {dataset['ever_had_fire'].mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / VAL SPLIT  +  SUBSAMPLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Splitting train (2018-2020) / val (2021) …")

train_df = dataset[dataset["Year"].isin(TRAIN_YEARS)].copy()
val_df   = dataset[dataset["Year"].isin(VAL_YEARS)].copy()

X_train_raw = train_df[CANDIDATE_FEATURES].values
y_train     = train_df["fire_event"].values
X_val_raw   = val_df[CANDIDATE_FEATURES].values
y_val       = val_df["fire_event"].values

print(f"    Train : {len(y_train):,} samples  (fire rate {y_train.mean()*100:.1f}%)")
print(f"    Val   : {len(y_val):,} samples  (fire rate {y_val.mean()*100:.1f}%)")

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
          f"for VQC tractability.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE SELECTION  (top N_QUBITS via mutual information)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6] Selecting top {N_QUBITS} features via mutual information …")

selector = SelectKBest(score_func=mutual_info_classif, k=N_QUBITS)
selector.fit(X_train_raw, y_train)

selected_features = [
    f for f, keep in zip(CANDIDATE_FEATURES, selector.get_support()) if keep
]
mi_scores = dict(zip(CANDIDATE_FEATURES, selector.scores_))
print(f"    MI scores : { {k: round(v,4) for k,v in mi_scores.items()} }")
print(f"    Selected  : {selected_features}")

X_train_sel = selector.transform(X_train_raw)
X_val_sel   = selector.transform(X_val_raw)


# ══════════════════════════════════════════════════════════════════════════════
# 7. STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Standardising features (StandardScaler fit on train) …")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sel)
X_val_sc   = scaler.transform(X_val_sel)

print(f"    Train mean (post-scale): {X_train_sc.mean(axis=0).round(4)}")
print(f"    Train std  (post-scale): {X_train_sc.std(axis=0).round(4)}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. BUILD VQC COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[8] Building VQC circuit ({N_QUBITS} qubits) …")

# ZZFeatureMap: encodes each of the 4 features into a qubit via
# H + Rz + ZZ cross-coupling gates (same family as kernel pipeline).
feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=ZZ_REPS)

# RealAmplitudes: variational ansatz with Ry gates + CNOT entanglement.
# reps=3, linear entanglement → 4 × (3+1) = 16 trainable parameters.
ansatz = RealAmplitudes(
    num_qubits=N_QUBITS,
    reps=ANSATZ_REPS,
    entanglement="linear",
)

# Full VQC circuit = feature_map → ansatz → measurement
full_circuit  = feature_map.compose(ansatz)
circuit_depth = full_circuit.decompose().depth()
num_params    = ansatz.num_parameters

print(f"    Feature map  : ZZFeatureMap  (reps={ZZ_REPS}, depth={feature_map.decompose().depth()})")
print(f"    Ansatz       : RealAmplitudes (reps={ANSATZ_REPS}, linear, {num_params} params)")
print(f"    Full circuit depth (decomposed) : {circuit_depth}")


# ── Class weights for balanced loss ──────────────────────────────────────────
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
w0, w1 = float(cw[0]), float(cw[1])
print(f"\n    Class weights : no-fire={w0:.3f}, fire={w1:.3f}  "
      f"(ratio {w1/w0:.1f}×)")

loss_fn = WeightedCrossEntropyLoss(w0=w0, w1=w1)


# ── COBYLA optimizer (gradient-free, via scipy wrapper) ───────────────────────
iteration_log: list[float] = []

def cobyla_minimizer(fun, x0, jac=None, bounds=None):
    """
    Wraps scipy COBYLA for use as a qiskit-machine-learning Minimizer callable.
    COBYLA is gradient-free, so `jac` is accepted but ignored.
    """
    def tracked_fun(params):
        val = fun(params)
        iteration_log.append(val)
        if len(iteration_log) % 20 == 0 or len(iteration_log) == 1:
            print(f"      iter {len(iteration_log):>3} | loss = {val:.5f}")
        return val

    return scipy_minimize(
        tracked_fun, x0,
        method="COBYLA",
        options={"maxiter": MAX_ITER, "rhobeg": 0.5},
    )


# ── Sampler ───────────────────────────────────────────────────────────────────
# StatevectorSampler: computes exact probability distributions via statevector
# simulation before sampling.  default_shots=1024 gives clean probability
# estimates; seed ensures reproducibility.
sampler = StatevectorSampler(seed=RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 9. TRAIN VQC
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[9] Training VQC (COBYLA maxiter={MAX_ITER}) …")
print(f"    Samples : {len(y_train)}  |  Parameters : {num_params}")
print(f"    Logging loss every 20 iterations …")
print()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    loss=loss_fn,
    optimizer=cobyla_minimizer,
    sampler=sampler,
)

t0 = time.perf_counter()
vqc.fit(X_train_sc, y_train)
vqc_runtime = time.perf_counter() - t0

print()
print(f"    Training complete in {vqc_runtime:.1f}s")
print(f"    Iterations run : {len(iteration_log)}")
if iteration_log:
    print(f"    Initial loss   : {iteration_log[0]:.5f}")
    print(f"    Final loss     : {iteration_log[-1]:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. EVALUATE ON VAL SET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Evaluating on 2021 validation set …")

y_pred_vqc  = vqc.predict(X_val_sc)
y_prob_vqc  = vqc.predict_proba(X_val_sc)[:, 1]

vqc_metrics = {
    "roc_auc"   : float(roc_auc_score(y_val, y_prob_vqc)),
    "f1"        : float(f1_score(y_val, y_pred_vqc, zero_division=0)),
    "precision" : float(precision_score(y_val, y_pred_vqc, zero_division=0)),
    "recall"    : float(recall_score(y_val, y_pred_vqc, zero_division=0)),
}
print(f"    ROC-AUC   : {vqc_metrics['roc_auc']:.4f}")
print(f"    F1        : {vqc_metrics['f1']:.4f}")
print(f"    Precision : {vqc_metrics['precision']:.4f}")
print(f"    Recall    : {vqc_metrics['recall']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. SAVE RESULTS TO JSON
# ══════════════════════════════════════════════════════════════════════════════
run_log = {
    "experiment"   : "vqc_wildfire",
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

    "vqc_config": {
        "num_qubits"       : N_QUBITS,
        "feature_map"      : f"ZZFeatureMap (reps={ZZ_REPS})",
        "feature_map_depth": int(feature_map.decompose().depth()),
        "ansatz"           : f"RealAmplitudes (reps={ANSATZ_REPS}, linear)",
        "ansatz_depth"     : int(ansatz.decompose().depth()),
        "full_circuit_depth": int(circuit_depth),
        "num_parameters"   : int(num_params),
        "optimizer"        : f"COBYLA (maxiter={MAX_ITER}, scipy wrapper)",
        "sampler"          : "StatevectorSampler (default_shots=1024)",
        "loss"             : "WeightedCrossEntropyLoss (class_weight=balanced)",
        "class_weights"    : {"no_fire": round(w0, 4), "fire": round(w1, 4)},
        "iterations_run"   : len(iteration_log),
        "initial_loss"     : round(iteration_log[0], 6) if iteration_log else None,
        "final_loss"       : round(iteration_log[-1], 6) if iteration_log else None,
        "train_runtime_s"  : round(vqc_runtime, 2),
    },

    "vqc_results": {
        **{k: round(v, 4) for k, v in vqc_metrics.items()},
    },
}

out_path = RESULTS_DIR / f"run_{int(time.time())}_vqc.json"
out_path.write_text(json.dumps(run_log, indent=2))

print(f"\n{'='*65}")
print(f"Results saved → {out_path}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 12. COMPARISON TABLE
#     VQC  vs  Quantum Kernel SVM (6-qubit)  vs  Best Classical (Logistic Reg.)
# ══════════════════════════════════════════════════════════════════════════════

# ── Load quantum kernel results (most recent 4v6-qubit run) ──────────────────
qk_metrics_6q = {}
qk_metrics_4q = {}
try:
    kernel_jsons = sorted(
        [f for f in RESULTS_DIR.glob("run_*.json")
         if "vqc" not in f.name],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for jf in kernel_jsons:
        data = json.loads(jf.read_text())
        if "quantum_6qubit" in data and "quantum_4qubit" in data:
            q6 = data["quantum_6qubit"]
            q4 = data["quantum_4qubit"]
            for k in ["roc_auc", "f1", "precision", "recall"]:
                qk_metrics_6q[k] = q6.get(k, float("nan"))
                qk_metrics_4q[k] = q4.get(k, float("nan"))
            break
except Exception:
    pass   # comparison columns will show n/a if files are missing

# ── Load best classical baseline (Logistic Regression) ───────────────────────
cl_metrics = {}
try:
    cl_data = json.loads((METRICS_DIR / "classical_baselines.json").read_text())
    lr = cl_data["models"]["logistic_regression"]
    for k in ["roc_auc", "f1", "precision", "recall"]:
        cl_metrics[k] = lr.get(k, float("nan"))
except Exception:
    pass

# ── Print table ───────────────────────────────────────────────────────────────
col = 13
metrics = ["roc_auc", "f1", "precision", "recall"]

print()
print("{:<12} {:>{c}} {:>{c}} {:>{c}} {:>{c}}".format(
    "Metric",
    "VQC (4q)", "Q-Kernel 4q", "Q-Kernel 6q", "LR (classical)",
    c=col))
print("-" * (12 + col * 4 + 4))

for m in metrics:
    def fmt(d, key):
        v = d.get(key, None)
        return f"{v:>{col}.4f}" if v is not None else f"{'n/a':>{col}}"
    print("{:<12}{}{}{}{}".format(
        m,
        fmt(vqc_metrics,   m),
        fmt(qk_metrics_4q, m),
        fmt(qk_metrics_6q, m),
        fmt(cl_metrics,    m),
    ))

print()
# Circuit info row
print("{:<28} {:>8} {:>10} {:>10}".format("", "VQC", "Q-Ker 4q", "Q-Ker 6q"))
print("{:<28} {:>8} {:>10} {:>10}".format(
    "Num qubits", N_QUBITS,
    qk_metrics_4q and 4 or "n/a",
    qk_metrics_6q and 6 or "n/a"))
print("{:<28} {:>8} {:>10} {:>10}".format(
    "Circuit depth (decomp)", circuit_depth, 31, 49))
print("{:<28} {:>8} {:>10} {:>10}".format(
    "Trainable parameters", num_params, "n/a", "n/a"))
print("{:<28} {:>7.1f}s".format("VQC train runtime", vqc_runtime))
print()
print("Train samples  : {} (subsampled={}) | Val samples : {}".format(
    len(y_train), train_sampled, len(y_val)))
print("VQC features   : {}".format(selected_features))
print("Loss           : WeightedCrossEntropyLoss  "
      "(w0={:.3f}, w1={:.3f})".format(w0, w1))
print("Optimizer      : COBYLA  maxiter={}  iterations_run={}".format(
    MAX_ITER, len(iteration_log)))
