"""
Wildfire Risk Predictions for 2021 Zip Codes
=============================================
Trains two models on 2018-2020 data and outputs per-zip predicted
wildfire risk probabilities for 2021.

  • Quantum model  : Quantum Kernel SVM (ZZ-6q, 800 samples, balanced)
                     — best quantum ROC-AUC (0.6222) from the ablation study
  • Classical model: Logistic Regression (full 7,779 train samples, balanced)
                     — best classical ROC-AUC (0.8647) from the baseline study

Output
------
  reports/tables/wildfire_risk_predictions_2021.csv
    columns: zip, quantum_risk_prob, classical_risk_prob

Data pipeline is identical to quantum_kernel_pipeline.py (same split, same
lag features, same scaler, same seed) so results are directly comparable.

Reproducibility: RANDOM_SEED = 42 throughout.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
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
N_QUBITS_QUANTUM  = 6       # ZZ-6q is the best quantum model
ZZ_REPS           = 2
MAX_TRAIN_SAMPLES = 800     # quantum kernel cap

TRAIN_YEARS = [2018, 2019, 2020]
VAL_YEARS   = [2021]

RAW_DIR     = Path("data/raw")
OUT_DIR     = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR / "wildfire_risk_predictions_2021.csv"

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
# 1. LOAD & SEPARATE ROW TYPES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("WILDFIRE RISK PREDICTION GENERATOR — 2021")
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

print(f"    Lag features added. Dataset shape: {dataset.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / VAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Splitting train (2018-2020) / val (2021) …")

train_df = dataset[dataset["Year"].isin(TRAIN_YEARS)].copy()
val_df   = dataset[dataset["Year"].isin(VAL_YEARS)].copy()

X_train_all = train_df[CANDIDATE_FEATURES].values
y_train_all = train_df["fire_event"].values
X_val_all   = val_df[CANDIDATE_FEATURES].values
y_val       = val_df["fire_event"].values
val_zips    = val_df["zip"].values

print(f"    Train : {len(y_train_all):,} samples  (fire rate {y_train_all.mean()*100:.1f}%)")
print(f"    Val   : {len(y_val):,} samples  (fire rate {y_val.mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUANTUM MODEL — ZZ-6q, 800 samples
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6] Training Quantum Kernel SVM (ZZ-{N_QUBITS_QUANTUM}q, 800 samples) …")

# Subsample train set for kernel tractability
X_tr_q, y_tr_q = resample(
    X_train_all, y_train_all,
    n_samples=MAX_TRAIN_SAMPLES,
    stratify=y_train_all,
    random_state=RANDOM_SEED,
)

# Select top N_QUBITS_QUANTUM features via mutual information
selector_q = SelectKBest(score_func=mutual_info_classif, k=N_QUBITS_QUANTUM)
selector_q.fit(X_tr_q, y_tr_q)
q_features = [f for f, keep in zip(CANDIDATE_FEATURES, selector_q.get_support()) if keep]
print(f"    Selected features : {q_features}")

X_tr_q_sel = selector_q.transform(X_tr_q)
X_val_q    = selector_q.transform(X_val_all)

scaler_q   = StandardScaler()
X_tr_q_sc  = scaler_q.fit_transform(X_tr_q_sel)
X_val_q_sc = scaler_q.transform(X_val_q)

# Quantum kernel
feature_map = ZZFeatureMap(feature_dimension=N_QUBITS_QUANTUM, reps=ZZ_REPS)
qkernel     = FidelityStatevectorKernel(feature_map=feature_map)

print(f"    Building kernel matrix (800×800 train) …")
K_train = qkernel.evaluate(x_vec=X_tr_q_sc)
K_val   = qkernel.evaluate(x_vec=X_val_q_sc, y_vec=X_tr_q_sc)
print(f"    K_train: {K_train.shape}  K_val: {K_val.shape}")

qsvm = SVC(kernel="precomputed", class_weight="balanced", probability=True,
           random_state=RANDOM_SEED)
qsvm.fit(K_train, y_tr_q)
quantum_probs = qsvm.predict_proba(K_val)[:, 1]
print(f"    Done. Prob range: [{quantum_probs.min():.4f}, {quantum_probs.max():.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLASSICAL MODEL — Logistic Regression, full train set
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[7] Training Logistic Regression (full {len(y_train_all):,} samples) …")

scaler_cl   = StandardScaler()
X_tr_cl_sc  = scaler_cl.fit_transform(X_train_all)
X_val_cl_sc = scaler_cl.transform(X_val_all)

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_SEED,
)
lr.fit(X_tr_cl_sc, y_train_all)
classical_probs = lr.predict_proba(X_val_cl_sc)[:, 1]
print(f"    Done. Prob range: [{classical_probs.min():.4f}, {classical_probs.max():.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 8. BUILD OUTPUT CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Building predictions CSV …")

predictions = pd.DataFrame({
    "zip"                 : val_zips,
    "quantum_risk_prob"   : quantum_probs.round(6),
    "classical_risk_prob" : classical_probs.round(6),
})
predictions = predictions.sort_values("zip").reset_index(drop=True)

predictions.to_csv(OUT_CSV, index=False)
print(f"    Saved → {OUT_CSV}  ({len(predictions):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 9. PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("WILDFIRE RISK PREDICTIONS — 2021 (first 10 rows)")
print("=" * 65)
print()
print("{:<10} {:>20} {:>22}".format("zip", "quantum_risk_prob", "classical_risk_prob"))
print("-" * 55)
for _, row in predictions.head(10).iterrows():
    print("{:<10} {:>20.6f} {:>22.6f}".format(
        row["zip"], row["quantum_risk_prob"], row["classical_risk_prob"]))

print()
print("─" * 55)
print("Summary statistics (2,593 zip-year records):")
print()
for col, label in [
    ("quantum_risk_prob",   "Quantum  (ZZ-6q)  "),
    ("classical_risk_prob", "Classical (LR)     "),
]:
    s = predictions[col]
    print("  {}  min={:.4f}  max={:.4f}  mean={:.4f}  std={:.4f}".format(
        label, s.min(), s.max(), s.mean(), s.std()))

print()
print("Models used:")
print("  Quantum   : FidelityStatevectorKernel + SVC  "
      "(ZZFeatureMap reps=2, 6 qubits, 800 train samples, balanced)")
print("  Classical : LogisticRegression  "
      "(max_iter=1000, 7,779 train samples, balanced)")
print()
print(f"Output → {OUT_CSV}")
