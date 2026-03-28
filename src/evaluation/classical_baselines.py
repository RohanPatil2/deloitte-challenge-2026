"""
Classical Baseline Models for Wildfire Prediction
==================================================
Replicates the exact data loading, feature engineering, and train/val split
from quantum_kernel_pipeline.py, then trains four classical classifiers:

  1. Logistic Regression
  2. Random Forest
  3. GradientBoostingClassifier  (sklearn, used as XGBoost proxy)
  4. SVM (RBF kernel)

All models use class_weight='balanced' (or equivalent sample weighting for
GradientBoostingClassifier, which does not expose class_weight directly).
Classical models train on the full 7,779-sample training set — no subsampling
needed unlike the quantum kernel SVM.

Train years : 2018–2020  |  Val year : 2021
Features    : all 8 candidate features (no qubit-count constraint)
Output      : results/quantum_metrics/classical_baselines.json

Reproducibility: RANDOM_SEED = 42 throughout.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_YEARS = [2018, 2019, 2020]
VAL_YEARS   = [2021]

RAW_DIR     = Path("data/raw")
RESULTS_DIR = Path("results/quantum_metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All candidate features — no qubit constraint here, so use the full set
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
#    (identical to quantum_kernel_pipeline.py steps 1-4)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("CLASSICAL BASELINES – WILDFIRE PREDICTION")
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

# Weather rows: OBJECTID is NaN; year comes from year_month (YYYY-MM)
weather = df[df["OBJECTID"].isna()].copy()
weather["year"] = weather["year_month"].str[:4].astype(float).astype("Int64")
weather = weather.dropna(subset=["year", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"])
weather = weather[weather["year"].isin([2018, 2019, 2020, 2021])]

# Fire rows: OBJECTID not NaN; year comes from the Year column
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
# 3. CREATE BINARY TARGET
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

# Need 2016 for lag-2 of 2018, and 2017 for lag-1 of 2018
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

# cumulative_fire_count: total fires in ALL years before the current year
grid["cumulative_fire_count"] = (
    grid.groupby("zip")["fire_count"]
    .transform(lambda s: s.cumsum().shift(1).fillna(0))
)

# ever_had_fire: did this zip have any fire before this year?
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
# 5. TRAIN / VALIDATION SPLIT  (full dataset — no subsampling)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Splitting train (2018-2020) / val (2021) …")

train_df = dataset[dataset["Year"].isin(TRAIN_YEARS)].copy()
val_df   = dataset[dataset["Year"].isin(VAL_YEARS)].copy()

X_train = train_df[CANDIDATE_FEATURES].values
y_train = train_df["fire_event"].values
X_val   = val_df[CANDIDATE_FEATURES].values
y_val   = val_df["fire_event"].values

print(f"    Train : {len(y_train):,} samples  (fire rate {y_train.mean()*100:.1f}%)")
print(f"    Val   : {len(y_val):,} samples  (fire rate {y_val.mean()*100:.1f}%)")
print(f"    Features used : {CANDIDATE_FEATURES}")

# StandardScaler — fit on train, applied to both splits
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

# Sample weights for GradientBoostingClassifier (no native class_weight param)
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEFINE & TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Training classical baselines …")

def evaluate(name, model, X_tr, y_tr, X_v, y_v,
             fit_kwargs=None, use_scaled=True):
    """Fit model, predict on val, return metrics dict."""
    Xtr = X_tr if use_scaled else X_tr
    Xv  = X_v  if use_scaled else X_v
    t0  = time.perf_counter()
    model.fit(Xtr, y_tr, **(fit_kwargs or {}))
    runtime = time.perf_counter() - t0

    y_pred = model.predict(Xv)
    y_prob = model.predict_proba(Xv)[:, 1]

    metrics = {
        "roc_auc"   : float(roc_auc_score(y_v, y_prob)),
        "f1"        : float(f1_score(y_v, y_pred, zero_division=0)),
        "precision" : float(precision_score(y_v, y_pred, zero_division=0)),
        "recall"    : float(recall_score(y_v, y_pred, zero_division=0)),
        "train_runtime_s": round(runtime, 4),
    }
    print(f"\n    [{name}]")
    print(f"      ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"      F1        : {metrics['f1']:.4f}")
    print(f"      Precision : {metrics['precision']:.4f}")
    print(f"      Recall    : {metrics['recall']:.4f}")
    print(f"      Train time: {metrics['train_runtime_s']}s")
    return metrics


# 1. Logistic Regression
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_SEED,
)
lr_metrics = evaluate("Logistic Regression", lr, X_train_sc, y_train, X_val_sc, y_val)

# 2. Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf_metrics = evaluate("Random Forest", rf, X_train_sc, y_train, X_val_sc, y_val)

# 3. Gradient Boosting (sklearn, XGBoost proxy)
#    GradientBoostingClassifier has no class_weight param; pass sample_weight
#    via fit() to achieve the same balanced-class effect.
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=RANDOM_SEED,
)
gb_metrics = evaluate(
    "GradientBoosting (XGBoost proxy)", gb,
    X_train_sc, y_train, X_val_sc, y_val,
    fit_kwargs={"sample_weight": sample_weights},
)

# 4. Classical SVM (RBF kernel)
svm = SVC(
    kernel="rbf",
    class_weight="balanced",
    probability=True,
    random_state=RANDOM_SEED,
)
svm_metrics = evaluate("SVM (RBF)", svm, X_train_sc, y_train, X_val_sc, y_val)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS TO JSON
# ══════════════════════════════════════════════════════════════════════════════
results = {
    "experiment"   : "classical_baselines_wildfire",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "random_seed"  : RANDOM_SEED,
    "data": {
        "source_file"    : "wildfire_weather.csv",
        "train_years"    : TRAIN_YEARS,
        "val_years"      : VAL_YEARS,
        "train_samples"  : int(len(y_train)),
        "val_samples"    : int(len(y_val)),
        "subsampled"     : False,
        "class_balance"  : {
            "train_fire_rate": round(float(y_train.mean()), 4),
            "val_fire_rate"  : round(float(y_val.mean()),   4),
        },
    },
    "features": {
        "all_features_used": CANDIDATE_FEATURES,
        "scaler"           : "StandardScaler (fit on train)",
        "class_weight"     : "balanced (sample_weight for GradientBoosting)",
    },
    "models": {
        "logistic_regression": {
            "params"     : {"max_iter": 1000, "class_weight": "balanced"},
            **{k: round(v, 4) for k, v in lr_metrics.items()},
        },
        "random_forest": {
            "params"     : {"n_estimators": 300, "class_weight": "balanced"},
            **{k: round(v, 4) for k, v in rf_metrics.items()},
        },
        "gradient_boosting": {
            "params"     : {"n_estimators": 300, "learning_rate": 0.05,
                            "max_depth": 4, "subsample": 0.8,
                            "class_weight": "balanced (via sample_weight)"},
            **{k: round(v, 4) for k, v in gb_metrics.items()},
        },
        "svm_rbf": {
            "params"     : {"kernel": "rbf", "class_weight": "balanced"},
            **{k: round(v, 4) for k, v in svm_metrics.items()},
        },
    },
}

out_path = RESULTS_DIR / "classical_baselines.json"
out_path.write_text(json.dumps(results, indent=2))

print(f"\n{'='*65}")
print(f"Results saved → {out_path}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 8. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
models_ordered = [
    ("Logistic Regression",          lr_metrics),
    ("Random Forest",                rf_metrics),
    ("Gradient Boosting",            gb_metrics),
    ("SVM (RBF)",                    svm_metrics),
]

col_w = 26
print()
print("{:<{w}} {:>10} {:>10} {:>11} {:>9} {:>10}".format(
    "Model", "ROC-AUC", "F1", "Precision", "Recall", "Time (s)", w=col_w))
print("-" * 80)
for name, m in models_ordered:
    print("{:<{w}} {:>10.4f} {:>10.4f} {:>11.4f} {:>9.4f} {:>10.4f}".format(
        name,
        m["roc_auc"], m["f1"], m["precision"], m["recall"],
        m["train_runtime_s"],
        w=col_w,
    ))
print()
print("Train : {:,} samples (2018-2020) | Val : {:,} samples (2021)".format(
    len(y_train), len(y_val)))
print("Features : {}".format(", ".join(CANDIDATE_FEATURES)))
print("class_weight = balanced for all models")
