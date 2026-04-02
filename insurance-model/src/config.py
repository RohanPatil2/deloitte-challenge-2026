# src/config.py

from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA      = ROOT / "data" / "raw" / "cal_insurance_fire_census_weather.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_DIR  = ROOT / "data" / "external"

# Wildfire risk files from Shreyas (Person 2)
CLASSICAL_RISK_FILE = EXTERNAL_DIR / "wildfire_risk_classical_2021.csv"
QUANTUM_RISK_FILE   = EXTERNAL_DIR / "wildfire_risk_quantum_2021.csv"

# ── Results ───────────────────────────────────────────────────────────────────
METRICS_DIR     = ROOT / "results" / "metrics"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
FIGURES_DIR     = ROOT / "results" / "figures"

# ── Experiment settings ───────────────────────────────────────────────────────
TRAIN_YEARS = [2018, 2019, 2020]
TEST_YEAR   = 2021
TARGET      = "Earned Premium"
RANDOM_SEED = 42

# ── Column groups ─────────────────────────────────────────────────────────────
ID_COLS = ["Year", "ZIP"]

# Fire-event columns (post-event data, ~83% null, leakage risk — drop all)
DROP_COLS = [
    "OBJECTID", "AGENCY", "UNIT_ID", "FIRE_NAME", "INC_NUM",
    "ALARM_DATE", "CONT_DATE", "CAUSE", "C_METHOD", "OBJECTIVE",
    "GIS_ACRES", "COMMENTS", "COMPLEX_NA", "FIRE_NUM", "COMPLEX_ID",
    "DECADES", "Shape__Are", "Shape__Len", "latitude", "longitude",
    "Alarm_Date2", "year_month", "AGENCY_ID", "FIRE_NAME_ID",
    # Weather columns — 83% missing, not usable
    "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm", "station",
]

# Core insurance + census features (used in all experiments)
BASE_FEATURES = [
    # Insurance core
    "Earned Exposure",
    "Avg PPC",
    "Cov A Amount Weighted Avg",
    "Cov C Amount Weighted Avg",
    "CAT Cov A Fire -  Incurred Losses",
    "CAT Cov A Fire -  Number of Claims",
    "CAT Cov A Smoke -  Incurred Losses",
    "CAT Cov A Smoke -  Number of Claims",
    "Non-CAT Cov A Fire -  Incurred Losses",
    "Non-CAT Cov A Fire -  Number of Claims",
    "Non-CAT Cov A Smoke -  Incurred Losses",
    "Non-CAT Cov A Smoke -  Number of Claims",
    "CAT Cov C Fire -  Incurred Losses",
    "Non-CAT Cov C Fire -  Incurred Losses",
    # Fire risk exposure bands
    "Number of Very High Fire Risk Exposure",
    "Number of High Fire Risk Exposure",
    "Number of Moderate Fire Risk Exposure",
    "Number of Low Fire Risk Exposure",
    "Number of Negligible Fire Risk Exposure",
    # Insurance category (one-hot encoded)
    "Category_HO",
    "Category_CO",
    "Category_DO",
    "Category_DT",
    "Category_MH",
    "Category_RT",
    # Census demographics
    "total_population",
    "median_income",
    "total_housing_units",
    "average_household_size",
    "housing_value",
    "median_monthly_housing_costs",
    "owner_occupied_housing_units",
    "renter_occupied_housing_units",
    "poverty_status",
]

# Lag features (built in feature engineering step)
LAG_FEATURES = [
    "premium_lag1",
    "premium_lag2",
    "premium_yoy_change",
    "premium_pct_change",
    "loss_ratio_lag1",
    "prem_per_exp_lag1",
]

# Wildfire risk column names
FIRE_RISK_COL      = "Avg Fire Risk Score"       # Experiment B — already in dataset
CLASSICAL_RISK_COL = "classical_wildfire_risk"   # Experiment C — from Shreyas
QUANTUM_RISK_COL   = "quantum_wildfire_risk"     # Experiment D — from Shreyas

# ── Experiment feature sets ───────────────────────────────────────────────────
FEATURES_A = BASE_FEATURES + LAG_FEATURES
FEATURES_B = BASE_FEATURES + LAG_FEATURES + [FIRE_RISK_COL]
FEATURES_C = BASE_FEATURES + LAG_FEATURES + [CLASSICAL_RISK_COL]
FEATURES_D = BASE_FEATURES + LAG_FEATURES + [QUANTUM_RISK_COL]
