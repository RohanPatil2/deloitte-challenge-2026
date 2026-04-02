# src/models.py

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import TARGET, RANDOM_SEED


def get_models():
    return {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_SEED
        ),
    }


def evaluate(y_true, y_pred, model_name, experiment_name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1))) * 100
    print(f"    {model_name:<25} MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  R2={r2:>7.4f}  MAPE={mape:>6.2f}%")
    return {
        "Experiment": experiment_name,
        "Model":      model_name,
        "MAE":        round(mae, 2),
        "RMSE":       round(rmse, 2),
        "R2":         round(r2, 4),
        "MAPE":       round(mape, 2),
    }


def run_experiment(train, test, features, experiment_name):
    # Work on copies
    train_c = train[features + [TARGET]].copy()
    test_c  = test[features + [TARGET]].copy()

    # Impute nulls with train median per column
    # This handles lag features that are NaN for the first year of a ZIP
    for col in features:
        median_val = train_c[col].median()
        train_c[col] = train_c[col].fillna(median_val)
        test_c[col]  = test_c[col].fillna(median_val)

    # Drop only rows where TARGET itself is null
    train_c = train_c.dropna(subset=[TARGET])
    test_c  = test_c.dropna(subset=[TARGET])

    X_train = train_c[features]
    y_train = train_c[TARGET]
    X_test  = test_c[features]
    y_test  = test_c[TARGET]

    print(f"\n{'='*65}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Train rows : {len(X_train):,}  |  Test rows: {len(X_test):,}")
    print(f"  Features   : {len(features)}")
    print(f"{'='*65}")

    all_metrics = []

    # Naive baseline: last year's premium
    if "premium_lag1" in test_c.columns:
        naive_pred = test_c["premium_lag1"]
        m = evaluate(y_test, naive_pred, "Naive (Lag-1)", experiment_name)
        all_metrics.append(m)

    # Scale for Ridge
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ML models
    for name, model in get_models().items():
        if name == "Ridge":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        m = evaluate(y_test, preds, name, experiment_name)
        all_metrics.append(m)

    return all_metrics
