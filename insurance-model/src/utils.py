# src/utils.py

import pandas as pd
from pathlib import Path
from src.config import METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR


def save_metrics(metrics_list, filename):
    """Save experiment metrics list to CSV."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    path = METRICS_DIR / filename
    df.to_csv(path, index=False)
    print(f"💾 Saved metrics → {path}")
    return df


def save_predictions(zip_codes, y_true, y_pred, model_name, experiment_name, filename):
    """Save predictions alongside true values for a given model."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "ZIP":        zip_codes,
        "y_true":     y_true.values,
        "y_pred":     y_pred,
        "model":      model_name,
        "experiment": experiment_name,
    })
    path = PREDICTIONS_DIR / filename
    df.to_csv(path, index=False)
    print(f"💾 Saved predictions → {path}")


def load_all_metrics():
    """Load and combine all experiment metric CSVs from results/metrics/."""
    files = sorted(METRICS_DIR.glob("*.csv"))
    if not files:
        print("No metric files found in results/metrics/")
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded {len(files)} metric files → {len(combined)} rows")
    return combined
