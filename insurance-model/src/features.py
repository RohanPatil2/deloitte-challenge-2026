# src/features.py

import pandas as pd
import numpy as np
from src.config import TARGET


def add_lag_premium(df):
    """
    Lag features for Earned Premium per ZIP.
    premium_lag1 = last year's premium (strongest single predictor)
    premium_lag2 = two years prior
    NO leakage — shift() only looks backwards.
    """
    df = df.sort_values(["ZIP", "Year"])
    df["premium_lag1"] = df.groupby("ZIP")[TARGET].shift(1)
    df["premium_lag2"] = df.groupby("ZIP")[TARGET].shift(2)
    return df


def add_premium_trend(df):
    """Year-over-year change and % change in premium per ZIP."""
    df["premium_yoy_change"] = df["premium_lag1"] - df["premium_lag2"]
    df["premium_pct_change"] = (
        df["premium_yoy_change"] / (df["premium_lag2"].abs() + 1)
    )
    return df


def add_loss_ratio(df):
    """
    Loss ratio = total fire/smoke losses / earned premium.
    Key insurance metric — higher means more claims relative to premium.
    Using lag so no future leakage.
    """
    total_losses = (
        df["CAT Cov A Fire -  Incurred Losses"].fillna(0)
        + df["Non-CAT Cov A Fire -  Incurred Losses"].fillna(0)
        + df["CAT Cov A Smoke -  Incurred Losses"].fillna(0)
        + df["Non-CAT Cov A Smoke -  Incurred Losses"].fillna(0)
    )
    df["loss_ratio"]      = total_losses / (df[TARGET] + 1)
    df["loss_ratio_lag1"] = df.groupby("ZIP")["loss_ratio"].shift(1)
    return df


def add_premium_per_exposure(df):
    """
    Premium per policy unit — normalizes for ZIP size.
    A ZIP with 1 policy and $5k premium is very different from
    one with 1000 policies and $5k total premium.
    """
    df["premium_per_exposure"] = df[TARGET] / (df["Earned Exposure"] + 1)
    df["prem_per_exp_lag1"]    = df.groupby("ZIP")["premium_per_exposure"].shift(1)
    return df


def build_all_features(df):
    """Apply all feature engineering steps in order."""
    df = add_lag_premium(df)
    df = add_premium_trend(df)
    df = add_loss_ratio(df)
    df = add_premium_per_exposure(df)
    print(f"✅ Feature engineering complete → {df.shape[1]} total columns")
    return df
