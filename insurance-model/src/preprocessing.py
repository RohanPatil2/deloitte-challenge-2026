# src/preprocessing.py

import pandas as pd
import numpy as np
from src.config import (
    RAW_DATA, DROP_COLS, TRAIN_YEARS, TEST_YEAR, TARGET
)


def load_data():
    """Load raw insurance dataset."""
    df = pd.read_csv(RAW_DATA, low_memory=False)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def drop_leakage_columns(df):
    """Remove fire-event columns (post-event, ~83% null) and weather columns."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"✅ Dropped {len(cols_to_drop)} leakage/null columns → {df.shape[1]} remain")
    return df


def fix_target(df):
    """
    Clip negative Earned Premium values to 0.
    A small number of rows have negative premiums (refunds/adjustments).
    We clip rather than drop to preserve ZIP coverage.
    """
    neg_count = (df[TARGET] < 0).sum()
    if neg_count > 0:
        print(f"⚠️  Clipping {neg_count} negative '{TARGET}' values to 0")
        df[TARGET] = df[TARGET].clip(lower=0)
    return df


def impute_missing(df, features):
    """Fill nulls in feature columns with column median."""
    imputed = []
    for col in features:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed.append(col)
    if imputed:
        print(f"✅ Imputed {len(imputed)} columns with median")
    return df


def encode_booleans(df):
    """Convert True/False columns to 1/0."""
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
    if bool_cols:
        print(f"✅ Encoded {len(bool_cols)} boolean columns to int")
    return df


def split_train_test(df):
    """
    Strict temporal split — NO random splitting.
    Train: 2018–2020 | Test: 2021
    """
    train = df[df["Year"].isin(TRAIN_YEARS)].copy()
    test  = df[df["Year"] == TEST_YEAR].copy()
    print(f"✅ Split → Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


def merge_wildfire_risk(df, risk_filepath, new_col_name):
    """
    Merge external wildfire risk scores into the insurance dataframe.
    Files come from Shreyas (Person 2) — columns: ZIP, wildfire_risk_prob.

    Args:
        df           : insurance dataframe (all years)
        risk_filepath: path to Shreyas's CSV
        new_col_name : column name to use ('classical_wildfire_risk'
                       or 'quantum_wildfire_risk')
    Returns:
        df with new risk column (0 for unmatched ZIPs)
    """
    risk_df = pd.read_csv(risk_filepath)
    risk_df = risk_df.rename(columns={"wildfire_risk_prob": new_col_name})
    risk_df["ZIP"] = risk_df["ZIP"].astype(int)

    df = df.merge(risk_df[["ZIP", new_col_name]], on="ZIP", how="left")
    df[new_col_name] = df[new_col_name].fillna(0)

    matched = risk_df["ZIP"].isin(df["ZIP"]).sum()
    total   = len(risk_df)
    print(f"✅ Merged '{new_col_name}': {matched}/{total} ZIPs matched")
    return df


def get_clean_data(features, wildfire_risk_file=None, wildfire_risk_col=None):
    """
    Full preprocessing pipeline.

    Args:
        features          : list of feature column names to use
        wildfire_risk_file: optional path to external risk CSV (Exp C or D)
        wildfire_risk_col : column name for the merged risk score

    Returns:
        train, test dataframes ready for modeling
    """
    df = load_data()
    df = drop_leakage_columns(df)
    df = fix_target(df)
    df = encode_booleans(df)

    # Merge external wildfire risk if provided (Experiments C and D)
    if wildfire_risk_file and wildfire_risk_col:
        df = merge_wildfire_risk(df, wildfire_risk_file, wildfire_risk_col)

    df = impute_missing(df, features)
    train, test = split_train_test(df)
    return train, test
