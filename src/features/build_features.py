from __future__ import annotations

import numpy as np
import pandas as pd


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Example feature engineering step."""
    df = df.copy()
    if "Amount" in df.columns and "Time" in df.columns:
        df["Amount_per_Time"] = df["Amount"] / (df["Time"] + 1e-3)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df_feat = add_ratio_features(df)
    return df_feat
