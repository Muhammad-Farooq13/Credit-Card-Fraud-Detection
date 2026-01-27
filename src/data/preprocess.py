from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1


def preprocess_dataset(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """Clean, split, scale, and persist datasets and preprocessors."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Shuffle for randomness
    df = shuffle(df, random_state=SEED)

    # Split features/target
    if "Class" not in df.columns:
        raise KeyError("Expected 'Class' column as target")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train/val/test split with stratification when feasible; fallback to unstratified for tiny samples
    stratify_main = y if y.value_counts().min() >= 2 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, stratify=stratify_main, random_state=SEED
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    stratify_temp = y_temp if y_temp.value_counts().min() >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=stratify_temp, random_state=SEED
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Persist datasets
    train_path = output / "train.npz"
    val_path = output / "val.npz"
    test_path = output / "test.npz"
    np.savez(train_path, X=X_train_s, y=y_train.values)
    np.savez(val_path, X=X_val_s, y=y_val.values)
    np.savez(test_path, X=X_test_s, y=y_test.values)

    # Persist preprocessor
    preproc_path = output / "preprocessor.joblib"
    joblib.dump(scaler, preproc_path)

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "preprocessor": str(preproc_path),
    }
