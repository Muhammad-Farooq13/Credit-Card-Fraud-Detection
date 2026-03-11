"""Tests for the preprocessing pipeline."""
import os

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess_dataset


def _make_df(n: int = 40) -> pd.DataFrame:
    """Return a synthetic credit-card DataFrame large enough for a 3-way split.

    Uses n=40 by default: 80% / 10% / 10% gives 32 / 4 / 4 rows — enough for
    stratified splitting when each class has >= 2 representatives in every
    partition.
    """
    rng = np.random.default_rng(seed=42)
    n_fraud = max(4, n // 10)          # ~10% fraud, at least 4 samples
    n_normal = n - n_fraud
    return pd.DataFrame(
        {
            "Time": np.arange(n),
            "Amount": rng.lognormal(mean=3, sigma=1, size=n),
            "Class": np.array([0] * n_normal + [1] * n_fraud, dtype=int),
        }
    )


def test_preprocess_outputs(tmp_path):
    """Preprocessing should produce train/val/test .npz files and a scaler."""
    df = _make_df(40)
    artifacts = preprocess_dataset(df, tmp_path)

    # All expected keys are present
    assert set(artifacts.keys()) == {"train", "val", "test", "preprocessor"}

    # Files actually exist on disk
    for key in ("train", "val", "test", "preprocessor"):
        assert os.path.exists(artifacts[key]), f"Missing artifact: {key}"

    # Train split has correct array keys
    data = np.load(artifacts["train"])
    assert "X" in data and "y" in data, "train.npz must contain 'X' and 'y' arrays"


def test_preprocess_shapes_consistent(tmp_path):
    """Feature count must be the same across all splits."""
    df = _make_df(60)
    artifacts = preprocess_dataset(df, tmp_path)

    train = np.load(artifacts["train"])
    val = np.load(artifacts["val"])
    test = np.load(artifacts["test"])

    n_features = train["X"].shape[1]
    assert val["X"].shape[1] == n_features
    assert test["X"].shape[1] == n_features


def test_preprocess_labels_binary(tmp_path):
    """Labels must only be 0 or 1 after preprocessing."""
    df = _make_df(40)
    artifacts = preprocess_dataset(df, tmp_path)
    for split in ("train", "val", "test"):
        y = np.load(artifacts[split])["y"]
        assert set(y.tolist()).issubset({0, 1}), f"{split} contains non-binary labels"


def test_preprocess_no_data_leakage(tmp_path):
    """Scaler must be fit on the training set only.

    Verify that the StandardScaler's mean was not computed from the full dataset
    by checking that the training-set mean of each feature is (approximately)
    zero after scaling — which holds if and only if the scaler was fit on the
    training partition.
    """
    df = _make_df(60)
    artifacts = preprocess_dataset(df, tmp_path)
    X_train = np.load(artifacts["train"])["X"]
    # After fitting StandardScaler on training data, the training mean ≈ 0
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6), (
        "Training features should have ~zero mean after scaling "
        "(scaler was fit on the training set only)"
    )
