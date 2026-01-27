from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


SEED = 42


def load_npz(path: str):
    data = np.load(path)
    return data["X"], data["y"]


def train_model(config: Dict[str, Any]) -> str:
    """Train a simple classifier and persist it."""
    train_path = config["train_path"]
    val_path = config.get("val_path")
    output_path = config.get("output_path", "artifacts/model.joblib")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_npz(train_path)
    if val_path:
        X_val, y_val = load_npz(val_path)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight_dict,
    )
    model.fit(X_train, y_train)

    # Simple validation
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    print(json.dumps({"val_auc": auc}))

    joblib.dump(model, output_path)
    return output_path
