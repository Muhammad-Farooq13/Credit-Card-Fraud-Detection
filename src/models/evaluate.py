from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc


def load_npz(path: str):
    data = np.load(path)
    return data["X"], data["y"]


def evaluate_model(model_path: str, report_path: str) -> Dict[str, Any]:
    model = joblib.load(model_path)

    # For demo purposes, expect sibling test.npz relative to model by default
    default_test = Path(model_path).parent / "test.npz"
    X_test, y_test = load_npz(str(default_test))

    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    report = classification_report(y_test, y_prob > 0.5, output_dict=True)

    payload = {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "classification_report": report,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
