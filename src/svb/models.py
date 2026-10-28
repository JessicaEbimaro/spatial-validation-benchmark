"""Model, metrics and uncertainty summaries."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score


def make_rf(seed: int = 0, n_estimators: int = 200) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=1
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict:
    out = {
        "oa": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
    }
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    for c, r in zip(classes, recalls):
        out[f"recall_c{int(c)}"] = float(r)
    return out


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of class-probability vectors, per sample."""
    p = np.clip(proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def expected_calibration_error(proba: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE. ``y_true_idx`` are column indices into ``proba``."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
