"""Random vs spatially blocked cross-validation experiments."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .models import (
    classification_metrics,
    expected_calibration_error,
    make_rf,
    predictive_entropy,
)
from .splits import random_kfold, spatial_block_kfold


def cross_validate(X, y, splits, classes, seed: int = 0, n_estimators: int = 200) -> dict:
    """Pooled out-of-fold metrics (plus ECE and mean entropy) for given splits."""
    n = len(y)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = np.zeros((n, len(classes)), dtype=float)
    covered = np.zeros(n, dtype=bool)
    n_train = []
    for train, test in splits:
        if len(train) == 0 or len(test) == 0:
            continue
        model = make_rf(seed, n_estimators)
        model.fit(X[train], y[train])
        proba = model.predict_proba(X[test])
        cols = np.searchsorted(classes, model.classes_)
        full = np.zeros((len(test), len(classes)))
        full[:, cols] = proba
        oof_proba[test] = full
        oof_pred[test] = classes[full.argmax(axis=1)]
        covered[test] = True
        n_train.append(len(train))
    if not covered.any():
        raise RuntimeError("No usable folds.")
    m = classification_metrics(y[covered], oof_pred[covered], classes)
    y_idx = np.searchsorted(classes, y[covered])
    m["ece"] = expected_calibration_error(oof_proba[covered], y_idx)
    m["mean_entropy"] = float(predictive_entropy(oof_proba[covered]).mean())
    m["n_test"] = int(covered.sum())
    m["mean_n_train"] = float(np.mean(n_train))
    return m


def run_gap_experiment(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    block_sizes_m: Sequence[float],
    n_splits: int = 5,
    buffer_m: float = 0.0,
    seeds: Iterable[int] = (0, 1, 2),
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Evaluate random K-fold and spatial block K-fold at several block sizes."""
    classes = np.unique(y)
    rows: List[dict] = []
    for seed in seeds:
        res = cross_validate(
            X, y, random_kfold(len(y), n_splits, seed), classes, seed, n_estimators
        )
        rows.append({"scheme": "random", "block_size_m": 0.0, "buffer_m": 0.0, "seed": seed, **res})
        for bs in block_sizes_m:
            try:
                splits = list(spatial_block_kfold(coords, bs, n_splits, buffer_m, seed))
            except ValueError:
                continue  # too few blocks at this size
            res = cross_validate(X, y, splits, classes, seed, n_estimators)
            rows.append(
                {"scheme": "block", "block_size_m": float(bs), "buffer_m": float(buffer_m), "seed": seed, **res}
            )
    return pd.DataFrame(rows)


def reference_accuracy(X, y, pool_idx, region_idx, classes, seed: int = 0, n_estimators: int = 200) -> dict:
    """Train on the pool, test on a spatially separate region (the reference estimate)."""
    model = make_rf(seed, n_estimators)
    model.fit(X[pool_idx], y[pool_idx])
    pred = model.predict(X[region_idx])
    return classification_metrics(y[region_idx], pred, classes)


def summarise(results: pd.DataFrame, reference_oa: float | None = None) -> pd.DataFrame:
    """Mean/SD of OA and macro-F1 per scheme+block size, and gap vs random CV."""
    g = (
        results.groupby(["scheme", "block_size_m"])
        .agg(
            oa_mean=("oa", "mean"),
            oa_sd=("oa", "std"),
            f1_mean=("macro_f1", "mean"),
            ece_mean=("ece", "mean"),
            entropy_mean=("mean_entropy", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    random_oa = g.loc[g.scheme == "random", "oa_mean"].iloc[0]
    g["gap_vs_random_oa"] = random_oa - g["oa_mean"]
    if reference_oa is not None:
        g["reference_oa"] = reference_oa
        g["abs_err_vs_reference"] = (g["oa_mean"] - reference_oa).abs()
    return g
