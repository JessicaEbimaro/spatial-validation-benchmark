"""Empirical semivariogram, used to choose defensible block sizes."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def empirical_semivariogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 20,
    max_dist: Optional[float] = None,
    n_pairs: int = 300_000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo estimate of the semivariogram from random point pairs.

    Returns ``(lag_centres, gamma)``; bins with no pairs are NaN.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    g = 0.5 * (values[i] - values[j]) ** 2
    if max_dist is None:
        max_dist = float(np.quantile(d, 0.5))
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(d, edges) - 1
    gamma = np.full(n_lags, np.nan)
    for b in range(n_lags):
        m = idx == b
        if m.sum() >= 30:
            gamma[b] = g[m].mean()
    return centres, gamma


def estimate_range(lags: np.ndarray, gamma: np.ndarray, frac: float = 0.95) -> float:
    """Crude effective range: first lag where gamma reaches ``frac`` of the sill.

    The sill is the median of the last third of valid lags. If the curve never
    reaches that level, the largest lag is returned (range >= that value).
    This is a screening heuristic, not a fitted variogram model.
    """
    ok = np.isfinite(gamma)
    lags, gamma = lags[ok], gamma[ok]
    if len(lags) < 6:
        return float("nan")
    sill = float(np.median(gamma[-max(3, len(gamma) // 3):]))
    hit = np.flatnonzero(gamma >= frac * sill)
    return float(lags[hit[0]]) if len(hit) else float(lags[-1])
