"""Cross-validation splitters.

All coordinates must be in a *projected* CRS with units of metres, so that
block sizes and buffers are expressed in metres.

Two schemes are provided:

* ``random_kfold``: the conventional pixel-level random split (the thing this
  project argues is optimistic for spatially autocorrelated data).
* ``spatial_block_kfold``: points are grouped into square blocks, whole blocks
  are assigned to folds, and (optionally) training points closer than
  ``buffer`` metres to any test point are dropped, so that no training point is
  within ``buffer`` metres of a test point.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from scipy.spatial import cKDTree

Split = Tuple[np.ndarray, np.ndarray]


def random_kfold(n: int, n_splits: int = 5, seed: int = 0) -> Iterator[Split]:
    """Plain random K-fold over ``n`` samples (every sample tested once)."""
    if n_splits < 2 or n < n_splits:
        raise ValueError("Need n_splits >= 2 and n >= n_splits.")
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_splits)
    for k in range(n_splits):
        test = np.sort(folds[k])
        train = np.sort(np.concatenate([folds[j] for j in range(n_splits) if j != k]))
        yield train, test


def assign_blocks(coords: np.ndarray, block_size: float) -> np.ndarray:
    """Return an integer block id for every point (square grid, metres)."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    coords = np.asarray(coords, dtype=float)
    ij = np.floor((coords - coords.min(axis=0)) / block_size).astype(np.int64)
    _, block_id = np.unique(ij, axis=0, return_inverse=True)
    return block_id.reshape(-1)


def spatial_block_kfold(
    coords: np.ndarray,
    block_size: float,
    n_splits: int = 5,
    buffer: float = 0.0,
    seed: int = 0,
) -> Iterator[Split]:
    """Spatially blocked K-fold with an optional training buffer.

    Whole blocks go to a single fold. With ``buffer > 0`` any training point
    within ``buffer`` metres of a test point is removed from that fold's
    training set, which guarantees a minimum train/test separation.
    """
    coords = np.asarray(coords, dtype=float)
    blocks = assign_blocks(coords, block_size)
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) < n_splits:
        raise ValueError(
            f"Only {len(unique_blocks)} blocks of size {block_size} m for "
            f"{n_splits} folds; use a smaller block size."
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_blocks)
    fold_of_block = np.empty(unique_blocks.max() + 1, dtype=np.int64)
    for i, b in enumerate(shuffled):
        fold_of_block[b] = i % n_splits
    fold = fold_of_block[blocks]

    for k in range(n_splits):
        test = np.flatnonzero(fold == k)
        train = np.flatnonzero(fold != k)
        if buffer > 0 and len(test) and len(train):
            dist, _ = cKDTree(coords[test]).query(coords[train], k=1)
            train = train[dist >= buffer]
        yield train, test


def min_train_test_distance(coords: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Smallest distance (m) from any test point to its nearest training point."""
    if len(train) == 0 or len(test) == 0:
        return float("inf")
    dist, _ = cKDTree(np.asarray(coords)[train]).query(np.asarray(coords)[test], k=1)
    return float(dist.min())


def region_holdout(
    coords: np.ndarray, frac: float = 0.25, axis: int = 0, buffer: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out one contiguous strip of the study area as a reference test region.

    Returns ``(pool_idx, region_idx)``. Pool points within ``buffer`` metres of
    the region are dropped so the reference test is spatially independent.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1).")
    coords = np.asarray(coords, dtype=float)
    cut = np.quantile(coords[:, axis], 1.0 - frac)
    region = np.flatnonzero(coords[:, axis] >= cut)
    pool = np.flatnonzero(coords[:, axis] < cut)
    if buffer > 0 and len(region) and len(pool):
        dist, _ = cKDTree(coords[region]).query(coords[pool], k=1)
        pool = pool[dist >= buffer]
    return pool, region
