"""Raster I/O and stratified point sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import xy


@dataclass
class Raster:
    data: np.ndarray  # (bands, height, width), float32, NaN = nodata
    transform: object
    crs: object
    band_names: List[str]


def read_raster(path: str) -> Raster:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        names = [d if d else f"band{i + 1}" for i, d in enumerate(src.descriptions)]
        return Raster(data, src.transform, src.crs, names)


def read_labels(path: str) -> Tuple[np.ndarray, object, object]:
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def write_raster(
    path: str,
    data: np.ndarray,
    transform,
    crs,
    band_names: Optional[List[str]] = None,
    nodata: Optional[float] = None,
) -> None:
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(data)
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)


def sample_points(
    labels: np.ndarray,
    transform,
    n_per_class: int,
    spacing_px: int = 1,
    seed: int = 0,
    nodata: int = 0,
    valid_mask: Optional[np.ndarray] = None,
):
    """Stratified random sample of label pixels.

    ``spacing_px > 1`` keeps at most one candidate per ``spacing_px`` x
    ``spacing_px`` cell (per class), chosen at random within the cell. This
    controls how densely neighbouring samples sit, which directly affects how
    optimistic random validation is. Note: it limits density but does not
    guarantee an exact minimum distance between points in adjacent cells.

    Returns ``(rows, cols, coords, y)`` where ``coords`` are pixel-centre
    coordinates in the raster's CRS (must be projected, in metres).
    """
    rng = np.random.default_rng(seed)
    valid = labels != nodata
    if valid_mask is not None:
        valid &= valid_mask
    all_r, all_c = [], []
    for cls in np.unique(labels[valid]):
        r, c = np.nonzero((labels == cls) & valid)
        if len(r) == 0:
            continue
        order = rng.permutation(len(r))
        r, c = r[order], c[order]
        if spacing_px > 1:
            cell = (r // spacing_px).astype(np.int64) * 10_000_000 + (c // spacing_px)
            _, first = np.unique(cell, return_index=True)
            first = np.sort(first)
            r, c = r[first], c[first]
        k = min(n_per_class, len(r))
        all_r.append(r[:k])
        all_c.append(c[:k])
    rows = np.concatenate(all_r)
    cols = np.concatenate(all_c)
    xs, ys = xy(transform, rows, cols, offset="center")
    coords = np.column_stack([np.asarray(xs), np.asarray(ys)])
    return rows, cols, coords, labels[rows, cols]
