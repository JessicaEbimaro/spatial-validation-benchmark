"""Feature construction from a Sentinel-2 surface-reflectance stack."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .io import Raster

EPS = 1e-9
# Band names expected in the composite (Sentinel-2 L2A):
#   B02 blue, B03 green, B04 red, B08 NIR, B11 SWIR1, B12 SWIR2
BASE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def spectral_indices(bands: dict) -> dict:
    """Return NDVI, NDBI, MNDWI and BSI computed from named band arrays."""
    out = {}
    if all(k in bands for k in ("B08", "B04")):
        out["NDVI"] = _nd(bands["B08"], bands["B04"])
    if all(k in bands for k in ("B11", "B08")):
        out["NDBI"] = _nd(bands["B11"], bands["B08"])
    if all(k in bands for k in ("B03", "B11")):
        out["MNDWI"] = _nd(bands["B03"], bands["B11"])
    if all(k in bands for k in ("B11", "B04", "B08", "B02")):
        num = (bands["B11"] + bands["B04"]) - (bands["B08"] + bands["B02"])
        den = (bands["B11"] + bands["B04"]) + (bands["B08"] + bands["B02"])
        out["BSI"] = num / (den + EPS)
    return out


def build_feature_matrix(
    raster: Raster, rows: np.ndarray, cols: np.ndarray, with_indices: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Extract per-point features (bands + indices). Drops no rows; NaNs are kept
    so the caller can decide how to handle them (see ``drop_nan_rows``)."""
    values = {name: raster.data[i, rows, cols] for i, name in enumerate(raster.band_names)}
    names = list(raster.band_names)
    cols_out = [values[n] for n in names]
    if with_indices:
        for name, arr in spectral_indices(values).items():
            names.append(name)
            cols_out.append(arr)
    return np.column_stack(cols_out).astype("float32"), names


def finite_rows(X: np.ndarray) -> np.ndarray:
    """Boolean mask of rows of ``X`` that contain no NaN/inf."""
    return np.isfinite(X).all(axis=1)
