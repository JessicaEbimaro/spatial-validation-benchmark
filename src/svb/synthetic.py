"""Synthetic spatially autocorrelated imagery for testing the pipeline.

This exists only to verify that the code runs and that the splitters behave.
Results on synthetic data say nothing about real land cover in Abuja.

The synthetic scene has (a) a spatially smooth class map, and (b) a smooth
"site effect" added to all bands, so that nearby pixels share a spectral
offset. A random split can exploit that shared offset; a spatially separated
split cannot.
"""

from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from .features import BASE_BANDS
from .io import write_raster


def make_synthetic_scene(
    path_raster: str,
    path_labels: str,
    size: int = 400,
    n_classes: int = 4,
    seed: int = 0,
    class_smoothness: float = 25.0,
    site_effect_smoothness: float = 40.0,
    site_effect_strength: float = 0.06,
    noise: float = 0.03,
) -> None:
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=(size, size)), class_smoothness)
    qs = np.quantile(field, np.linspace(0, 1, n_classes + 1)[1:-1])
    labels = (np.digitize(field, qs) + 1).astype("uint8")  # classes 1..K

    base = np.array(
        [
            [0.04, 0.07, 0.04, 0.35, 0.18, 0.09],  # vegetation-like
            [0.10, 0.13, 0.15, 0.22, 0.30, 0.25],  # built-up-like
            [0.15, 0.20, 0.25, 0.30, 0.38, 0.32],  # bare-like
            [0.06, 0.08, 0.05, 0.03, 0.02, 0.01],  # water-like
            [0.05, 0.09, 0.07, 0.28, 0.20, 0.12],
            [0.08, 0.11, 0.10, 0.26, 0.24, 0.17],
        ]
    )[:n_classes]

    bands = np.zeros((len(BASE_BANDS), size, size), dtype="float32")
    for b in range(len(BASE_BANDS)):
        site = gaussian_filter(rng.normal(size=(size, size)), site_effect_smoothness)
        site = site / (site.std() + 1e-9) * site_effect_strength
        pix = rng.normal(scale=noise, size=(size, size))
        bands[b] = base[labels - 1, b] + site + pix
    bands = np.clip(bands, 0.001, 1.0)

    transform = from_origin(400_000.0, 1_000_000.0, 10.0, 10.0)
    crs = CRS.from_epsg(32632)
    write_raster(path_raster, bands, transform, crs, band_names=list(BASE_BANDS))
    write_raster(path_labels, labels, transform, crs, nodata=0)
