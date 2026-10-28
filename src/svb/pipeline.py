"""End-to-end benchmark pipeline: composite + label rasters -> results.

Steps: sample points -> features -> semivariogram range -> reference region
holdout -> random vs blocked CV at several block sizes -> tables, figure,
and a JSON record of parameters and package versions.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from .experiment import reference_accuracy, run_gap_experiment, summarise  # noqa: E402
from .features import build_feature_matrix, finite_rows  # noqa: E402
from .io import read_labels, read_raster, sample_points  # noqa: E402
from .splits import region_holdout  # noqa: E402
from .variogram import empirical_semivariogram, estimate_range  # noqa: E402


@dataclass
class Params:
    raster: str
    labels: str
    out_dir: str = "results"
    n_per_class: int = 500
    spacing_px: int = 1
    block_sizes_m: List[float] = field(default_factory=lambda: [100, 250, 500, 1000, 2000])
    buffer_m: float = 0.0
    n_splits: int = 5
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    holdout_frac: float = 0.25
    holdout_buffer_m: float = 500.0
    n_estimators: int = 200
    sample_seed: int = 0


def run(p: Params) -> dict:
    out = Path(p.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raster = read_raster(p.raster)
    labels, l_transform, _ = read_labels(p.labels)
    if labels.shape != raster.data.shape[1:]:
        raise ValueError("Label raster and composite must share the same grid/shape.")
    if not raster.crs.is_projected:
        raise ValueError("Raster CRS must be projected (metres).")

    rows, cols, coords, y = sample_points(
        labels, l_transform, p.n_per_class, p.spacing_px, p.sample_seed
    )
    X, feat_names = build_feature_matrix(raster, rows, cols)
    ok = finite_rows(X)
    X, y, coords = X[ok], y[ok], coords[ok]
    classes = np.unique(y)

    # Autocorrelation screening on the first principal component of the features.
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, _, vt = np.linalg.svd(Xs, full_matrices=False)
    pc1 = Xs @ vt[0]
    lags, gamma = empirical_semivariogram(coords, pc1)
    est_range = estimate_range(lags, gamma)

    pool, region = region_holdout(coords, p.holdout_frac, axis=0, buffer=p.holdout_buffer_m)
    ref = reference_accuracy(X, y, pool, region, classes, seed=0, n_estimators=p.n_estimators)

    results = run_gap_experiment(
        X[pool], y[pool], coords[pool], p.block_sizes_m, p.n_splits, p.buffer_m, p.seeds, p.n_estimators
    )
    summary = summarise(results, reference_oa=ref["oa"])

    results.to_csv(out / "cv_results_all_runs.csv", index=False)
    summary.to_csv(out / "cv_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    blk = summary[summary.scheme == "block"].sort_values("block_size_m")
    rnd = summary[summary.scheme == "random"].iloc[0]
    ax.errorbar(blk.block_size_m, blk.oa_mean, yerr=blk.oa_sd.fillna(0), marker="o", label="Spatial block CV")
    ax.axhline(rnd.oa_mean, linestyle="--", label="Random CV")
    ax.axhline(ref["oa"], linestyle=":", color="k", label="Held-out region (reference)")
    if np.isfinite(est_range):
        ax.axvline(est_range, color="grey", alpha=0.5, label=f"Est. range ~{est_range:.0f} m")
    ax.set_xscale("log")
    ax.set_xlabel("Block size (m)")
    ax.set_ylabel("Overall accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "oa_vs_block_size.png", dpi=200)
    plt.close(fig)

    meta = {
        "params": asdict(p),
        "n_points_used": int(len(y)),
        "n_pool": int(len(pool)),
        "n_region": int(len(region)),
        "classes": [int(c) for c in classes],
        "features": feat_names,
        "estimated_autocorrelation_range_m": None if not np.isfinite(est_range) else float(est_range),
        "reference_region_metrics": ref,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return {"summary": summary, "reference": ref, "est_range_m": est_range, "meta": meta}
