"""Run the benchmark on any composite + label GeoTIFF pair.

Example:
    python scripts/run_benchmark.py --raster data/processed/abuja_s2_composite.tif \
        --labels data/processed/abuja_labels.tif --out results/abuja
"""

import argparse

from svb.pipeline import Params, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--spacing-px", type=int, default=1, help="one candidate per NxN-pixel cell")
    ap.add_argument("--block-sizes", default="100,250,500,1000,2000", help="metres, comma-separated")
    ap.add_argument("--buffer", type=float, default=0.0, help="training buffer around test blocks (m)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--holdout-buffer", type=float, default=500.0)
    ap.add_argument("--n-estimators", type=int, default=200)
    a = ap.parse_args()

    p = Params(
        raster=a.raster,
        labels=a.labels,
        out_dir=a.out,
        n_per_class=a.n_per_class,
        spacing_px=a.spacing_px,
        block_sizes_m=[float(x) for x in a.block_sizes.split(",")],
        buffer_m=a.buffer,
        n_splits=a.n_splits,
        seeds=[int(x) for x in a.seeds.split(",")],
        holdout_frac=a.holdout_frac,
        holdout_buffer_m=a.holdout_buffer,
        n_estimators=a.n_estimators,
    )
    res = run(p)
    print(res["summary"].round(4).to_string(index=False))
    print("\nReference (held-out region) OA:", round(res["reference"]["oa"], 4))
    print("Estimated autocorrelation range (m):", res["est_range_m"])


if __name__ == "__main__":
    main()
