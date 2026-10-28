"""Code-verification demo on synthetic data (NOT a result about Abuja).

    python scripts/run_synthetic_demo.py
"""

import tempfile
from pathlib import Path

from svb.pipeline import Params, run
from svb.synthetic import make_synthetic_scene


def main() -> None:
    out = Path("results/synthetic_demo")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        r, l = f"{tmp}/scene.tif", f"{tmp}/labels.tif"
        make_synthetic_scene(r, l, size=400, seed=0)
        res = run(
            Params(
                raster=r,
                labels=l,
                out_dir=str(out),
                n_per_class=600,
                block_sizes_m=[100, 250, 500, 1000],
                seeds=[0, 1],
                n_estimators=100,
                holdout_buffer_m=300.0,
            )
        )
    print(res["summary"].round(4).to_string(index=False))
    print("\nReference OA:", round(res["reference"]["oa"], 4))


if __name__ == "__main__":
    main()
