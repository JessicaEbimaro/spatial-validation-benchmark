import json

from svb.pipeline import Params, run
from svb.synthetic import make_synthetic_scene


def test_pipeline_runs_end_to_end_and_random_cv_is_optimistic(tmp_path):
    r, l = str(tmp_path / "s.tif"), str(tmp_path / "l.tif")
    make_synthetic_scene(r, l, size=300, seed=0)
    out = tmp_path / "res"
    res = run(
        Params(
            raster=r,
            labels=l,
            out_dir=str(out),
            n_per_class=400,
            block_sizes_m=[250, 500, 1000],
            seeds=[0, 1],
            n_estimators=60,
            holdout_buffer_m=300.0,
        )
    )
    for name in ["cv_results_all_runs.csv", "cv_summary.csv", "oa_vs_block_size.png", "run_metadata.json"]:
        assert (out / name).exists()
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["n_pool"] > 0 and meta["n_region"] > 0

    s = res["summary"]
    random_oa = s.loc[s.scheme == "random", "oa_mean"].iloc[0]
    block_oa = s.loc[s.scheme == "block", "oa_mean"]
    # On this synthetic scene the shared site effect makes random CV optimistic.
    assert random_oa > block_oa.min()
