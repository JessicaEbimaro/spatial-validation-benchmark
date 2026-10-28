# Spatial Validation Benchmark

**How much does random validation overestimate GeoAI accuracy? A spatially blocked benchmark of Sentinel-2 land-cover classification around Abuja, Nigeria.**

> **Status: work in progress.** The code and tests are in place. No real results on Abuja data have been produced yet. The only numbers the repo can currently produce come from a synthetic scene (`scripts/run_synthetic_demo.py`), which is a code sanity check and says nothing about Abuja or about real-world accuracy.

## Research question

Random pixel-level k-fold cross-validation puts spatially adjacent, autocorrelated samples in both train and test sets, which inflates accuracy. This project quantifies that inflation for a Random Forest classifier on Sentinel-2 imagery around Abuja, and asks:

1. How large is the gap between random CV and spatially blocked CV (overall accuracy, macro-F1, per-class recall)?
2. How does the gap change with block size, relative to the spatial autocorrelation range from a semivariogram?
3. How far is each CV estimate from a held-out-region "reference" estimate?
4. Do top-label calibration (ECE) and predictive entropy change between schemes?

The general phenomenon is already documented in the literature (e.g. Karasiak et al., Machine Learning; Schadauer et al. 2024). The contribution here is a reproducible, open Abuja case study and benchmark code, not a new discovery.

## Design

- **Imagery:** Sentinel-2 L2A dry-season composite (default Jan–Feb 2021), bands B02, B03, B04, B08, B11, B12, SCL-based cloud mask, UTM zone 32N (EPSG:32632).
- **Features:** six bands plus NDVI, NDBI, MNDWI, BSI.
- **Labels:** ESA WorldCover 2021.
- **Sampling:** stratified by class with a minimum pixel spacing.
- **Validation schemes:** random k-fold; spatial block k-fold over several block sizes with a distance buffer between train and test points (guarantee checked by tests); region holdout as the reference.
- **Model:** Random Forest.
- **Block size choice:** informed by an empirical semivariogram range (heuristic).

## Limitations (read before citing anything)

- **Label circularity.** WorldCover is itself a Sentinel-based classification. Accuracy here is agreement with that product, not with ground truth. This is acceptable for studying the *gap* between validation schemes, but it must not be reported as map accuracy. An independent reference set from high-resolution imagery would strengthen the study.
- The region holdout is one reference estimate, not a truth oracle.
- The semivariogram range estimate is a heuristic.
- `scripts/download_s2_worldcover.py` has not been verified against the live Planetary Computer servers. Run it with `--dry-run` first and fix any collection or asset-name differences.
- A single study area and a single model family.

## Reproduce

```bash
pip install -e ".[download,dev]"
pytest                                          # 14 tests
python scripts/run_synthetic_demo.py            # code check on synthetic data only

python scripts/download_s2_worldcover.py --dry-run          # check access first
python scripts/download_s2_worldcover.py --out data/processed
python scripts/run_benchmark.py --help                      # see options
```

Outputs go to `results/`: `cv_results_all_runs.csv`, `cv_summary.csv`, `oa_vs_block_size.png`, `run_metadata.json`.

## Layout

```
src/svb/        splits, io, features, models, variogram, experiment, pipeline, synthetic
scripts/        download, run_benchmark, run_synthetic_demo
tests/          split/leakage tests, sampling/feature tests, end-to-end test
```

## License

Code: MIT. ESA WorldCover and Sentinel-2 data have their own licences; see their providers.
