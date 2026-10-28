"""Download a Sentinel-2 dry-season composite and ESA WorldCover labels for a study area.

Writes two co-registered GeoTIFFs (same grid):
    <out>/s2_composite.tif   6 bands (B02,B03,B04,B08,B11,B12), surface reflectance 0-1
    <out>/labels.tif         WorldCover classes (uint8), 0 = nodata

STATUS: written but check it with --dry-run first. Collection/asset names and
band handling below follow Microsoft Planetary Computer conventions as I
understand them; verify on first run and fix anything that differs.

Label caveat: ESA WorldCover is itself a Sentinel-based classification. Using
it as labels means accuracies measure agreement with that product, not with
ground truth. That is acceptable for studying the *validation gap* between
random and spatially blocked CV, but must be stated in any write-up.

Reflectance scaling: scenes acquired before 2022-01-25 (processing baseline
< 04.00) need no BOA offset. For later scenes, subtract 1000 from the DN
before dividing by 10000. Defaults below use Jan-Feb 2021 to match the
WorldCover 2021 epoch and avoid the offset.

Example:
    python scripts/download_s2_worldcover.py --dry-run
    python scripts/download_s2_worldcover.py --out data/processed
"""

import argparse
import time
from pathlib import Path

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
# Abuja area (lon_min, lat_min, lon_max, lat_max); default is a ~20 km box. Adjust.
DEFAULT_BBOX = (7.35, 8.95, 7.55, 9.15)
EPSG = 32632  # UTM zone 32N covers Abuja
WORLDCOVER_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}


def open_catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def search_s2(catalog, bbox, date_range, max_cloud):
    return catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()


def compute_with_retries(arr, retries: int, wait_s: float):
    """Compute a dask-backed array, retrying on truncated/failed remote reads.

    Reads over a flaky link can return short tiles (rasterio 'TIFFFillTile: got X bytes,
    expected Y'). Single-threaded compute plus whole-array retries is crude but robust
    for a ~20 km box.
    """
    last = None
    for attempt in range(1, retries + 1):
        try:
            return arr.compute(scheduler="synchronous")
        except Exception as e:  # noqa: BLE001 - network/rasterio errors vary
            last = e
            print(f"  attempt {attempt}/{retries} failed: {type(e).__name__}: {str(e)[:160]}")
            if attempt < retries:
                time.sleep(wait_s * attempt)
    raise RuntimeError(f"compute failed after {retries} attempts") from last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX))
    ap.add_argument("--dates", default="2021-01-01/2021-02-28")
    ap.add_argument("--max-cloud", type=float, default=30.0)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--retries", type=int, default=5, help="retries for failed remote reads")
    ap.add_argument("--retry-wait", type=float, default=10.0, help="seconds, multiplied by attempt number")
    ap.add_argument("--dry-run", action="store_true", help="only count scenes / check access")
    a = ap.parse_args()

    catalog = open_catalog()
    items = search_s2(catalog, a.bbox, a.dates, a.max_cloud)
    print(f"Sentinel-2 L2A scenes found: {len(items)} (cloud < {a.max_cloud}%, {a.dates})")
    wc = catalog.search(collections=["esa-worldcover"], bbox=list(a.bbox), datetime="2021-01-01/2021-12-31").item_collection()
    print(f"ESA WorldCover items found: {len(wc)}")
    if a.dry_run or len(items) == 0 or len(wc) == 0:
        return

    import numpy as np
    import rasterio
    import stackstac

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    stack = stackstac.stack(
        items,
        assets=BANDS + ["SCL"],
        bounds_latlon=tuple(a.bbox),
        resolution=10,
        epsg=EPSG,
        dtype="float64",
        fill_value=np.nan,
        rescale=False,
        chunksize=512,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
            always=dict(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=2, GDAL_HTTP_TIMEOUT=60)
        ),
    )
    scl = stack.sel(band="SCL")
    valid = scl.isin([4, 5, 6, 7])  # vegetation, bare, water, unclassified
    refl = stack.sel(band=BANDS).where(valid) / 10000.0
    comp = compute_with_retries(refl.median(dim="time", skipna=True).clip(0, 1), a.retries, a.retry_wait)
    comp = comp.sortby("y", ascending=False)

    x, y = comp.x.values, comp.y.values
    transform = rasterio.transform.from_origin(float(x.min()) - 5, float(y.max()) + 5, 10, 10)
    data = comp.transpose("band", "y", "x").values.astype("float32")
    with rasterio.open(
        out / "s2_composite.tif", "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=len(BANDS), dtype="float32", crs=f"EPSG:{EPSG}", transform=transform, nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(data)
        for i, b in enumerate(BANDS, start=1):
            dst.set_band_description(i, b)

    lab = stackstac.stack(
        wc, assets=["map"], bounds_latlon=tuple(a.bbox), resolution=10, epsg=EPSG,
        dtype="float64", fill_value=0,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
            always=dict(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=2, GDAL_HTTP_TIMEOUT=60)
        ),
    )
    lab = compute_with_retries(lab.isel(time=0).squeeze().sortby("y", ascending=False), a.retries, a.retry_wait)
    lab_arr = np.nan_to_num(lab.values, nan=0).astype("uint8")
    if lab_arr.shape != data.shape[1:]:
        raise RuntimeError(f"Grid mismatch: labels {lab_arr.shape} vs composite {data.shape[1:]}")
    with rasterio.open(
        out / "labels.tif", "w", driver="GTiff", height=lab_arr.shape[0], width=lab_arr.shape[1],
        count=1, dtype="uint8", crs=f"EPSG:{EPSG}", transform=transform, nodata=0, compress="deflate",
    ) as dst:
        dst.write(lab_arr, 1)
    print("Wrote", out / "s2_composite.tif", "and", out / "labels.tif")
    print("WorldCover classes:", WORLDCOVER_CLASSES)


if __name__ == "__main__":
    main()