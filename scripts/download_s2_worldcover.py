"""Download a Sentinel-2 dry-season composite and ESA WorldCover labels (resumable).

Writes two co-registered GeoTIFFs on the same 10 m UTM grid:
    <out>/s2_composite.tif   6 bands (B02,B03,B04,B08,B11,B12), reflectance 0-1
    <out>/labels.tif         WorldCover classes (uint8), 0 = nodata

Designed for unreliable connections (e.g. Azure blob storage from China):
  * each band of each scene is read and retried separately;
  * every successful band is cached in <out>/cache, so re-running resumes;
  * it stops after --max-scenes good scenes (clearest first).

Label caveat: WorldCover is itself Sentinel-based, so accuracy against it
measures agreement with that product, not with ground truth.

Example:
    python scripts/download_s2_worldcover.py --dry-run
    python scripts/download_s2_worldcover.py --out data/processed
    (if it stops, just run the same command again)
"""

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
DEFAULT_BBOX = (7.35, 8.95, 7.55, 9.15)  # lon_min, lat_min, lon_max, lat_max
EPSG = 32632
RES = 10.0
BOA_CUTOFF = datetime(2022, 1, 25, tzinfo=timezone.utc)
WORLDCOVER_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}

GDAL_OPTS = dict(
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="3",
    GDAL_HTTP_TIMEOUT="120",
    GDAL_HTTP_CONNECTTIMEOUT="30",
    CPL_VSIL_CURL_USE_HEAD="NO",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="100000000",
    GDAL_HTTP_MULTIPLEX="NO",
)


def open_catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def search_s2(catalog, bbox, dates, max_cloud):
    return list(catalog.search(
        collections=["sentinel-2-l2a"], bbox=list(bbox), datetime=dates,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection())


def make_grid(bbox):
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    l, b, r, t = transform_bounds("EPSG:4326", f"EPSG:{EPSG}", *bbox)
    l, b = math.floor(l / RES) * RES, math.floor(b / RES) * RES
    r, t = math.ceil(r / RES) * RES, math.ceil(t / RES) * RES
    return from_origin(l, t, RES, RES), int((t - b) / RES), int((r - l) / RES)


def read_on_grid(href, grid, resampling):
    import rasterio
    from rasterio.vrt import WarpedVRT

    transform, height, width = grid
    with rasterio.Env(**GDAL_OPTS):
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=f"EPSG:{EPSG}", transform=transform, width=width,
                           height=height, resampling=resampling) as vrt:
                return vrt.read(1)


def read_with_retries(fetch_href, grid, resampling, attempts, wait, label):
    """fetch_href() returns a freshly signed URL; retried per band."""
    for k in range(1, attempts + 1):
        try:
            return read_on_grid(fetch_href(), grid, resampling)
        except Exception as e:  # noqa: BLE001 - network errors vary
            print(f"    {label} attempt {k}/{attempts} failed: {type(e).__name__}: {str(e)[:90]}")
            if k < attempts:
                time.sleep(wait * k)
    return None


def load_scene(catalog, item_id, grid, cache, attempts, wait):
    """Return (6, H, W) reflectance for one scene, or None. Uses/creates the cache."""
    import numpy as np
    import planetary_computer
    from rasterio.enums import Resampling

    col = catalog.get_collection("sentinel-2-l2a")

    def href_for(asset):
        def f():
            return planetary_computer.sign(col.get_item(item_id)).assets[asset].href
        return f

    def cached(asset, resampling):
        path = cache / f"{item_id}_{asset}.npy"
        if path.exists():
            return np.load(path)
        arr = read_with_retries(href_for(asset), grid, resampling, attempts, wait, f"{asset}")
        if arr is not None:
            np.save(path, arr)
        return arr

    scl = cached("SCL", Resampling.nearest)
    if scl is None:
        return None
    dt = col.get_item(item_id).datetime
    offset = 1000.0 if dt and dt >= BOA_CUTOFF else 0.0
    valid = np.isin(scl, [4, 5, 6, 7])
    out = []
    for b in BANDS:
        dn = cached(b, Resampling.bilinear)
        if dn is None:
            return None
        arr = dn.astype("float32")
        arr[dn == 0] = np.nan
        arr = (arr - offset) / 10000.0
        arr[~valid] = np.nan
        out.append(arr)
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX))
    ap.add_argument("--dates", default="2021-01-01/2021-02-28")
    ap.add_argument("--max-cloud", type=float, default=30.0)
    ap.add_argument("--max-scenes", type=int, default=8, help="stop after this many good scenes")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--retries", type=int, default=6, help="attempts per band")
    ap.add_argument("--retry-wait", type=float, default=5.0, help="seconds x attempt number")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    catalog = open_catalog()
    items = search_s2(catalog, a.bbox, a.dates, a.max_cloud)
    wc = list(catalog.search(collections=["esa-worldcover"], bbox=list(a.bbox),
                             datetime="2021-01-01/2021-12-31").item_collection())
    print(f"Sentinel-2 scenes: {len(items)} | WorldCover items: {len(wc)}")
    if a.dry_run or not items or not wc:
        return

    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    out = Path(a.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    grid = make_grid(a.bbox)
    transform, height, width = grid
    print(f"Target grid: {height} x {width} pixels at {RES:.0f} m (EPSG:{EPSG})")

    # Labels first: small, and a rerun reuses the cache.
    lab_path = cache / "labels.npy"
    if lab_path.exists():
        lab = np.load(lab_path)
    else:
        print("Reading WorldCover labels...")
        lab = None
        for it in wc:
            tile_id = it.id

            def f(tile_id=tile_id):
                import planetary_computer
                return planetary_computer.sign(
                    catalog.get_collection("esa-worldcover").get_item(tile_id)).assets["map"].href

            arr = read_with_retries(f, grid, Resampling.nearest, a.retries, a.retry_wait, f"label {tile_id[-8:]}")
            if arr is None:
                raise RuntimeError("WorldCover label download failed; rerun to resume.")
            lab = arr if lab is None else np.where(lab == 0, arr, lab)
        np.save(lab_path, lab)
    lab = lab.astype("uint8")

    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    scenes = []
    for it in items:
        if len(scenes) >= a.max_scenes:
            break
        print(f"Scene {it.id} (cloud {it.properties.get('eo:cloud_cover'):.1f}%)")
        arr = load_scene(catalog, it.id, grid, cache, a.retries, a.retry_wait)
        if arr is None or not np.isfinite(arr).any():
            print("  skipped (rerun later to retry it; finished bands are cached)")
            continue
        scenes.append(arr)
        print(f"  ok ({len(scenes)}/{a.max_scenes})")

    if len(scenes) < 3:
        print(f"Only {len(scenes)} good scene(s). Rerun the same command to resume; "
              "not writing a composite from fewer than 3 scenes.")
        return

    with np.errstate(all="ignore"):
        comp = np.clip(np.nanmedian(np.stack(scenes), axis=0), 0, 1).astype("float32")
    prof = dict(driver="GTiff", height=height, width=width, crs=f"EPSG:{EPSG}",
                transform=transform, compress="deflate")
    with rasterio.open(out / "s2_composite.tif", "w", count=len(BANDS), dtype="float32",
                       nodata=np.nan, **prof) as dst:
        dst.write(comp)
        for i, b in enumerate(BANDS, 1):
            dst.set_band_description(i, b)
    with rasterio.open(out / "labels.tif", "w", count=1, dtype="uint8", nodata=0, **prof) as dst:
        dst.write(lab, 1)
    print(f"Wrote {out/'s2_composite.tif'} and {out/'labels.tif'} from {len(scenes)} scenes")
    print("WorldCover classes:", WORLDCOVER_CLASSES)


if __name__ == "__main__":
    main()