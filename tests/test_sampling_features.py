import numpy as np

from svb.features import BASE_BANDS, build_feature_matrix
from svb.io import read_labels, read_raster, sample_points
from svb.synthetic import make_synthetic_scene
from svb.variogram import empirical_semivariogram, estimate_range


def _scene(tmp_path, **kw):
    r, l = str(tmp_path / "s.tif"), str(tmp_path / "l.tif")
    make_synthetic_scene(r, l, size=200, seed=1, **kw)
    return r, l


def test_sampling_is_stratified_and_inside_raster(tmp_path):
    r, l = _scene(tmp_path)
    labels, tr, _ = read_labels(l)
    rows, cols, coords, y = sample_points(labels, tr, n_per_class=100, seed=0)
    for c in np.unique(labels):
        assert (y == c).sum() == 100
    assert rows.max() < labels.shape[0] and cols.max() < labels.shape[1]
    assert (labels[rows, cols] == y).all()
    assert coords.shape == (len(y), 2)


def test_spacing_reduces_local_density(tmp_path):
    r, l = _scene(tmp_path)
    labels, tr, _ = read_labels(l)
    rows, cols, *_ = sample_points(labels, tr, n_per_class=10_000, spacing_px=1, seed=0)
    rows2, cols2, *_ = sample_points(labels, tr, n_per_class=10_000, spacing_px=10, seed=0)
    assert len(rows2) < len(rows)
    cells = set(zip((rows2 // 10).tolist(), (cols2 // 10).tolist(), labels[rows2, cols2].tolist()))
    assert len(cells) == len(rows2)  # at most one point per cell per class


def test_feature_matrix_has_indices(tmp_path):
    r, l = _scene(tmp_path)
    ras = read_raster(r)
    labels, tr, _ = read_labels(l)
    rows, cols, _, _ = sample_points(labels, tr, 50, seed=0)
    X, names = build_feature_matrix(ras, rows, cols)
    assert names[: len(BASE_BANDS)] == BASE_BANDS
    assert {"NDVI", "NDBI", "MNDWI", "BSI"} <= set(names)
    assert X.shape == (len(rows), len(names))
    assert np.isfinite(X).all()


def test_semivariogram_range_is_larger_for_smoother_field():
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter

    def field(sigma):
        f = gaussian_filter(rng.normal(size=(300, 300)), sigma)
        return f / f.std()

    xs, ys = np.meshgrid(np.arange(300) * 10.0, np.arange(300) * 10.0)
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    ranges = []
    for sigma in (3, 15):
        v = field(sigma).ravel()
        lags, g = empirical_semivariogram(coords, v, n_lags=25, max_dist=1500, seed=0)
        ranges.append(estimate_range(lags, g))
    assert ranges[1] > ranges[0]
