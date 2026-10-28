import numpy as np
import pytest

from svb.splits import (
    assign_blocks,
    min_train_test_distance,
    random_kfold,
    region_holdout,
    spatial_block_kfold,
)


@pytest.fixture
def coords():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 10_000, size=(3000, 2))


def test_random_kfold_partitions_everything():
    n = 101
    tests = []
    for train, test in random_kfold(n, 5, seed=1):
        assert len(np.intersect1d(train, test)) == 0
        assert len(train) + len(test) == n
        tests.append(test)
    assert np.array_equal(np.sort(np.concatenate(tests)), np.arange(n))


def test_blocks_stay_whole_and_cover_all_points(coords):
    blocks = assign_blocks(coords, 1000)
    fold_of_point = np.full(len(coords), -1)
    for k, (train, test) in enumerate(spatial_block_kfold(coords, 1000, 5, seed=3)):
        assert len(np.intersect1d(train, test)) == 0
        fold_of_point[test] = k
    assert (fold_of_point >= 0).all()
    # every block belongs to exactly one fold
    for b in np.unique(blocks):
        assert len(np.unique(fold_of_point[blocks == b])) == 1


@pytest.mark.parametrize("buffer", [100.0, 500.0, 1500.0])
def test_buffer_guarantees_minimum_separation(coords, buffer):
    """The leakage test: no test point may be closer than `buffer` to any training point."""
    for train, test in spatial_block_kfold(coords, 1000, 5, buffer=buffer, seed=2):
        assert len(train) > 0 and len(test) > 0
        assert min_train_test_distance(coords, train, test) >= buffer


def test_without_buffer_neighbouring_blocks_can_touch(coords):
    dists = [
        min_train_test_distance(coords, tr, te)
        for tr, te in spatial_block_kfold(coords, 1000, 5, buffer=0.0, seed=2)
    ]
    assert min(dists) < 50.0  # dense points: adjacent blocks nearly touch


def test_random_split_has_tiny_separation(coords):
    d = [min_train_test_distance(coords, tr, te) for tr, te in random_kfold(len(coords), 5, 0)]
    assert max(d) < 20.0


def test_too_few_blocks_raises(coords):
    with pytest.raises(ValueError):
        list(spatial_block_kfold(coords, 20_000, 5))


def test_region_holdout_is_separated(coords):
    pool, region = region_holdout(coords, frac=0.25, axis=0, buffer=400.0)
    assert len(np.intersect1d(pool, region)) == 0
    assert min_train_test_distance(coords, pool, region) >= 400.0
    assert abs(len(region) / len(coords) - 0.25) < 0.02
