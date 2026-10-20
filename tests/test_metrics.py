"""TC-07..10: agreement metrics and the statistics used in the analysis."""
import numpy as np
import pytest

from xaiaudit.analysis import bh, cluster_boot_mean, regime_test
from xaiaudit.metrics import against_reference, jaccard_topk, pair_metrics, rank_corr, set_stability, top1, top_k


def test_top_k_uses_absolute_values_and_stable_ties():
    a = np.array([0.1, -0.9, 0.5, 0.5])
    assert top_k(a, 1) == {1}
    assert top_k(a, 2) == {1, 2}          # tie 0.5/0.5 broken by lower index
    assert top1(a) == 1


def test_jaccard():
    a = np.array([5, 4, 3, 0, 0, 0.0])
    b = np.array([5, 0, 3, 4, 0, 0.0])     # top-3 {0,1,2} vs {0,2,3}
    assert jaccard_topk(a, b, 3) == pytest.approx(2 / 4)
    assert jaccard_topk(a, a, 3) == 1.0
    assert jaccard_topk(a, np.array([0, 0, 0, 3, 4, 5.0]), 3) == 0.0


def test_rank_corr():
    a = np.array([1.0, 2, 3, 4])
    assert rank_corr(a, a) == pytest.approx(1.0)
    assert rank_corr(a, -a) == pytest.approx(1.0)          # signs ignored
    assert rank_corr(a, a[::-1]) == pytest.approx(-1.0)
    assert np.isnan(rank_corr(np.ones(4), a))               # undefined for constant input


def test_pair_and_set_stability():
    a = np.array([3.0, 2, 1, 0])
    m = pair_metrics(a, a, 2)
    assert m == {"jaccard": 1.0, "spearman": pytest.approx(1.0), "top1_same": 1.0}
    s = set_stability([a, a, np.array([0, 2, 1, 3.0])], 2)
    assert s["n_distinct_top1"] == 2
    assert s["modal_top1_share"] == pytest.approx(2 / 3)
    assert s["top1_same"] == pytest.approx(1 / 3)           # pairs: (a,a) same, two different
    r = against_reference(a, [a, np.array([0, 2, 1, 3.0])], 2)
    assert r["top1_changed"] == pytest.approx(0.5)


def test_benjamini_hochberg_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005,0.01,0.03,0.04 -> 0.02,0.02,0.04,0.04
    assert np.allclose(bh(p), [0.02, 0.04, 0.04, 0.02])
    assert np.all(bh(np.array([0.9, 0.95])) <= 1)


def test_cluster_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(size=200)
    c = np.repeat(np.arange(20), 10)
    m, lo, hi = cluster_boot_mean(v, c, 500, rng)
    assert lo < m < hi and m == pytest.approx(v.mean())


def test_regime_test_detects_real_gap_and_not_a_null():
    rng = np.random.default_rng(1)
    c = np.repeat(np.arange(30), 10)
    g = np.tile(np.r_[np.zeros(5, bool), np.ones(5, bool)], 30)
    # calibration: under the null, p < 0.05 should happen about 5% of the time
    ps = [regime_test(rng.normal(size=300), g, c, 200, rng)["p_perm"] for _ in range(40)]
    assert np.mean(np.array(ps) < 0.05) <= 0.15
    real = regime_test(rng.normal(size=300) + g * 1.0, g, c, 500, rng)
    assert real["p_perm"] < 0.01 and real["ci_lo"] > 0
