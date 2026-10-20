"""TC-11..12: TreeSHAP is deterministic and additive; stochastic explainers are seed-reproducible."""
import numpy as np
import pytest

from xaiaudit import explainers as E
from xaiaudit.features import FEATURES
from xaiaudit.models import fit_model


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES)))
    y = (X[:, 0] + 0.5 * X[:, 3] + rng.normal(scale=1.0, size=400) > 0).astype(float)
    return X, y


@pytest.mark.parametrize("name", ["xgb", "lgbm"])
def test_treeshap_deterministic_and_additive(data, name):
    X, y = data
    m = fit_model(name, X, y, seed=0)
    a, b = E.tree_shap(m, X[:20]), E.tree_shap(m, X[:20])
    assert np.array_equal(a, b)                              # bit-for-bit identical
    m2 = fit_model(name, X, y, seed=0)
    assert np.array_equal(a, E.tree_shap(m2, X[:20]))        # same seed -> same model -> same SHAP
    # local accuracy: margin - sum(SHAP) must be the same constant (the base value) for every row.
    # (We don't compare to shap's expected_value: with xgboost 3 it misreads base_score by a constant.)
    p = m.predict_proba(X[:20])[:, 1].astype(float)
    resid = np.log(p / (1 - p)) - a.sum(1)
    assert np.ptp(resid) < 1e-4


def test_stochastic_explainers_reproducible_with_same_seed(data):
    X, y = data
    m = fit_model("xgb", X, y, seed=0)
    bg = E.make_background(X, 10)
    k1 = E.kernel_shap(m, X[:3], bg, 100, seed=5)
    assert np.array_equal(k1, E.kernel_shap(m, X[:3], bg, 100, seed=5))
    lw = E.LimeWrapper(X, FEATURES)
    l1 = lw.explain(m, X[:3], 300, seed=5)
    assert np.array_equal(l1, lw.explain(m, X[:3], 300, seed=5))
    assert not np.array_equal(l1, lw.explain(m, X[:3], 300, seed=6))   # and a new seed does change it
