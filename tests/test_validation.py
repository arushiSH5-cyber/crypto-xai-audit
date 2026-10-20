"""TC-04..06: walk-forward folds are time-ordered and disjoint; regimes use train data only."""
import numpy as np
import pandas as pd

from xaiaudit.config import get_config
from xaiaudit.data import load_coin
from xaiaudit.experiments import FoldContext
from xaiaudit.features import FEATURES, build_features
from xaiaudit.validation import regime_labels, walk_forward_splits

CFG = get_config()


def _frame():
    return build_features(load_coin("eth", CFG["raw_dir"], CFG["start_date"], CFG["end_date"]))


def _folds(F):
    return walk_forward_splits(F.index, CFG["test_start"], CFG["refit_every_days"],
                               CFG["min_train_days"], CFG["embargo_days"])


def test_folds_time_ordered_and_disjoint():
    F = _frame()
    folds = _folds(F)
    assert len(folds) >= 5
    seen_test = set()
    for f in folds:
        d = F.index
        assert d[f.train_idx].max() < d[f.test_idx].min()                     # train strictly before test
        gap = (d[f.test_idx].min() - d[f.train_idx].max()).days
        assert gap >= CFG["embargo_days"] + 1                                  # embargo respected
        assert np.all(np.diff(f.train_idx) == 1) and np.all(np.diff(f.test_idx) == 1)
        assert not seen_test & set(f.test_idx)                                 # test blocks never overlap
        seen_test |= set(f.test_idx)
    for a, b in zip(folds, folds[1:]):
        assert F.index[a.test_idx].max() < F.index[b.test_idx].min()          # folds move forward in time
        assert len(b.train_idx) > len(a.train_idx)                             # expanding window
    assert F.index[folds[0].test_idx].min() >= pd.Timestamp(CFG["test_start"])


def test_regime_threshold_uses_training_data_only():
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    test = pd.Series([2.9, 3.1])
    base = regime_labels(train, test)
    assert list(base) == ["calm", "volatile"]
    # Changing other test values (even wildly) must not move the threshold
    assert list(regime_labels(train, pd.Series([2.9, 3.1, 1e6]))[:2]) == list(base)


def test_fold_context_threshold_ignores_future():
    F = _frame()
    fold = _folds(F)[3]
    X, y = F[FEATURES].to_numpy(float), F["target"].to_numpy(float)
    dates, vol = F.index.to_numpy(), F["vol_30"].to_numpy(float)
    cfg = {**CFG, "points_per_fold": 4}
    ctx = FoldContext("eth", fold, X, y, dates, vol, cfg)
    assert ctx.threshold == float(np.median(vol[fold.train_idx]))
    vol2 = vol.copy()
    vol2[fold.test_idx[-1] + 1:] = 99.0          # corrupt everything after the test block
    vol2[fold.test_idx] *= 1.0                   # test values themselves unchanged
    ctx2 = FoldContext("eth", fold, X, y, dates, vol2, cfg)
    assert ctx2.threshold == ctx.threshold
    assert (ctx2.regime_te == ctx.regime_te).all()
