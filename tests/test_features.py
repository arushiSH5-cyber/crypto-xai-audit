"""TC-01..03: features are causal and the target is the NEXT day's direction."""
import numpy as np
import pandas as pd
import pytest

from xaiaudit.config import get_config
from xaiaudit.data import load_coin
from xaiaudit.features import FEATURES, build_features

CFG = get_config()


@pytest.fixture(scope="module")
def raw():
    return load_coin("btc", CFG["raw_dir"], CFG["start_date"], CFG["end_date"])


@pytest.mark.parametrize("cut", [400, 900, 1500, 2500, 3000])
def test_no_lookahead_truncation(raw, cut):
    """Removing future rows must not change any past feature value."""
    full = build_features(raw)
    trunc = build_features(raw.iloc[:cut])
    common = trunc.index.intersection(full.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(trunc.loc[common, FEATURES], full.loc[common, FEATURES],
                                  check_exact=False, rtol=1e-12, atol=1e-14)


def test_future_shock_does_not_leak(raw):
    """A huge price shock on day t may only affect features from day t on."""
    shocked = raw.copy()
    t = 1500
    shocked.iloc[t:, shocked.columns.get_loc("price")] *= 3.0
    a, b = build_features(raw), build_features(shocked)
    before = a.index[a.index < raw.index[t]]
    pd.testing.assert_frame_equal(a.loc[before, FEATURES], b.loc[before, FEATURES])
    # ...but the target of day t-1 (return t-1 -> t) must see it: that is the label, not a feature
    assert b.loc[raw.index[t - 1], "target"] == 1.0


def test_target_is_next_day_direction(raw):
    F = build_features(raw)
    r = np.log(raw["price"]).diff()
    nxt = r.shift(-1).loc[F.index]
    assert ((nxt > 0).astype(float) == F["target"]).all()
    assert F[FEATURES + ["target"]].notna().all().all()


def test_macd_trio_is_collinear(raw):
    F = build_features(raw)
    assert np.allclose(F["macd_hist"], F["macd"] - F["macd_signal"])
