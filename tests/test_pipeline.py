"""TC-13..16: experiment helpers and end-to-end reproducibility of the quick run."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xaiaudit.config import get_config
from xaiaudit.experiments import FI, block_bootstrap, perturb, sample_points


def test_sample_points_balanced_and_deterministic():
    reg = np.array(["calm"] * 150 + ["volatile"] * 30)
    a = sample_points(reg, 10, np.random.default_rng(3))
    assert (reg[a] == "volatile").sum() == 5 and (reg[a] == "calm").sum() == 5
    assert np.array_equal(a, sample_points(reg, 10, np.random.default_rng(3)))
    few = np.array(["calm"] * 50 + ["volatile"] * 2)
    b = sample_points(few, 10, np.random.default_rng(0))
    assert len(b) == 10 and (few[b] == "volatile").sum() == 2      # shortfall filled from other regime


def test_block_bootstrap_shape():
    rows = block_bootstrap(1000, 30, np.random.default_rng(0))
    assert len(rows) == 1000 and rows.min() >= 0 and rows.max() < 1000
    assert np.all(np.diff(rows[:30]) == 1)                           # first block is contiguous


def test_perturb_keeps_feature_identities():
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(5, len(FI)))) * 0.01
    out = perturb(X, X.std(0) + 1, 0.5, 4, rng)
    assert out.shape == (4, 5, len(FI))
    assert np.allclose(out[..., FI["macd_hist"]], out[..., FI["macd"]] - out[..., FI["macd_signal"]])
    assert (out[..., FI["vol_30"]] >= 0).all() and (out[..., FI["vol_7"]] >= 0).all()


def _run_quick(tmp: Path):
    import run_all
    run_all.main(["--quick", "--results-dir", str(tmp)])
    return {p.name: pd.read_csv(p) for p in sorted((tmp / "cache").glob("*.csv"))} | \
           {"T_" + p.name: pd.read_csv(p) for p in sorted((tmp / "tables").glob("*.csv"))}


@pytest.mark.slow
def test_two_quick_runs_identical(tmp_path):
    a = _run_quick(tmp_path / "a")
    b = _run_quick(tmp_path / "b")
    assert a.keys() == b.keys() and len(a) > 10
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], check_exact=True, obj=k)
    # sanity: the quick run covers every audit and the control model
    assert set(a["predictions.csv"]["model"]) == {"xgb", "lgbm", "mlp", "xgb_shuffled"}
    assert set(a["predictions.csv"]["coin"]) == set(get_config(quick=True)["coins"])
