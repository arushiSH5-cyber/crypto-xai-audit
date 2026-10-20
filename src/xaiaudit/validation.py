"""Strictly time-ordered walk-forward splits and causal regime labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray   # positional indices into the feature frame
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(dates: pd.DatetimeIndex, test_start: str, refit_every_days: int,
                        min_train_days: int, embargo_days: int = 1) -> list[Fold]:
    """Expanding-window walk-forward. No shuffling, no overlap.

    Fold i trains on every row strictly before (test_block_start - embargo) and
    tests on [test_block_start, test_block_start + refit_every_days).
    """
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted")
    folds = []
    block_start = pd.Timestamp(test_start)
    last = dates[-1]
    fid = 0
    while block_start <= last:
        block_end = block_start + pd.Timedelta(days=refit_every_days)
        train_cut = block_start - pd.Timedelta(days=embargo_days)
        train_idx = np.where(dates < train_cut)[0]
        test_idx = np.where((dates >= block_start) & (dates < block_end))[0]
        if len(test_idx) and len(train_idx) >= min_train_days:
            folds.append(Fold(fid, train_idx, test_idx, dates[train_idx[-1]],
                              dates[test_idx[0]], dates[test_idx[-1]]))
            fid += 1
        block_start = block_end
    return folds


def regime_labels(regime_values_train: pd.Series, regime_values_test: pd.Series) -> np.ndarray:
    """'volatile' if trailing volatility on the test day exceeds the TRAINING median.

    The threshold uses only the fold's training window, so the label is known
    at prediction time (no look-ahead in the regime definition itself).
    """
    thr = float(np.median(regime_values_train))
    return np.where(regime_values_test.values > thr, "volatile", "calm")
