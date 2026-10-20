"""Single source of truth for every experimental setting.

Change values here (or pass a dict of overrides to `get_config`) instead of
editing code. `QUICK` overrides give a fast smoke-test run used by the tests.
"""
from __future__ import annotations

import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    # ---------------- data ----------------
    "raw_dir": str(ROOT / "data" / "raw"),
    "processed_dir": str(ROOT / "data" / "processed"),
    "results_dir": str(ROOT / "results"),
    "coins": ["btc", "eth", "ada", "link", "ltc"],  # 2 large caps + 3 mid caps
    "start_date": "2017-12-01",   # first date loaded (feature warm-up starts here)
    "end_date": "2026-05-22",     # last date with complete price + volume rows
    # ---------------- validation ----------------
    "test_start": "2021-01-01",   # first out-of-sample day
    "refit_every_days": 182,      # walk-forward step (each fold = ~6 months of test days)
    "min_train_days": 730,        # never train on less than ~2 years
    "embargo_days": 1,            # rows dropped between train end and test start
    # ---------------- regimes ----------------
    "regime_feature": "vol_30",   # trailing 30-day realised volatility (causal)
    # threshold = median of regime_feature over the fold's TRAINING window
    # ---------------- models ----------------
    "models": ["xgb", "lgbm", "mlp"],
    "base_seed": 0,
    # ---------------- explanations ----------------
    "points_per_fold": 10,        # explained test days per (coin, fold), half per regime
    "top_k": 3,
    "kernel_background": 20,      # k-means summary of training data
    "kernel_nsamples": 200,
    "lime_num_samples": 1000,
    "perm_repeats": 10,
    # ---------------- audits ----------------
    "retrain_seeds": [1, 2, 3, 4, 5],
    "bootstrap_reps": 5,
    "bootstrap_block": 30,        # moving-block bootstrap keeps autocorrelation
    "perturb_eps": [0.01, 0.05],  # noise = eps * training std of each feature
    "perturb_reps_tree": 10,
    "perturb_reps_slow": 3,       # KernelSHAP / LIME are expensive
    "noise_floor_reps": 3,        # repeated runs of a stochastic explainer, same input
    "placebo_reps": 2000,         # random regime relabellings for the null distribution
    "ci_boot_reps": 2000,
    "n_jobs": 2,
}

QUICK = {
    "coins": ["btc", "eth"],
    "test_start": "2024-01-01",
    "points_per_fold": 4,
    "kernel_nsamples": 80,
    "lime_num_samples": 300,
    "perm_repeats": 3,
    "retrain_seeds": [1, 2],
    "bootstrap_reps": 2,
    "perturb_eps": [0.05],
    "perturb_reps_tree": 3,
    "perturb_reps_slow": 2,
    "noise_floor_reps": 2,
    "placebo_reps": 200,
    "ci_boot_reps": 200,
}


def get_config(quick: bool = False, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(CONFIG)
    if quick:
        cfg.update(QUICK)
    if overrides:
        cfg.update(overrides)
    return cfg
