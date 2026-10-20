"""The three predictive models. Deliberately small and standard.

Randomness sources controlled by `seed`:
  xgb / lgbm : row subsampling (0.8) and column subsampling (0.8) per tree
  mlp        : weight initialisation, mini-batch order, early-stopping split
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TREE_MODELS = {"xgb", "lgbm"}


def make_model(name: str, seed: int):
    if name == "xgb":
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
            random_state=seed, n_jobs=1, verbosity=0, eval_metric="logloss")
    if name == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=200, num_leaves=8, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
            random_state=seed, n_jobs=1, verbose=-1)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3,
            max_iter=300, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=seed))
    raise ValueError(name)


def fit_model(name: str, X: np.ndarray, y: np.ndarray, seed: int):
    model = make_model(name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def predict_p(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
