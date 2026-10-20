"""Wrappers that return attributions as plain (n_points, n_features) arrays.

Output spaces (rankings are compared, not raw magnitudes):
  TreeSHAP   : log-odds, path-dependent (shap default, as used in most papers)
  KernelSHAP : log-odds (link='logit'), interventional w.r.t. a k-means background
  LIME       : probability, local weighted linear surrogate (default discretisation)
  Permutation: global only (drop in ROC-AUC when a feature is shuffled)
"""
from __future__ import annotations

import warnings

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

from .models import predict_p

warnings.filterwarnings("ignore", category=UserWarning)


def tree_shap(model, X: np.ndarray) -> np.ndarray:
    expl = shap.TreeExplainer(model)
    vals = expl.shap_values(X)
    if isinstance(vals, list):          # older shap returns one array per class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:                  # (n, p, classes)
        vals = vals[:, :, 1]
    return vals


def make_background(X_train: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X_train)
    return km.cluster_centers_


def kernel_shap(model, X: np.ndarray, background: np.ndarray, nsamples: int, seed: int) -> np.ndarray:
    f = lambda Z: np.clip(predict_p(model, Z), 1e-6, 1 - 1e-6)
    np.random.seed(seed)
    expl = shap.KernelExplainer(f, background, link="logit")
    out = np.zeros_like(X, dtype=float)
    for i in range(len(X)):
        np.random.seed(seed + 7919 * i)
        out[i] = np.asarray(expl.shap_values(X[i:i + 1], nsamples=nsamples, silent=True)).reshape(-1)
    return out


class LimeWrapper:
    def __init__(self, X_train: np.ndarray, feature_names: list[str]):
        self.X_train = X_train
        self.names = feature_names

    def explain(self, model, X: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
        expl = LimeTabularExplainer(self.X_train, feature_names=self.names, mode="classification",
                                    discretize_continuous=True, random_state=seed)
        fn = lambda Z: model.predict_proba(Z)
        out = np.zeros_like(X, dtype=float)
        p = X.shape[1]
        for i in range(len(X)):
            e = expl.explain_instance(X[i], fn, labels=(1,), num_features=p, num_samples=num_samples)
            for j, w in e.as_map()[1]:
                out[i, j] = w
        return out


def perm_importance(model, X_test: np.ndarray, y_test: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=repeats,
                               random_state=seed, n_jobs=1)
    return r.importances_mean
