"""The four explanation audits, run per coin and parallel over coins.

For every (coin, walk-forward fold, model) this module:
  1. fits the model at `base_seed` and stores every out-of-sample prediction;
  2. samples `points_per_fold` test days (half per volatility regime) to explain;
  3. AGREEMENT     - compares TreeSHAP / KernelSHAP / LIME on the same point, and
                     permutation importance vs mean|attribution| globally;
  4. NOISE FLOOR   - reruns the stochastic explainers with new seeds on the identical
                     model + input (TreeSHAP must be bit-for-bit reproducible);
  5. RETRAINING    - refits with new seeds and moving-block bootstraps of the training
                     rows and re-explains the same points with the primary method;
  6. PERTURBATION  - adds eps * training-std Gaussian noise to the inputs and compares
                     to the unperturbed explanation.
The control model `xgb_shuffled` is XGBoost trained on permuted labels: it has no skill
by construction, so its explanations show what "explaining noise" looks like.

Seeding: every random draw comes from `_seed(base_seed, coin, fold, tag)`, a CRC of the
parts (not Python's salted `hash`), so runs are identical across processes and machines.
Stochastic-explainer seeds are shared between noise-floor rep r and perturbation rep r
(common random numbers), so the perturbation-minus-noise-floor gap isolates the input change.
"""
from __future__ import annotations

import os

# one thread per worker: avoids oversubscription and thread-order nondeterminism
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle
import time
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import explainers as E
from .data import load_coin
from .features import FEATURES, build_features
from .metrics import against_reference, pair_metrics, set_stability, top1
from .models import TREE_MODELS, fit_model, predict_p
from .validation import regime_labels, walk_forward_splits

CONTROL = "xgb_shuffled"
EXPLAINER_SEED_STEP = 100_003   # explainer seed for rep r = base_seed + STEP * (r + 1)
FI = {f: i for i, f in enumerate(FEATURES)}


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31 - 1)


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(_seed(*parts))


def model_names(cfg: dict) -> list[str]:
    return list(cfg["models"]) + [CONTROL]


def base_learner(model: str) -> str:
    return "xgb" if model == CONTROL else model


def methods_for(model: str) -> list[str]:
    return (["tree"] if base_learner(model) in TREE_MODELS else []) + ["kernel", "lime"]


def primary_method(model: str) -> str:
    return "tree" if base_learner(model) in TREE_MODELS else "kernel"


def sample_points(regimes: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positions within the test fold, half volatile / half calm when possible."""
    if n >= len(regimes):
        return np.arange(len(regimes))
    vol, calm = np.where(regimes == "volatile")[0], np.where(regimes == "calm")[0]
    n_vol = min(n // 2, len(vol))
    n_calm = min(n - n_vol, len(calm))
    n_vol = n - n_calm                     # fill any shortfall from the other regime
    pick = np.concatenate([rng.choice(vol, n_vol, replace=False),
                           rng.choice(calm, n_calm, replace=False)])
    return np.sort(pick)


def block_bootstrap(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap of row positions (keeps short-range autocorrelation)."""
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def perturb(Xp: np.ndarray, std: np.ndarray, eps: float, reps: int,
            rng: np.random.Generator) -> np.ndarray:
    """(reps, n_points, n_features) noisy copies that stay on the feature manifold:
    macd_hist is recomputed from the noisy macd/macd_signal and volatilities stay >= 0."""
    out = Xp[None] + rng.standard_normal((reps,) + Xp.shape) * eps * std
    out[..., FI["macd_hist"]] = out[..., FI["macd"]] - out[..., FI["macd_signal"]]
    for f in ("vol_7", "vol_30"):
        out[..., FI[f]] = np.maximum(out[..., FI[f]], 0.0)
    return out


class FoldContext:
    """Everything one fold's audits share: data slices, explainers, explained points."""

    def __init__(self, coin, fold, X, y, dates, vol, cfg):
        self.coin, self.fold, self.cfg = coin, fold, cfg
        tr, te = fold.train_idx, fold.test_idx
        self.Xtr, self.ytr, self.Xte, self.yte = X[tr], y[tr], X[te], y[te]
        self.dates_te = dates[te]
        self.threshold = float(np.median(vol[tr]))
        self.regime_te = regime_labels(pd.Series(vol[tr]), pd.Series(vol[te]))
        bs = cfg["base_seed"]
        self.pts = sample_points(self.regime_te, cfg["points_per_fold"], _rng(bs, coin, fold.fold_id, "points"))
        self.Xp = self.Xte[self.pts]
        self.std = self.Xtr.std(axis=0)
        self.background = E.make_background(self.Xtr, cfg["kernel_background"], seed=bs)
        self.lime = E.LimeWrapper(self.Xtr, FEATURES)
        self.y_shuffled = _rng(bs, coin, fold.fold_id, "shuffle").permutation(self.ytr)

    def labels(self, model: str) -> np.ndarray:
        return self.y_shuffled if model == CONTROL else self.ytr

    def fit(self, model: str, seed: int, rows: np.ndarray | None = None):
        X, y = self.Xtr, self.labels(model)
        if rows is not None:
            X, y = X[rows], y[rows]
        return fit_model(base_learner(model), X, y, seed)

    def explain(self, method: str, model, X: np.ndarray, seed: int) -> np.ndarray:
        c = self.cfg
        if method == "tree":
            return E.tree_shap(model, X)
        if method == "kernel":
            return E.kernel_shap(model, X, self.background, c["kernel_nsamples"], seed)
        if method == "lime":
            return self.lime.explain(model, X, c["lime_num_samples"], seed)
        raise ValueError(method)

    def point_keys(self) -> list[dict]:
        return [{"coin": self.coin, "fold": self.fold.fold_id, "point": int(i),
                 "date": self.dates_te[i], "regime": self.regime_te[i]} for i in self.pts]


def audit_model(ctx: FoldContext, model: str, out: dict, attrs: dict) -> None:
    cfg, bs, k = ctx.cfg, ctx.cfg["base_seed"], ctx.cfg["top_k"]
    fid, coin = ctx.fold.fold_id, ctx.coin
    keys = ctx.point_keys()
    methods, prim = methods_for(model), primary_method(model)
    expl_seed = lambda r: bs + EXPLAINER_SEED_STEP * (r + 1)

    # ---- base fit + predictions -------------------------------------------------
    m0 = ctx.fit(model, bs)
    p_te = predict_p(m0, ctx.Xte)
    out["predictions"].append(pd.DataFrame({
        "coin": coin, "model": model, "fold": fid, "date": ctx.dates_te,
        "regime": ctx.regime_te, "y": ctx.yte.astype(int), "p": p_te}))
    p0 = p_te[ctx.pts]

    # ---- 1. agreement -----------------------------------------------------------
    base = {m: ctx.explain(m, m0, ctx.Xp, bs) for m in methods}
    for a, b in combinations(methods, 2):
        for j, key in enumerate(keys):
            out["agreement_local"].append({**key, "model": model, "pair": f"{a}~{b}",
                                           **pair_metrics(base[a][j], base[b][j], k)})
    perm = E.perm_importance(m0, ctx.Xte, ctx.yte, cfg["perm_repeats"], bs)
    glob = {"perm": perm, "lime": np.abs(base["lime"]).mean(0), "kernel": np.abs(base["kernel"]).mean(0)}
    if "tree" in methods:  # TreeSHAP is cheap: use the whole test fold for its global view
        glob["tree"] = np.abs(E.tree_shap(m0, ctx.Xte)).mean(0)
    for name, v in glob.items():
        out["global_importance"].extend({"coin": coin, "fold": fid, "model": model, "method": name,
                                         "feature": f, "value": float(x)} for f, x in zip(FEATURES, v))
    for a, b in combinations(glob, 2):
        out["agreement_global"].append({"coin": coin, "fold": fid, "model": model, "pair": f"{a}~{b}",
                                        **pair_metrics(glob[a], glob[b], k)})

    # ---- 2. noise floor ---------------------------------------------------------
    noise = {}
    for m in methods:
        if m == "tree":
            again = E.tree_shap(m0, ctx.Xp)
            if not np.array_equal(again, base["tree"]):
                raise AssertionError(f"TreeSHAP not reproducible: {coin} fold {fid} {model}")
            reps = [again]
        else:
            reps = [ctx.explain(m, m0, ctx.Xp, expl_seed(r)) for r in range(cfg["noise_floor_reps"])]
        noise[m] = reps
        for j, key in enumerate(keys):
            vecs = [r[j] for r in reps]
            ref = against_reference(base[m][j], vecs, k)
            out["noise_floor"].append({**key, "model": model, "method": m,
                                       **set_stability([base[m][j]] + vecs, k),
                                       **{f"ref_{x}": v for x, v in ref.items()}})

    # ---- 3. retraining stability ------------------------------------------------
    retrain = {"seed": [], "bootstrap": []}
    ps = {"seed": [], "bootstrap": []}
    variants = [("seed", r, s, None) for r, s in enumerate(cfg["retrain_seeds"])]
    for r in range(cfg["bootstrap_reps"]):
        rows = block_bootstrap(len(ctx.Xtr), cfg["bootstrap_block"], _rng(bs, coin, fid, model, "boot", r))
        variants.append(("bootstrap", r, bs, rows))
    for source, r, seed, rows in variants:
        mr = ctx.fit(model, seed, rows)
        p_full = predict_p(mr, ctx.Xte)
        for reg in ("calm", "volatile"):
            mask = ctx.regime_te == reg
            if mask.any():
                out["retrain_accuracy"].append({
                    "coin": coin, "fold": fid, "model": model, "source": source, "rep": r,
                    "regime": reg, "n": int(mask.sum()),
                    "accuracy": float(((p_full[mask] > 0.5) == ctx.yte[mask]).mean())})
        ps[source].append(p_full[ctx.pts])
        retrain[source].append(ctx.explain(prim, mr, ctx.Xp, bs))
    for source in retrain:
        P = np.vstack([p0] + ps[source])                    # (1 + reps, n_points)
        for j, key in enumerate(keys):
            vecs = [base[prim][j]] + [a[j] for a in retrain[source]]
            out["retrain_stability"].append({
                **key, "model": model, "source": source, "method": prim, **set_stability(vecs, k),
                "p_std": float(P[:, j].std()),
                "class_flip_rate": float(np.mean((P[1:, j] > 0.5) != (P[0, j] > 0.5)))})
            for r, v in enumerate(vecs):
                out["retrain_top1"].append({**key, "model": model, "source": source,
                                            "rep": r - 1, "top1_feature": FEATURES[top1(v)]})

    # ---- 4. perturbation stability ----------------------------------------------
    perturbed = {}
    for e_i, eps in enumerate(cfg["perturb_eps"]):
        n_reps = {m: cfg["perturb_reps_tree"] if m == "tree" else cfg["perturb_reps_slow"] for m in methods}
        Xn = perturb(ctx.Xp, ctx.std, eps, max(n_reps.values()), _rng(bs, coin, fid, model, "eps", e_i))
        P = np.vstack([predict_p(m0, Xn[r]) for r in range(len(Xn))])  # (reps, n_points)
        for m in methods:
            reps = [ctx.explain(m, m0, Xn[r], expl_seed(r)) for r in range(n_reps[m])]
            perturbed[(m, eps)] = reps
            Pm = P[:n_reps[m]]
            for j, key in enumerate(keys):
                out["perturbation"].append({
                    **key, "model": model, "method": m, "eps": eps,
                    **against_reference(base[m][j], [a[j] for a in reps], k),
                    "flip_rate": float(np.mean((Pm[:, j] > 0.5) != (p0[j] > 0.5))),
                    "abs_dp": float(np.mean(np.abs(Pm[:, j] - p0[j])))})

    attrs[(coin, model, fid)] = {"points": keys, "base": base, "noise": noise,
                                 "retrain": retrain, "perturbed": perturbed, "global": glob}


TABLES = ["predictions", "agreement_local", "agreement_global", "global_importance", "noise_floor",
          "retrain_stability", "retrain_top1", "retrain_accuracy", "perturbation", "folds"]


def run_coin(coin: str, cfg: dict) -> tuple[dict, dict]:
    t0 = time.time()
    df = load_coin(coin, cfg["raw_dir"], cfg["start_date"], cfg["end_date"])
    F = build_features(df)
    X, y = F[FEATURES].to_numpy(float), F["target"].to_numpy(float)
    dates, vol = F.index.to_numpy(), F[cfg["regime_feature"]].to_numpy(float)
    folds = walk_forward_splits(F.index, cfg["test_start"], cfg["refit_every_days"],
                                cfg["min_train_days"], cfg["embargo_days"])
    out = {t: [] for t in TABLES}
    attrs = {}
    for fold in folds:
        ctx = FoldContext(coin, fold, X, y, dates, vol, cfg)
        out["folds"].append({
            "coin": coin, "fold": fold.fold_id, "train_start": dates[fold.train_idx[0]],
            "train_end": fold.train_end, "test_start": fold.test_start, "test_end": fold.test_end,
            "n_train": len(fold.train_idx), "n_test": len(fold.test_idx), "regime_threshold": ctx.threshold,
            "n_volatile": int((ctx.regime_te == "volatile").sum()),
            "n_calm": int((ctx.regime_te == "calm").sum()), "n_explained": len(ctx.pts)})
        for model in model_names(cfg):
            audit_model(ctx, model, out, attrs)
    print(f"[{coin}] {len(folds)} folds in {time.time() - t0:.1f}s", flush=True)
    return out, attrs


def _tidy(rows) -> pd.DataFrame:
    df = pd.concat(rows, ignore_index=True) if rows and isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
    for c in ("date", "train_start", "train_end", "test_start", "test_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df


def run_all_coins(cfg: dict) -> dict[str, pd.DataFrame]:
    """Run every coin (in parallel), write tidy CSVs + raw attributions to results/cache/."""
    results = Parallel(n_jobs=cfg["n_jobs"])(delayed(run_coin)(c, cfg) for c in cfg["coins"])
    cache = Path(cfg["results_dir"]) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tables = {}
    for t in TABLES:
        rows = [r for out, _ in results for r in out[t]]
        tables[t] = _tidy(rows)
        tables[t].to_csv(cache / f"{t}.csv", index=False, float_format="%.10g")
    attrs = {k: v for _, a in results for k, v in a.items()}
    with open(cache / "attributions.pkl", "wb") as fh:
        pickle.dump({"features": FEATURES, "attributions": attrs}, fh)
    return tables
