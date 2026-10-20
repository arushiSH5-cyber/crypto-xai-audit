"""Turn the tidy audit CSVs in results/cache/ into tables, tests and figures.

Statistical unit. Explained points are nested in (coin, fold) clusters: points from
one fold share a model and a market period, so they are not independent. Therefore
  * confidence intervals use a CLUSTER bootstrap over (coin, fold);
  * regime p-values permute the calm/volatile labels WITHIN each (coin, fold), which
    keeps the fold structure and the per-fold regime counts fixed;
  * the PLACEBO test gives each fold the regime sequence of a different, randomly chosen
    fold (another half-year), which keeps realistic run lengths and proportions but breaks
    the labels' link to the volatility on the actual dates. Its null distribution is the
    "how big a regime gap appears by chance" floor;
  * Benjamini-Hochberg controls the false discovery rate over the whole test family.
Every number the report quotes is written to results/tables/ (CSV + Markdown) or to
results/tables/key_numbers.json.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

from .experiments import CONTROL
from .features import FEATURES

CLUSTER = ["coin", "fold"]
REGIMES = ["calm", "volatile"]
COLORS = {"calm": "#2a78d6", "volatile": "#eb6834", "floor": "#8a8984", "ink": "#2b2b29",
          "grid": "#e4e3df", "sig": "#e34948"}
MODEL_ORDER = ["xgb", "lgbm", "mlp", CONTROL]
SHORT = {"tree": "TS", "kernel": "KS", "lime": "LIME"}
METHOD_LABEL = {"tree": "TreeSHAP", "kernel": "KernelSHAP", "lime": "LIME", "perm": "Permutation"}


# ------------------------------------------------------------------ statistics
def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _cluster_codes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby(CLUSTER, sort=True).ngroup().to_numpy()


def cluster_boot_mean(values: np.ndarray, clusters: np.ndarray, reps: int, rng) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling whole clusters."""
    values = np.asarray(values, float)
    ok = ~np.isnan(values)
    values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    uc, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uc))
    cnts = np.bincount(inv, minlength=len(uc)).astype(float)
    draws = rng.integers(0, len(uc), size=(reps, len(uc)))
    boots = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def _diff(values, is_vol) -> float:
    return float(values[is_vol].mean() - values[~is_vol].mean())


def regime_test(values: np.ndarray, is_vol: np.ndarray, clusters: np.ndarray, reps: int, rng,
                placebo_labels: list[np.ndarray] | None = None) -> dict:
    """volatile - calm mean difference with cluster-bootstrap CI, within-cluster permutation
    p-value and (optionally) a placebo p-value from pre-computed relabellings."""
    values = np.asarray(values, float)
    ok = ~np.isnan(values)
    v, g, c = values[ok], is_vol[ok], clusters[ok]
    res = {"n_calm": int((~g).sum()), "n_volatile": int(g.sum()),
           "mean_calm": float(v[~g].mean()) if (~g).any() else np.nan,
           "mean_volatile": float(v[g].mean()) if g.any() else np.nan}
    if not g.any() or g.all():
        return {**res, "diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_perm": np.nan, "p_placebo": np.nan}
    obs = _diff(v, g)
    # within-cluster permutation: shuffle labels inside each cluster
    order = np.argsort(c, kind="stable")
    v_s, g_s, c_s = v[order], g[order], c[order]
    bounds = np.flatnonzero(np.diff(c_s)) + 1
    segs = np.split(np.arange(len(c_s)), bounds)
    null = np.empty(reps)
    for r in range(reps):
        gp = g_s.copy()
        for s in segs:
            gp[s] = g_s[rng.permutation(s)]
        null[r] = _diff(v_s, gp)
    p_perm = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + reps)
    # cluster bootstrap of the difference
    uc, inv = np.unique(c, return_inverse=True)
    sv = np.bincount(inv, weights=v * g, minlength=len(uc))
    nv = np.bincount(inv, weights=g.astype(float), minlength=len(uc))
    sc = np.bincount(inv, weights=v * ~g, minlength=len(uc))
    nc = np.bincount(inv, weights=(~g).astype(float), minlength=len(uc))
    draws = rng.integers(0, len(uc), size=(reps, len(uc)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = sv[draws].sum(1) / nv[draws].sum(1) - sc[draws].sum(1) / nc[draws].sum(1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    res.update(diff=obs, ci_lo=float(lo), ci_hi=float(hi), p_perm=float(p_perm))
    if placebo_labels is not None:
        pl = np.array([_diff(v, lab[ok]) if 0 < lab[ok].sum() < ok.sum() else np.nan for lab in placebo_labels])
        pl = pl[~np.isnan(pl)]
        res["p_placebo"] = float((1 + np.sum(np.abs(pl) >= abs(obs) - 1e-12)) / (1 + len(pl)))
        res["placebo_lo"], res["placebo_hi"] = (float(x) for x in np.percentile(pl, [2.5, 97.5]))
        res["_placebo_null"] = pl
    return res


class PlaceboLabeller:
    """Give each (coin, fold) the regime sequence of a DIFFERENT walk-forward fold (random donor,
    any coin, different half-year), aligned by position in the test window.

    The borrowed labels have realistic run lengths and calm/volatile proportions but no link to
    the volatility on the actual dates. (A circular shift of the fold's own sequence was tried
    first and rejected: regimes persist for months, so small shifts leave labels near-identical
    to the real ones and the 'null' absorbs part of any real effect.)"""

    def __init__(self, predictions: pd.DataFrame, reps: int, seed: int):
        first = predictions[predictions["model"] == predictions["model"].iloc[0]]
        seqs = {k: (g.sort_values("date")["regime"].to_numpy() == "volatile")
                for k, g in first.groupby(CLUSTER)}
        keys = list(seqs)
        self.keys = {k: i for i, k in enumerate(keys)}
        self.flat = np.concatenate(list(seqs.values()))
        self.lens = np.array([len(s) for s in seqs.values()])
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        folds = np.array([k[1] for k in keys])
        rng = np.random.default_rng(seed)
        self.donors = np.empty((reps, len(keys)), int)
        for i, f in enumerate(folds):
            pool = np.flatnonzero(folds != f)
            self.donors[:, i] = rng.choice(pool, size=reps)

    def labels(self, df: pd.DataFrame) -> list[np.ndarray]:
        """`df['point']` = position of the row within its fold's test window."""
        k = np.array([self.keys[x] for x in zip(df["coin"], df["fold"])])
        pos = df["point"].to_numpy()
        return [self.flat[self.starts[d[k]] + pos % self.lens[d[k]]] for d in self.donors]


# ------------------------------------------------------------------ IO helpers
def load_cache(results_dir: str) -> dict[str, pd.DataFrame]:
    cache = Path(results_dir) / "cache"
    return {p.stem: pd.read_csv(p) for p in sorted(cache.glob("*.csv"))}


def _fmt(x):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        if x != 0 and abs(x) < 1e-3:
            return f"{x:.1e}"
        return f"{x:.3f}"
    return str(x)


def write_table(df: pd.DataFrame, name: str, tables_dir: Path) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False, float_format="%.6g")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (tables_dir / f"{name}.md").write_text("\n".join(lines) + "\n")


def _order_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df


# ------------------------------------------------------------------ summaries
def summarize(df: pd.DataFrame, by: list[str], metrics: list[str], reps: int, rng,
              with_regime_all: bool = True) -> pd.DataFrame:
    """Mean (+ cluster-bootstrap CI for the first metric) per group, per regime and pooled."""
    rows = []
    for keys, g in df.groupby(by, sort=False, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        subsets = [("all", g)] if with_regime_all else []
        subsets += [(r, g[g["regime"] == r]) for r in REGIMES]
        for reg, s in subsets:
            if s.empty:
                continue
            row = dict(zip(by, keys), regime=reg, n=len(s))
            m, lo, hi = cluster_boot_mean(s[metrics[0]].to_numpy(), _cluster_codes(s), reps, rng)
            row.update({metrics[0]: m, f"{metrics[0]}_lo": lo, f"{metrics[0]}_hi": hi})
            for mm in metrics[1:]:
                row[mm] = float(s[mm].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def accuracy_tables(pred: pd.DataFrame, reps: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred.copy()
    pred["correct"] = ((pred["p"] > 0.5) == (pred["y"] == 1)).astype(float)
    rows, coin_rows = [], []
    for model in MODEL_ORDER:
        g = pred[pred["model"] == model]
        for reg in ["all"] + REGIMES:
            s = g if reg == "all" else g[g["regime"] == reg]
            cl = _cluster_codes(s)
            acc, lo, hi = cluster_boot_mean(s["correct"].to_numpy(), cl, reps, rng)
            auc = roc_auc_score(s["y"], s["p"])
            # cluster bootstrap for AUC
            uc = np.unique(cl)
            idx_by = [np.flatnonzero(cl == u) for u in uc]
            aucs = []
            for _ in range(reps):
                take = np.concatenate([idx_by[i] for i in rng.integers(0, len(uc), len(uc))])
                yy = s["y"].to_numpy()[take]
                if 0 < yy.sum() < len(yy):
                    aucs.append(roc_auc_score(yy, s["p"].to_numpy()[take]))
            k = int(s["correct"].sum())
            rows.append({"model": model, "regime": reg, "n": len(s), "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                         "auc": auc, "auc_lo": float(np.percentile(aucs, 2.5)),
                         "auc_hi": float(np.percentile(aucs, 97.5)),
                         "binom_p_vs_50": binomtest(k, len(s), 0.5).pvalue,
                         "always_up_acc": float(s["y"].mean()),
                         "share_pred_up": float((s["p"] > 0.5).mean())})
        for (coin, reg), s in g.groupby(["coin", "regime"]):
            coin_rows.append({"coin": coin, "model": model, "regime": reg, "n": len(s),
                              "accuracy": s["correct"].mean(), "auc": roc_auc_score(s["y"], s["p"]),
                              "always_up_acc": s["y"].mean()})
        for coin, s in g.groupby("coin"):
            coin_rows.append({"coin": coin, "model": model, "regime": "all", "n": len(s),
                              "accuracy": s["correct"].mean(), "auc": roc_auc_score(s["y"], s["p"]),
                              "always_up_acc": s["y"].mean()})
    coin_df = pd.DataFrame(coin_rows).sort_values(["coin", "model", "regime"], key=lambda c: c.map(
        {m: i for i, m in enumerate(MODEL_ORDER)}) if c.name == "model" else c)
    return pd.DataFrame(rows), coin_df.reset_index(drop=True)


def build_tests(T: dict, cfg: dict, placebo: PlaceboLabeller, rng) -> tuple[pd.DataFrame, dict]:
    """The full family of volatile-vs-calm tests (one primary metric per audit cell)."""
    reps = cfg["placebo_reps"]
    specs = []   # (family, model, detail, metric, frame)
    pred = T["predictions"].copy()
    pred["correct"] = ((pred["p"] > 0.5) == (pred["y"] == 1)).astype(float)
    for m in MODEL_ORDER:
        specs.append(("accuracy", m, "all test days", "correct", pred[pred["model"] == m]))
    for (m, pair), g in T["agreement_local"].groupby(["model", "pair"]):
        specs.append(("agreement", m, pair, "jaccard", g))
    for (m, meth), g in T["noise_floor"].groupby(["model", "method"]):
        if meth != "tree":
            specs.append(("noise_floor", m, meth, "jaccard", g))
    for (m, src), g in T["retrain_stability"].groupby(["model", "source"]):
        specs.append(("retrain", m, src, "jaccard", g))
    for (m, meth, eps), g in T["perturbation"].groupby(["model", "method", "eps"]):
        specs.append(("perturbation", m, f"{meth} eps={eps:g}", "top1_changed", g))
    # pooled headline tests over the three real models (control excluded)
    real = lambda d: d[d["model"] != CONTROL]
    rs = real(T["retrain_stability"])
    specs.append(("headline", "pooled", "retrain seed", "jaccard", rs[rs["source"] == "seed"]))
    specs.append(("headline", "pooled", "retrain bootstrap", "jaccard", rs[rs["source"] == "bootstrap"]))
    pt = real(T["perturbation"])
    prim = pt[(pt["method"] == "tree") | ((pt["model"] == "mlp") & (pt["method"] == "kernel"))]
    eps_max = max(cfg["perturb_eps"])
    specs.append(("headline", "pooled", f"perturbation primary eps={eps_max:g}", "top1_changed",
                  prim[prim["eps"] == eps_max]))
    specs.append(("headline", "pooled", "agreement all pairs", "jaccard", real(T["agreement_local"])))

    rows, nulls = [], {}
    for fam, m, detail, metric, g in specs:
        g = g.reset_index(drop=True)
        is_vol = (g["regime"] == "volatile").to_numpy()
        cl = g.groupby(CLUSTER).ngroup().to_numpy()
        if fam == "accuracy":   # one row per test day: its position in the fold is its 'point'
            g = g.sort_values(CLUSTER + ["date"]).reset_index(drop=True)
            g["point"] = g.groupby(CLUSTER).cumcount()
            is_vol = (g["regime"] == "volatile").to_numpy()
            cl = g.groupby(CLUSTER).ngroup().to_numpy()
        pl = placebo.labels(g)
        vals = g[metric].to_numpy(float)
        if metric == "jaccard":          # orient every explanation metric as INSTABILITY
            vals, metric = 1 - vals, "1-jaccard"
        res = regime_test(vals, is_vol, cl, reps, rng, pl)
        null = res.pop("_placebo_null", None)
        key = f"{fam}|{m}|{detail}|{metric}"
        if null is not None:
            nulls[key] = (null, res["diff"])
        rows.append({"family": fam, "model": m, "detail": detail, "metric": metric, **res})
    tests = pd.DataFrame(rows)
    ok = tests["p_perm"].notna()
    tests.loc[ok, "q_bh"] = bh(tests.loc[ok, "p_perm"].to_numpy())
    tests["significant_q05"] = tests["q_bh"] < 0.05
    return tests, nulls


# ------------------------------------------------------------------ figures
def _style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b9b8b2")
    ax.tick_params(colors=COLORS["ink"], labelsize=8)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9.5, color=COLORS["ink"], loc="left")


def _save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _regime_bars(ax, labels, calm, vol, calm_err=None, vol_err=None, floor=None):
    x = np.arange(len(labels))
    w = 0.38
    for off, vals, err, reg in [(-w / 2, calm, calm_err, "calm"), (w / 2, vol, vol_err, "volatile")]:
        ax.bar(x + off, vals, w - 0.03, color=COLORS[reg], label=reg, zorder=2)
        if err is not None:
            ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=COLORS["ink"], lw=0.9, capsize=2, zorder=3)
    if floor is not None:
        for xi, f in zip(x, floor):
            if not np.isnan(f):
                ax.hlines(f, xi - w, xi + w, colors=COLORS["floor"], lw=2, linestyles="--", zorder=4,
                          label="explainer noise floor" if xi == x[0] or "explainer noise floor" not in
                          [h.get_label() for h in ax.collections] else None)
    ax.set_xticks(x, labels, fontsize=8)


def _err(df, col):
    return np.vstack([df[col] - df[f"{col}_lo"], df[f"{col}_hi"] - df[col]])


def fig_accuracy(acc: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, metric, lab in [(axes[0], "accuracy", "Accuracy"), (axes[1], "auc", "ROC-AUC")]:
        c = acc[acc["regime"] == "calm"].set_index("model").loc[MODEL_ORDER]
        v = acc[acc["regime"] == "volatile"].set_index("model").loc[MODEL_ORDER]
        lo, hi = ("acc_lo", "acc_hi") if metric == "accuracy" else ("auc_lo", "auc_hi")
        ce = np.vstack([c[metric] - c[lo], c[hi] - c[metric]])
        ve = np.vstack([v[metric] - v[lo], v[hi] - v[metric]])
        _regime_bars(ax, MODEL_ORDER, c[metric].to_numpy(), v[metric].to_numpy(), ce, ve)
        ax.axhline(0.5, color=COLORS["ink"], lw=0.9, ls=":", zorder=5)
        lo_all = min(c[lo].min(), v[lo].min(), 0.5)
        hi_all = max(c[hi].max(), v[hi].max(), 0.5)
        ax.set_ylim(lo_all - 0.02, hi_all + 0.02)
        _style(ax, lab)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Out-of-sample skill by regime (95% cluster-bootstrap CI); dotted line = coin flip (0.5)",
                 fontsize=9.5, x=0.01, ha="left", y=1.02)
    _save(fig, path)


def fig_agreement_heatmap(agree: pd.DataFrame, path: Path):
    methods = ["tree", "kernel", "lime"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), gridspec_kw={"wspace": 0.12})
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    for ax, m in zip(axes, MODEL_ORDER):
        M = np.full((3, 3), np.nan)
        for pair, val in agree[agree["model"] == m].groupby("pair")["jaccard"].mean().items():
            a, b = pair.split("~")
            M[methods.index(a), methods.index(a)] = M[methods.index(b), methods.index(b)] = 1.0
            M[methods.index(a), methods.index(b)] = M[methods.index(b), methods.index(a)] = val
        ax.imshow(M, vmin=0, vmax=1, cmap=cmap)
        for i in range(3):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 else COLORS["ink"])
        ax.set_xticks(range(3), [METHOD_LABEL[x] for x in methods], fontsize=7, rotation=30)
        ax.set_yticks(range(3), [METHOD_LABEL[x] for x in methods] if ax is axes[0] else [], fontsize=7)
        ax.set_title(m, fontsize=9.5, loc="left")
    fig.suptitle("Cross-method agreement: mean top-3 Jaccard per explained point (1 = same top-3)",
                 fontsize=9.5, x=0.01, ha="left")
    _save(fig, path)


def fig_retrain(rs: pd.DataFrame, nf: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, src in zip(axes, ["seed", "bootstrap"]):
        s = rs[rs["source"] == src]
        c = s[s["regime"] == "calm"].set_index("model").reindex(MODEL_ORDER)
        v = s[s["regime"] == "volatile"].set_index("model").reindex(MODEL_ORDER)
        floor = [1.0 if m != "mlp" else nf[(nf["model"] == m) & (nf["method"] == "kernel") &
                                            (nf["regime"] == "all")]["jaccard"].iloc[0] for m in MODEL_ORDER]
        _regime_bars(ax, MODEL_ORDER, c["jaccard"].to_numpy(), v["jaccard"].to_numpy(),
                     _err(c, "jaccard"), _err(v, "jaccard"), floor)
        ax.set_ylim(0, 1.05)
        _style(ax, f"Retraining ({src}): top-3 Jaccard across refits")
    axes[0].set_ylabel("mean pairwise top-3 Jaccard", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    _save(fig, path)


def fig_perturbation(ps: pd.DataFrame, nf: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, m in zip(axes, MODEL_ORDER):
        s = ps[(ps["model"] == m) & (ps["regime"] != "all")]
        meths = [x for x in ["tree", "kernel", "lime"] if x in set(s["method"])]
        labels, calm, vol, ce, ve, floor = [], [], [], [], [], []
        for meth in meths:
            for eps in sorted(s["eps"].unique()):
                c = s[(s["method"] == meth) & (s["eps"] == eps) & (s["regime"] == "calm")].iloc[0]
                v = s[(s["method"] == meth) & (s["eps"] == eps) & (s["regime"] == "volatile")].iloc[0]
                labels.append(f"{SHORT[meth]}\n{eps:g}")
                calm.append(c["top1_changed"]); vol.append(v["top1_changed"])
                ce.append([c["top1_changed"] - c["top1_changed_lo"], c["top1_changed_hi"] - c["top1_changed"]])
                ve.append([v["top1_changed"] - v["top1_changed_lo"], v["top1_changed_hi"] - v["top1_changed"]])
                f = nf[(nf["model"] == m) & (nf["method"] == meth) & (nf["regime"] == "all")]
                floor.append(f["ref_top1_changed"].iloc[0] if len(f) else np.nan)
        _regime_bars(ax, labels, np.array(calm), np.array(vol), np.array(ce).T, np.array(ve).T, floor)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0, 0.6)
        _style(ax, m)
    axes[0].set_ylabel("P(top-1 feature changes)", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Perturbation: how often the top-1 driver changes vs. the explainer's own noise floor (dashed).\n"
                 "x-axis: method (TS = TreeSHAP, KS = KernelSHAP) and ε (noise as a fraction of the training std)",
                 fontsize=9.5, x=0.01, ha="left", y=1.1)
    _save(fig, path)


def fig_forest(tests: pd.DataFrame, path: Path):
    t = tests[tests["diff"].notna()].reset_index(drop=True)
    parts = [t[t["family"] == "accuracy"], t[t["family"] != "accuracy"]]
    heights = [len(p) for p in parts]
    fig, axes = plt.subplots(2, 1, figsize=(8, 0.15 * sum(heights) + 1.6),
                             gridspec_kw={"height_ratios": heights})
    titles = ["Accuracy: volatile − calm (positive = more accurate when volatile)",
              "Explanation instability: volatile − calm (positive = LESS stable when volatile)\n"
              "instability = 1 − top-3 Jaccard, or P(top-1 changes) for perturbation"]
    for ax, part, title in zip(axes, parts, titles):
        y = np.arange(len(part))[::-1]
        for yi, r in zip(y, part.itertuples()):
            col = COLORS["sig"] if r.significant_q05 else (COLORS["ink"] if r.family == "headline" else COLORS["floor"])
            ax.hlines(yi, r.ci_lo, r.ci_hi, color=col, lw=1.4)
            ax.plot(r.diff, yi, "o", ms=4.5, color=col, mec="white", mew=0.8)
            if not np.isnan(getattr(r, "placebo_lo", np.nan)):
                ax.plot([r.placebo_lo, r.placebo_hi], [yi, yi], "|", color=COLORS["calm"], ms=6, mew=1.2)
        ax.axvline(0, color=COLORS["ink"], lw=0.8)
        ax.set_ylim(-0.7, len(part) - 0.3)
        ax.set_yticks(y, [f"{r.family}: {r.model} · {r.detail}" for r in part.itertuples()], fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(title, fontsize=8.5, loc="left")
    fig.suptitle("Grey/black line = 95% cluster-bootstrap CI; blue ticks = placebo 95% range; red = BH q < 0.05",
                 fontsize=8.5, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    _save(fig, path)


def fig_top1_frequency(top1: pd.DataFrame, path: Path):
    s = top1[top1["source"] == "seed"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True)
    for ax, m in zip(axes, MODEL_ORDER):
        g = s[s["model"] == m]
        y = np.arange(len(FEATURES))
        w = 0.4
        for off, reg in [(-w / 2, "calm"), (w / 2, "volatile")]:
            share = g[g["regime"] == reg]["top1_feature"].value_counts(normalize=True).reindex(FEATURES).fillna(0)
            ax.barh(y + off, share.to_numpy(), w - 0.04, color=COLORS[reg], label=reg)
        ax.set_yticks(y, FEATURES, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(m, fontsize=9.5, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Share of explanations (base model + retrain seeds) whose top-1 driver is each feature",
                 fontsize=9.5, x=0.01, ha="left")
    _save(fig, path)


def fig_placebo(nulls: dict, keys: list[str], path: Path):
    keys = [k for k in keys if k in nulls]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.8))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        null, obs = nulls[k]
        ax.hist(null, bins=40, color=COLORS["floor"], alpha=0.8)
        ax.axvline(obs, color=COLORS["volatile"], lw=2, label="observed")
        _, _, detail, metric = k.split("|")
        _style(ax, f"{detail}\n[{metric}]")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Placebo: regime labels borrowed from a different fold; null distribution of volatile − calm",
                 fontsize=9.5, x=0.01, ha="left", y=1.12)
    _save(fig, path)


# ------------------------------------------------------------------ main entry
def run_analysis(cfg: dict) -> dict:
    res = Path(cfg["results_dir"])
    tables_dir, fig_dir = res / "tables", res / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    T = load_cache(cfg["results_dir"])
    T = {k: (_order_models(v) if "model" in v.columns else v) for k, v in T.items()}
    T = {k: (v.assign(model=v["model"].astype(str)) if "model" in v.columns else v) for k, v in T.items()}
    reps, seed = cfg["ci_boot_reps"], cfg["base_seed"]
    rng = lambda tag: np.random.default_rng([seed, zlib.crc32(tag.encode())])

    acc, acc_coin = accuracy_tables(T["predictions"], reps, rng("acc"))
    write_table(acc, "accuracy_by_regime", tables_dir)
    write_table(acc_coin, "accuracy_by_coin", tables_dir)

    agree = summarize(_order_models(T["agreement_local"]).sort_values(["model", "pair"]),
                      ["model", "pair"], ["jaccard", "spearman", "top1_same"], reps, rng("agree"))
    write_table(agree, "agreement_local_summary", tables_dir)
    ag = _order_models(T["agreement_global"]).sort_values(["model", "pair"])
    agg = ag.groupby(["model", "pair"], observed=True).agg(
        n_folds=("jaccard", "size"), jaccard=("jaccard", "mean"), spearman=("spearman", "mean"),
        top1_same=("top1_same", "mean")).reset_index()
    write_table(agg, "agreement_global_summary", tables_dir)

    nf = summarize(_order_models(T["noise_floor"]).sort_values(["model", "method"]), ["model", "method"],
                   ["jaccard", "spearman", "top1_same", "n_distinct_top1", "ref_jaccard", "ref_top1_changed"],
                   reps, rng("nf"))
    write_table(nf, "noise_floor_summary", tables_dir)

    rs = summarize(_order_models(T["retrain_stability"]).sort_values(["model", "source"]), ["model", "source", "method"],
                   ["jaccard", "spearman", "top1_same", "n_distinct_top1", "modal_top1_share", "p_std",
                    "class_flip_rate"], reps, rng("rs"))
    floor_j = nf[nf["regime"] == "all"].set_index(["model", "method"])["jaccard"]
    rs["noise_floor_jaccard"] = [floor_j.get((m, me), np.nan) for m, me in zip(rs["model"], rs["method"])]
    ra = T["retrain_accuracy"]
    ra_w = (ra.assign(c=ra["accuracy"] * ra["n"]).groupby(["model", "source", "rep", "regime"])
            .agg(c=("c", "sum"), n=("n", "sum")).reset_index())
    ra_w["acc"] = ra_w["c"] / ra_w["n"]
    ra_all = ra_w.groupby(["model", "source", "rep"])[["c", "n"]].sum().reset_index().assign(regime="all")
    ra_all["acc"] = ra_all["c"] / ra_all["n"]
    ra_w = pd.concat([ra_w, ra_all], ignore_index=True)
    spread = ra_w.groupby(["model", "source", "regime"])["acc"].agg(["min", "max"]).reset_index()
    rs = rs.merge(spread.rename(columns={"min": "retrain_acc_min", "max": "retrain_acc_max"}),
                  on=["model", "source", "regime"], how="left")
    write_table(rs, "retrain_summary", tables_dir)

    ps = summarize(_order_models(T["perturbation"]).sort_values(["model", "method", "eps"]),
                   ["model", "method", "eps"], ["top1_changed", "jaccard", "spearman", "flip_rate", "abs_dp"],
                   reps, rng("ps"))
    nf_ref = nf.set_index(["model", "method", "regime"])["ref_top1_changed"]
    ps["noise_floor_top1_changed"] = [nf_ref.get((m, me, r), np.nan)
                                      for m, me, r in zip(ps["model"], ps["method"], ps["regime"])]
    ps["excess_over_floor"] = ps["top1_changed"] - ps["noise_floor_top1_changed"]
    write_table(ps, "perturbation_summary", tables_dir)

    # excess perturbation instability over the noise floor, paired per point (stochastic methods)
    key = ["coin", "fold", "point", "model", "method"]
    ex = T["perturbation"].merge(T["noise_floor"][key + ["ref_top1_changed", "ref_jaccard"]], on=key)
    ex = ex[ex["method"] != "tree"]
    ex["excess_top1_changed"] = ex["top1_changed"] - ex["ref_top1_changed"]
    exs = summarize(_order_models(ex).sort_values(["model", "method", "eps"]), ["model", "method", "eps"],
                    ["excess_top1_changed"], reps, rng("ex"))
    write_table(exs, "perturbation_excess_over_noise_floor", tables_dir)

    # global feature importance: mean rank (1 = most important) and share of folds in top-3
    gi = T["global_importance"].copy()
    gi["rank"] = gi.groupby(["coin", "fold", "model", "method"])["value"].rank(ascending=False, method="first")
    top = (gi.groupby(["model", "method", "feature"])
           .agg(mean_rank=("rank", "mean"), share_top3=("rank", lambda r: float((r <= 3).mean())))
           .reset_index().sort_values(["model", "method", "mean_rank"]))
    top = _order_models(top).sort_values(["model", "method", "mean_rank"])
    write_table(top, "global_feature_ranks", tables_dir)

    placebo = PlaceboLabeller(T["predictions"], cfg["placebo_reps"], seed)
    tests, nulls = build_tests(T, cfg, placebo, rng("tests"))
    write_table(tests, "regime_tests", tables_dir)

    folds = T["folds"]
    write_table(folds, "folds", tables_dir)

    # ---- figures
    fig_accuracy(acc, fig_dir / "fig1_accuracy_by_regime.png")
    fig_agreement_heatmap(T["agreement_local"], fig_dir / "fig2_method_agreement_heatmap.png")
    fig_retrain(rs, nf, fig_dir / "fig3_retrain_stability_by_regime.png")
    fig_perturbation(ps, nf, fig_dir / "fig4_perturbation_vs_noise_floor.png")
    fig_forest(tests, fig_dir / "fig5_volatile_minus_calm_forest.png")
    fig_top1_frequency(T["retrain_top1"], fig_dir / "fig6_top1_feature_frequency.png")
    head = [k for k in nulls if k.startswith("headline")]
    fig_placebo(nulls, head, fig_dir / "fig7_placebo_null.png")

    key_numbers = {
        "n_tests": int(tests["p_perm"].notna().sum()),
        "n_significant_q05": int(tests["significant_q05"].sum()),
        "n_raw_p_below_05": int((tests["p_perm"] < 0.05).sum()),
        "significant_tests": tests[tests["significant_q05"]][["family", "model", "detail", "metric", "diff",
                                                               "p_perm", "q_bh"]].to_dict("records"),
        # what two RANDOM rankings of the 12 features would score (exact, hypergeometric)
        "random_baseline_top3_jaccard": random_jaccard(len(FEATURES), cfg["top_k"]),
        "random_baseline_top1_same": 1 / len(FEATURES),
        "n_points_explained": int(T["agreement_local"][["coin", "fold", "point"]].drop_duplicates().shape[0]),
        "n_folds_total": int(len(folds)),
        "n_test_days_per_model": int((T["predictions"]["model"] == "xgb").sum()),
        "share_volatile_test_days": float((T["predictions"].query("model == 'xgb'")["regime"] == "volatile").mean()),
    }
    (tables_dir / "key_numbers.json").write_text(json.dumps(key_numbers, indent=2, default=float))
    return {"tests": tests, "accuracy": acc, "key_numbers": key_numbers}


def random_jaccard(p: int, k: int) -> float:
    """Expected top-k Jaccard between two independent uniformly random rankings of p features."""
    from math import comb
    return sum(comb(k, i) * comb(p - k, k - i) / comb(p, k) * i / (2 * k - i) for i in range(k + 1))
