# Defense prep: likely questions and honest answers

Numbers come from `results/tables/`. File references point to where the answer lives in the code.

---

**1. Your models are barely better than a coin flip. Why audit the explanations of models that don't work?**

Because that is the typical situation in published next-day crypto models, and those papers
still show SHAP plots. Our pooled accuracy is 0.512 (xgb), 0.512 (lgbm) and 0.507 (mlp),
against 0.502 for the shuffled-label control. The point of the audit is to show what such
explanations are worth. A key result is that the no-skill control's explanations are about as
stable as the real models' (TreeSHAP seed Jaccard 0.593 vs 0.624). I say plainly in the report
that our conclusions apply to low-skill models and may not hold for a model with real skill.

**2. How do you know there's no look-ahead leakage?**

Three safeguards. (i) Every feature uses data up to day *t* only, and the target is day *t+1*'s
direction (`features.py`). Test TC-01 rebuilds the features on the series truncated at 5 points
and checks that every past value is identical. TC-02 triples all future prices and checks that
no earlier feature moves. (ii) Walk-forward folds train only on rows before the test block,
with a 1-day embargo, and never shuffle (TC-05). (iii) The regime threshold is the median of the
*training* window (TC-06, TC-07).

**3. Why define "volatile" relative to the training median, and why use `vol_30`, which is also a feature?**

A full-sample median would use future volatility to label past days, which is look-ahead. The
training median is known on the prediction day. `vol_30` is the standard 30-day realised
volatility and is the natural regime variable. Being a model input doesn't leak anything,
because it is causal. The side effect is that 2021–2026 was calmer than the training years:
only 0.293 of test days are volatile, and 7 of the 55 folds have no volatile days at all. I
handled that by sampling 5 calm + 5 volatile points where possible (310 calm, 240 volatile overall).

**4. Is 550 explained points enough to detect a regime difference?**

It is enough to rule out large differences, not small ones. The 95% CIs on volatile − calm have
half-widths from 0.016 to 0.072 depending on the audit. For example, the pooled seed-retraining
difference is −0.007 [−0.022, 0.009]. So a regime effect bigger than about 0.02 in Jaccard is
unlikely there, but a smaller one can't be excluded. The full run takes about 5 minutes, so
raising `points_per_fold` in `config.py` is the obvious next step.

**5. Why a cluster bootstrap instead of treating every point as independent?**

The 10 points in a (coin, fold) share the same fitted model and the same 6-month market. Their
errors are correlated, so treating them as independent would make the CIs too narrow. I resample
whole (coin, fold) clusters (`analysis.cluster_boot_mean`), and the permutation test shuffles
labels only *within* a fold, so each fold keeps its own mix of regimes.

**6. Why Benjamini–Hochberg rather than Bonferroni?**

There are 56 tests. Bonferroni controls the chance of *any* false positive and is very
conservative at that size. BH controls the expected *share* of false discoveries, which suits an
exploratory audit. Raw p < 0.05 occurred in 12 tests; 4 survive BH at q < 0.05. With Bonferroni,
even fewer would survive, so the null conclusion on explanation stability would only get stronger.

**7. What is the placebo test, and why did you change it?**

Each fold borrows the calm/volatile sequence of a *different, randomly chosen* fold, 2000 times.
The fake labels look realistic but have no link to the real volatility. The spread of fake
"regime effects" shows what chance alone produces. My first version circularly shifted each
fold's own labels, but regimes last for months, so small shifts left the labels almost
unchanged and the "null" partly contained the real effect. I documented that switch in the
code and in `docs/design.md`.

**8. What is the noise floor, and why do you need it?**

KernelSHAP and LIME are random. Run them twice on the same model and input and you get different
answers. Without a noise floor, you can't tell "the model changed" from "the explainer rolled
different dice". LIME's top-3 Jaccard against itself is only 0.638 (xgb); KernelSHAP's is 0.938.
TreeSHAP's is exactly 1, which is asserted at runtime. That is why the TreeSHAP retraining
instability (seed Jaccard 0.624) is genuine model sensitivity. And it is why I report that a
1%-std nudge barely moves LIME beyond its noise (excess top-1 change 0.010 for xgb).

**9. TreeSHAP and KernelSHAP both compute Shapley values. Why do they agree only 0.686?**

They answer slightly different questions. TreeSHAP here is *path-dependent*: it uses the tree's
own split statistics for missing features. KernelSHAP is *interventional*: it replaces features
with background values from a k-means summary. With correlated features (the MACD trio, the
overlapping return windows), the two split credit differently. KernelSHAP is also an estimate
from 200 samples, and shap's default `l1_reg` zeroes 2 of the 12 attributions per point.

**10. Why is LIME so unstable?**

LIME fits a local linear model to 1000 random perturbations of binned features. The sampling is
random, and with near-flat model surfaces (a no-skill model) the fitted coefficients are
dominated by noise. Its top-1 feature changes in 0.238 (xgb) to 0.339 (control) of seed-only
reruns. This is the problem S-LIME (Zhou et al. 2021) was designed to fix, and I cite it for
that reason.

**11. Why does permutation importance disagree with SHAP so badly (Spearman about 0.29)?**

They measure different things. Permutation importance asks "how much does *accuracy* (AUC) drop
if I scramble this feature?" When the model has almost no skill, every feature's answer is near
zero and the ranking is mostly noise. SHAP asks "how much does this feature move the *output*?",
which can be large even when the output is useless. Their disagreement is itself a warning sign
of low skill.

**12. The shuffled-label control is as stable as the real models. What does that mean?**

That stability and agreement are properties of the explainer and the model class, not evidence
that the model found real signal. A model trained on random labels still has a fixed structure
that TreeSHAP describes consistently. The one place the real model differs is the *global*
ranking across folds: xgb puts `ret_lag1` in its TreeSHAP top-3 in 0.927 of folds, while the
control's most frequent top-3 feature (`ret_5d`) reaches only 0.418.

**13. You found higher accuracy in volatile regimes. Is that real, and does it make money?**

It is statistically robust for the tree models: xgb 0.538 vs 0.502 (q = 0.028), lgbm 0.537 vs
0.501 (q = 0.028), and the placebo p for xgb is 0.0005. It is not universal: for LINK, xgb is
0.492 in volatile regimes. I did not test profitability. There are no transaction costs or
position sizing, and a few percentage points of directional accuracy is not proof of a
tradable edge. I don't claim one.

**14. Doesn't the exactly collinear MACD trio make the instability an artefact?**

Partly. That is a real limitation, and I list it. When features are collinear, SHAP and LIME can
split credit among them in many ways, so the "top" feature among the three can flip easily.
`features.FEATURE_GROUPS` exists so that a grouped analysis could be run, but I did not run it.
Grouping would probably look more stable. I kept the feature-level view because that is what
applied papers report.

**15. Why the MLP results going the *other* way (more stable when volatile)?**

The data show it: MLP seed-retraining 1 − Jaccard is −0.026 lower when volatile (q = 0.028), and
MLP/LIME at ε = 0.05 has P(top-1 changes) −0.109 lower (q = 0.028). One plausible explanation is
that volatile days have larger feature values, so the strongest feature stands out more clearly.
I did not test that mechanism, so I present it only as a hypothesis. The pooled headline tests
are all non-significant.

---

### Quick facts to have ready

| Fact | Value | Source |
|---|---|---|
| Coins, folds, test days per model | 5, 55, 9835 | `key_numbers.json` |
| Explained points (calm / volatile) | 550 (310 / 240) | `regime_tests.csv` |
| Regime tests / significant at q < 0.05 | 56 / 4 | `key_numbers.json` |
| Random-ranking top-3 Jaccard | 0.164 | `key_numbers.json` |
| Full run time | 310.3 s (2 workers) | `results/run_log.json` |
| Tests | 25 passed | `pytest -q` |
