---
title: "Can You Trust the Explanation? Auditing the Agreement and Stability of SHAP and LIME for Cryptocurrency Direction Models Across Volatility Regimes"
subtitle: "CS 180 Research Project, San José State University, Fall 2026"
date: "October 2026"
---

# Abstract

Applied papers on cryptocurrency forecasting often report SHAP or LIME feature importances
as if they described the market. We audit whether such explanations can be trusted. We train
next-day direction models (XGBoost, LightGBM, a small MLP) for five coins (BTC, ETH, ADA,
LINK, LTC) under strictly time-ordered walk-forward validation (55 folds, 9835 out-of-sample
days per model). We then measure four properties: (a) agreement between TreeSHAP,
KernelSHAP, LIME and permutation importance; (b) stability under retraining with new seeds
and block bootstraps; (c) stability under small input perturbations; and (d) whether any of
this instability is worse in high-volatility regimes. Every instability figure is set
against the explainer's own seed-to-seed noise floor and against a shuffled-label control
model with no skill. The models are close to coin flips: pooled accuracy is 0.507–0.512
against 0.502 for the control, and the small edge exists only in volatile regimes (XGBoost
0.538 vs 0.502 when calm). Explanations are fragile. LIME shares only about a third of its
top-3 features with SHAP and only 0.64 with itself across seeds. TreeSHAP has a zero noise
floor, yet its top-3 set keeps a Jaccard of only 0.624 across retraining seeds and 0.312
across block bootstraps. The skill-less control is about as stable as the real models. On
the key question we find **no evidence that explanations are less stable in volatile
regimes**. None of 52 explanation tests shows significantly higher instability in volatile
periods after Benjamini–Hochberg correction, and the two significant explanation
differences point the other way. We conclude that per-prediction feature-importance claims
from such models should not be read as statements about crypto markets, in calm or volatile
conditions alike.

# 1 Introduction

Machine-learning models for cryptocurrency prices are now common, and many papers pair them
with post-hoc explainers such as SHAP [1] or LIME [4] to report which indicators "drive"
the predictions [7]. Those reports carry an implicit claim: that the explanation describes
something stable about the model, and through the model, about the market. Both links are
weak. Daily crypto direction is close to unpredictable, so a model may have little to
explain. Explanation methods themselves are known to be sensitive to random seeds, sampling,
correlated features and small input changes [2, 3, 5, 8], and different methods often
disagree [10].

This project quantifies how far the feature-importance story can be trusted in this specific
setting, and under what conditions it breaks down. We ask:

* **RQ-a.** Do TreeSHAP, KernelSHAP, LIME and permutation importance agree on the top drivers?
* **RQ-b.** Are the drivers stable when the model is retrained (new seeds, block bootstraps)?
* **RQ-c.** Are they stable when the inputs are perturbed slightly?
* **RQ-d (key question).** Is the instability worse in high-volatility regimes than in calm ones?

Our contribution is an evaluation design rather than a new method. Its parts: strictly causal
features and walk-forward validation; four audits run on the same explained points; an explicit
*noise floor* for every stochastic explainer, so that seed noise is not mistaken for
instability; a *shuffled-label control model* that shows what explanations of a no-skill
model look like; and regime comparisons with cluster-bootstrap confidence intervals,
within-fold permutation tests, a placebo relabelling test and false-discovery-rate control.
Model accuracy is reported alongside every stability result.

# 2 Related work

**Attribution methods.** LIME [4] fits a weighted linear surrogate around a point using
random perturbations. SHAP [1] grounds attributions in Shapley values. KernelSHAP estimates
them by weighted regression on sampled coalitions, and TreeSHAP [11] computes them exactly
for tree ensembles. Permutation importance [9, 16] measures the global drop in performance
when one feature is shuffled. Molnar [9] surveys these methods and their caveats, especially
under correlated features.

**Fragility of explanations.** Alvarez-Melis and Jaakkola [2] showed that LIME and SHAP
explanations can change sharply under small input changes. Ghorbani et al. [8] showed that
explanations of neural networks are fragile under imperceptible perturbations. Slack et
al. [3] built models that deliberately fool LIME and SHAP. Zhou et al. [5] traced LIME's
instability to its sampling and proposed S-LIME. Note that Ribeiro et al. [4] *introduce*
LIME and are not themselves a stability study.

**Disagreement.** Krishna et al. [10] named the "disagreement problem". They found that
practitioners routinely see methods disagree on top-k features, and they defined top-k
feature agreement, rank agreement and related metrics. Our top-3 Jaccard and Spearman
metrics follow that framing.

**Crypto forecasting.** Gradient-boosted trees [6, 15] and small neural networks are standard
in applied crypto forecasting [7]. Such work usually reports accuracy and a single SHAP plot,
without checking the plot's stability. We found no study that measures agreement, retraining
stability and perturbation stability together, against a noise floor and a no-skill control,
split by volatility regime.

# 3 Data and features

**Data.** We use daily Coin Metrics Community Data [14] for BTC, ETH, ADA, LINK and LTC,
taking `PriceUSD` (a daily reference price at 00:00 UTC) and `volume_reported_spot_usd_1d`
from 2017-12-01 to 2026-05-22. The loader refuses gaps and duplicates instead of filling
them. The data are close-only (no open/high/low). LTC replaces SOL, which was planned, because
SOL's volume history is missing in this dataset.

**Features.** Twelve features are computed from close and volume using information up to the
end of day *t* only: log returns at lags 1–3; 5- and 10-day cumulative returns; Wilder RSI(14);
MACD, its signal line and histogram (each scaled by price); 7- and 30-day realised volatility;
and log volume relative to its 30-day mean. The MACD trio is exactly collinear by construction
(`macd_hist = macd − macd_signal`). We keep it because correlated features are common in
practice and split attribution credit. The target is 1 if the next day's log return is
positive. The first 60 rows are dropped as EWM warm-up, leaving 3034 rows per coin. Tests
TC-01 to TC-04 check causality by truncating the series and by injecting a future price shock.

# 4 Methods

**Walk-forward validation.** The validation scheme is an expanding window. Test blocks are
182 days long, the first starts on 2021-01-01, and there is a 1-day embargo between training
and test data. Every fold trains on all earlier rows and has at least 730 training days.
This gives 11 folds per coin and 55 in total, with 1967 out-of-sample days per coin per model
and no shuffling anywhere (test TC-05).

**Volatility regimes.** A test day is *volatile* if its trailing 30-day volatility exceeds the
median of that feature over the fold's *training* window. Otherwise it is *calm*. The threshold
uses training data only (TC-06, TC-07), so the label is known on the prediction day. Because
2021–2026 was calmer than 2018–2020, 0.293 of test days are volatile.

**Models.** XGBoost (200 trees, depth 3), LightGBM (200 trees, 8 leaves), both with 0.8
row and column subsampling, and an MLP (32–16 hidden units, early stopping, standardised
inputs). The **control** `xgb_shuffled` is the same XGBoost trained on a random permutation of
the training labels, so it has no skill by construction. Every model is fit at `base_seed = 0`.

**Explainers.** *TreeSHAP* (path-dependent, log-odds; tree models only). *KernelSHAP* (logit
link, 20-centre k-means background from the training fold, 200 coalition samples; the shap
default `l1_reg = num_features(10)` keeps 10 of the 12 features non-zero). *LIME* (default
quartile discretisation, 1000 samples, probability space). *Permutation importance* (drop in
ROC-AUC on the test fold, 10 repeats; global only). We compare rankings of |attribution|,
never raw magnitudes, because the methods live in different output spaces.

**Explained points.** In each (coin, fold) we sample 10 test days, 5 per regime, giving 550
points (310 calm, 240 volatile; some folds had fewer than 5 volatile days). All four audits
use the same points.

**Metrics.** Our primary metric is **top-3 Jaccard** between the sets of the three features
with the largest |attribution| (1 = same set). We also report Spearman correlation of |attr|
over all 12 features, **top-1 agreement**, and for perturbations **P(top-1 changes)**. For
calibration: two *random* rankings of 12 features have expected top-3 Jaccard 0.164 and top-1
agreement 0.083.

**Audits.**
(a) *Agreement:* per point, every pair among the available methods (three for trees,
KernelSHAP–LIME for the MLP). Globally per fold, permutation importance against mean |attr|.
(b) *Noise floor:* KernelSHAP and LIME re-run 3 times with new seeds on the identical model and
input. TreeSHAP is re-run and asserted bit-identical.
(c) *Retraining:* 5 refits with new seeds and 5 refits on moving-block bootstraps (block 30
days) of the training rows. Each refit is re-explained with the model's *primary* method
(TreeSHAP for trees, KernelSHAP for the MLP), and we record the prediction spread.
(d) *Perturbation:* Gaussian noise of ε × (training std) for ε ∈ {0.01, 0.05}, with 10
repetitions for TreeSHAP and 3 for KernelSHAP/LIME. `macd_hist` is recomputed and
volatilities are clipped at 0, so inputs stay on the feature manifold. Perturbation rep *r*
reuses the explainer seed of noise-floor rep *r* (common random numbers), so the gap between
the two isolates the input change.

**Statistics.** Points within a (coin, fold) share a model and a market period. Confidence
intervals are therefore 95% **cluster bootstraps** over the 55 (coin, fold) clusters (2000
resamples). For each audit cell we test volatile − calm with a **within-cluster permutation
test** (2000 permutations, which keeps each fold's regime counts fixed). Every explanation
metric is oriented as *instability* (1 − Jaccard, or P(top-1 changes)), so a positive
difference means "less stable when volatile". **Benjamini–Hochberg** [12] controls the false
discovery rate over the full family of 56 tests: 4 accuracy, 10 agreement, 8 noise-floor,
8 retraining, 22 perturbation and 4 pooled headline tests. As a **placebo**, each fold receives
the regime sequence of a *different*, randomly chosen fold (2000 times). This keeps realistic
run lengths and proportions but removes any link to actual volatility. The placebo
distribution shows how large a regime gap can appear by chance.

**Reproducibility.** All randomness derives from `base_seed` through CRC-hashed task keys.
Two runs give bit-identical CSVs (TC-20). The full run took 310.3 s (about 5 minutes) on 2 worker processes
(Apple-silicon laptop). Since this is well under the 60-minute budget, the configured sample
sizes were kept unchanged.

# 5 Results

## 5.1 The models have almost no skill (context for everything else)

![Out-of-sample accuracy and ROC-AUC by regime, with 95% cluster-bootstrap CIs.](figures/fig1_accuracy_by_regime.png){width=95%}

| model | accuracy (all) | 95% CI | AUC | calm acc. | volatile acc. | volatile − calm (q) |
|---|---|---|---|---|---|---|
| xgb | 0.512 | 0.502–0.523 | 0.520 | 0.502 | 0.538 | +0.036 (0.028) |
| lgbm | 0.512 | 0.501–0.522 | 0.517 | 0.501 | 0.537 | +0.035 (0.028) |
| mlp | 0.507 | 0.498–0.517 | 0.509 | 0.501 | 0.522 | +0.021 (0.150) |
| xgb_shuffled (control) | 0.502 | 0.493–0.511 | 0.502 | 0.500 | 0.506 | +0.006 (0.758) |

Table: Accuracy over 9835 test days per model. "Always predict up" scores 0.504 overall
(0.506 calm, 0.496 volatile). Source: `accuracy_by_regime.csv`, `regime_tests.csv`.

Pooled accuracy is barely above a coin flip. The MLP's CI includes 0.5, and the tree models'
CIs start at 0.502 and 0.501. In **calm** regimes no model differs from the control: all are
at 0.500–0.502. In **volatile** regimes XGBoost and LightGBM reach 0.538 and 0.537 (AUC 0.544
and 0.542), which beats the control (0.506) and "always up" (0.496). This regime difference
survives BH correction for both tree models. The edge is not universal. For LINK the models
show no skill in either regime (XGBoost 0.498 overall, 0.492 volatile; `accuracy_by_coin.csv`).
**Every explanation below therefore describes a model that is mostly fitting noise. That is
the typical situation in published next-day crypto models, and it is exactly when
feature-importance stories are most tempting and least justified.**

## 5.2 RQ-a: the methods disagree, and LIME disagrees with itself

![Mean per-point top-3 Jaccard between explainers. Random rankings would score 0.164.](figures/fig2_method_agreement_heatmap.png){width=100%}

| model | TreeSHAP–KernelSHAP | TreeSHAP–LIME | KernelSHAP–LIME |
|---|---|---|---|
| xgb | 0.686 (top-1 same 0.742) | 0.324 (0.331) | 0.329 (0.355) |
| lgbm | 0.653 (0.747) | 0.328 (0.335) | 0.319 (0.324) |
| mlp | – | – | 0.438 (0.453) |
| xgb_shuffled | 0.651 (0.740) | 0.314 (0.276) | 0.288 (0.269) |

Table: Per-point top-3 Jaccard (top-1 agreement in brackets), 550 points per cell. Source: `agreement_local_summary.csv`.

The two SHAP estimators share about two-thirds of their top-3 features. They estimate the
same quantity, but path-dependent TreeSHAP and interventional KernelSHAP treat correlated
features differently. LIME agrees with either SHAP variant at about 0.32, which is above the
random baseline (0.164) but far below SHAP–SHAP. Part of this is LIME's own noise: re-running
LIME on the identical model and point gives a top-3 Jaccard of only 0.638 (xgb) to 0.678 (mlp),
and its top-1 feature changes in 0.231–0.339 of reruns (`noise_floor_summary.csv`). KernelSHAP
is far more repeatable (0.876–0.945; top-1 changes 0.043–0.068). TreeSHAP is exactly
repeatable (1.000).

*Globally* the picture is worse. Permutation importance, the only method that measures
dependence of *accuracy* on a feature, agrees poorly with mean |TreeSHAP|: per-fold top-3
Jaccard 0.267 for xgb and 0.278 for lgbm, Spearman 0.286 and 0.279. For the MLP, its Spearman
with mean |KernelSHAP| is 0.101 (`agreement_global_summary.csv`). This is what a near-zero-skill
model predicts: permutation importance measures a feature's effect on AUC, which is near
zero for every feature, while SHAP measures its effect on the output, which can be large even
when the output is uninformative.

The control shows the same pattern. Its method agreement (0.651 TreeSHAP–KernelSHAP) is about
the same as the real XGBoost's (0.686). **Agreement between explainers is not evidence that
the model has learned anything.**

## 5.3 RQ-b: drivers move under retraining, beyond any noise floor

![Retraining stability (top-3 Jaccard across the base model and 5 refits). Dashed line: the primary explainer's own noise floor (1.0 for TreeSHAP).](figures/fig3_retrain_stability_by_regime.png){width=95%}

| model (method) | seed Jaccard [95% CI] | seed top-1 same | bootstrap Jaccard [95% CI] | bootstrap top-1 same | noise floor | class flips (seed / boot) |
|---|---|---|---|---|---|---|
| xgb (TreeSHAP) | 0.624 [0.607, 0.641] | 0.664 | 0.312 [0.301, 0.323] | 0.301 | 1.000 | 0.128 / 0.280 |
| lgbm (TreeSHAP) | 0.582 [0.569, 0.595] | 0.662 | 0.320 [0.309, 0.332] | 0.316 | 1.000 | 0.150 / 0.279 |
| mlp (KernelSHAP) | 0.313 [0.302, 0.325] | 0.284 | 0.446 [0.425, 0.467] | 0.441 | 0.876 | 0.412 / 0.280 |
| xgb_shuffled (TreeSHAP) | 0.593 [0.577, 0.609] | 0.633 | 0.285 [0.274, 0.297] | 0.255 | 1.000 | 0.140 / 0.288 |

Table: Mean pairwise stability among the base model and 5 refits, 550 points each. "Class flips" = share of refits whose predicted class differs from the base model's. Source: `retrain_summary.csv`.

Changing only the random seed of XGBoost (which controls row and column subsampling) leaves
the top-3 set at a Jaccard of 0.624. The top-1 driver agrees in only 0.664 of model pairs, and
across 6 models a point has on average 1.936 distinct top-1 features. TreeSHAP's noise floor is
exactly 1, so **all of this is genuine model sensitivity, not explainer noise.** Resampling the
training years with a block bootstrap is much more disruptive: Jaccard falls to 0.312, close to
the random baseline of 0.164, and the predicted class itself flips in 0.280 of refits. For the
MLP, the KernelSHAP noise floor is 0.876, but retraining stability is only 0.313 (seed) and
0.446 (bootstrap). The instability is therefore far above measurement noise. Meanwhile overall
accuracy barely moves: XGBoost's pooled accuracy across seeds ranges only from 0.509 to 0.516
(`retrain_summary.csv`). Different seeds yield different explanations of equally (un)skilled
models.

Again the skill-less control is about as stable as the real models (seed 0.593 vs 0.624). There
is one aggregate signal of learning, though. Across folds the real XGBoost consistently ranks
`ret_lag1` in its global TreeSHAP top-3 (0.927 of folds), while the control has no such
feature: its most frequent, `ret_5d`, reaches only 0.418 (`global_feature_ranks.csv`).
Per-prediction drivers are unstable, but a coarse global ranking carries some information.

![Share of explanations (base + 5 seeds) whose top-1 feature is each feature, by regime.](figures/fig6_top1_feature_frequency.png){width=100%}

## 5.4 RQ-c: small input nudges change the top driver

![P(top-1 feature changes) under ε × std Gaussian input noise. Dashed: the explainer's noise floor (P(top-1 changes) between two seeds on the unperturbed input).](figures/fig4_perturbation_vs_noise_floor.png){width=100%}

| model | method | ε = 0.01 | ε = 0.05 | noise floor | excess at ε = 0.05 [95% CI] | prediction flips (ε = 0.05) |
|---|---|---|---|---|---|---|
| xgb | TreeSHAP | 0.096 | 0.280 | 0.000 | 0.280 | 0.136 |
| xgb | KernelSHAP | 0.095 | 0.232 | 0.043 | 0.189 [0.165, 0.214] | 0.133 |
| xgb | LIME | 0.247 | 0.273 | 0.238 | 0.036 [0.021, 0.052] | 0.133 |
| lgbm | TreeSHAP | 0.089 | 0.278 | 0.000 | 0.278 | 0.142 |
| mlp | KernelSHAP | 0.078 | 0.175 | 0.068 | 0.106 [0.081, 0.130] | 0.059 |
| mlp | LIME | 0.255 | 0.345 | 0.231 | 0.114 [0.091, 0.136] | 0.059 |
| xgb_shuffled | TreeSHAP | 0.110 | 0.298 | 0.000 | 0.298 | 0.145 |

Table: P(top-1 feature changes) relative to the unperturbed explanation, 550 points each. The excess over the noise floor is paired per point. Source: `perturbation_summary.csv`, `perturbation_excess_over_noise_floor.csv`.

A perturbation of 5% of one standard deviation, far below a day's typical movement in any
feature, changes the TreeSHAP top-1 feature for 0.280 (xgb) and 0.278 (lgbm) of points. Even
1% changes it for 0.096 and 0.089. For the tree models the explanation changes about twice as
often as the predicted class does (0.136 at ε = 0.05). For LIME the perturbation effect is
mostly hidden inside LIME's own noise. At ε = 0.01 its excess over the noise floor is 0.010
(xgb) and 0.005 [−0.002, 0.013] (lgbm), meaning a LIME top-1 change at that size is almost
entirely sampling noise. KernelSHAP has a low floor, and its excess (0.189 for xgb at
ε = 0.05) is real. The control is again as fragile as the real models (0.298).

## 5.5 RQ-d (key question): is instability worse in volatile regimes? No.

![Volatile − calm differences for all 56 tests. Lines: 95% cluster-bootstrap CI. Blue ticks: placebo 95% range. Red: BH q < 0.05. Lower panel: positive = less stable when volatile.](figures/fig5_volatile_minus_calm_forest.png){width=100%}

| headline test (pooled xgb, lgbm, mlp) | calm | volatile | volatile − calm [95% CI] | p (perm.) | q (BH) | p (placebo) |
|---|---|---|---|---|---|---|
| retraining (seed), 1 − Jaccard | 0.497 | 0.490 | −0.007 [−0.022, 0.009] | 0.492 | 0.758 | 0.507 |
| retraining (bootstrap), 1 − Jaccard | 0.648 | 0.631 | −0.017 [−0.033, −0.001] | 0.064 | 0.258 | 0.093 |
| perturbation (primary, ε = 0.05), P(top-1 changes) | 0.257 | 0.228 | −0.029 [−0.060, 0.003] | 0.029 | 0.150 | 0.133 |
| agreement (all pairs), 1 − Jaccard | 0.569 | 0.549 | −0.020 [−0.041, 0.002] | 0.102 | 0.336 | 0.130 |

Table: Pooled headline regime tests (930 calm / 720 volatile point-model rows; agreement 2170 / 1680 point-pairs). Source: `regime_tests.csv`.

Across the full family of 56 tests, 12 have a raw permutation p < 0.05 and 4 survive BH
correction at q < 0.05 (`key_numbers.json`):

* accuracy, xgb: volatile − calm = +0.036 (q = 0.028);
* accuracy, lgbm: +0.035 (q = 0.028);
* retraining (seed), mlp: 1 − Jaccard −0.026 (q = 0.028), i.e. **more** stable when volatile;
* perturbation, mlp/LIME, ε = 0.05: P(top-1 changes) −0.109 (q = 0.028), i.e. **more** stable when volatile.

**No explanation-instability measure is significantly higher in volatile regimes.** Only 9 of the 52 explanation differences are even positive. All four
pooled headline differences are negative (slightly *more* stable when volatile). They lie
inside or near the placebo range, and none survives correction. The noise floors themselves
do not differ by regime either (no noise-floor test is significant), so regime is not
confounded with explainer noise. The two significant explanation differences (MLP) go
*against* the hypothesis. One plausible reading: in volatile periods the inputs are larger
in magnitude, so the strongest feature stands out more clearly. We did not test that
mechanism.

The control's regime differences are all non-significant (smallest q 0.570 among its
explanation tests). Since the only robust regime effect is in **accuracy** and not in
**explanation stability**, the models' modest extra skill when volatile did not come with
measurably more (or less) stable explanations.

![Placebo null distributions (regime labels borrowed from a different fold) for the four headline tests, with the observed difference.](figures/fig7_placebo_null.png){width=100%}

# 6 Limitations

* **Close-only data.** Coin Metrics Community data have no open/high/low. Range-based
  volatility (Parkinson, Garman–Klass) and candle features, which are common in the
  literature, are absent. Our realised-volatility regime uses close-to-close returns only.
* **Data source.** Coin Metrics `PriceUSD` is a reference rate at 00:00 UTC aggregated across
  exchanges, not an exchange's own close. Reported spot volume is known to include
  wash-trading noise for some assets. Results may differ for exchange-specific data.
* **Coin set.** LTC replaces the planned SOL, whose volume history is missing in this dataset.
  The five coins are strongly co-moving, so the 5 × 11 clusters are not fully independent
  across coins, and our cluster CIs may still be somewhat optimistic.
* **Small explained sample.** 10 points per (coin, fold), 550 in total (240 volatile), and 3
  repetitions for the expensive explainers. With these sizes the 95% CIs on regime
  differences have half-widths from 0.016 to 0.072. The run took only a few minutes, so larger samples are
  feasible, and smaller regime effects than this cannot be ruled out.
* **Correlated features.** The MACD trio is exactly collinear, and the return-based features
  (`ret_lag1`, `ret_5d`, `ret_10d`) overlap. SHAP and LIME can split credit among correlated
  features arbitrarily, which inflates top-k disagreement and instability. A grouped analysis
  (`FEATURE_GROUPS` in `features.py`) would likely look more stable. We report the
  feature-level view because that is what applied papers show.
* **Model and explainer settings.** One configuration per model, one regime definition
  (training-median `vol_30`) and one perturbation scheme (independent Gaussian noise plus
  recomputed `macd_hist`). KernelSHAP used the shap default `l1_reg = num_features(10)`,
  which zeroes 2 of 12 attributions per point. TreeSHAP is path-dependent while KernelSHAP is
  interventional, so part of their disagreement is definitional.
* **Little model skill.** With accuracy near 0.5, much of the instability we measure is the
  instability of explaining noise. Our conclusions apply to models of this quality, which
  are typical for next-day crypto direction. They need not hold for a model with real skill.

# 7 Conclusion

For next-day direction models on five cryptocurrencies, feature-importance explanations fail
basic trust checks. Methods disagree: LIME shares about a third of its top-3 features with
SHAP and is noisy even against itself. The top driver changes when only the random seed
changes (TreeSHAP, whose noise floor is zero) and when inputs move by 5% of a standard
deviation. A model trained on shuffled labels produces explanations that are just as
agreeable and just as stable as those of the real models. The models themselves are at best
marginally better than a coin flip, and only in volatile regimes. On our key question we
find **no evidence that instability is worse in volatile regimes**. If anything, the
estimates lean slightly the other way, and the only robust regime effect is in accuracy.
The practical implication: per-prediction SHAP or LIME claims from such models should not be
reported as facts about the market in any regime. At minimum, papers should report
explanation stability across seeds and resamples, the explainer's noise floor, and a
no-skill control next to any feature-importance plot.

# References

[1] S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. *NeurIPS 30*, 2017, 4765–4774.

[2] D. Alvarez-Melis and T. S. Jaakkola. On the robustness of interpretability methods. *ICML Workshop on Human Interpretability in ML (WHI)*, 2018. arXiv:1806.08049.

[3] D. Slack, S. Hilgard, E. Jia, S. Singh, and H. Lakkaraju. Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. *AIES*, 2020, 180–186.

[4] M. T. Ribeiro, S. Singh, and C. Guestrin. "Why should I trust you?": Explaining the predictions of any classifier. *KDD*, 2016, 1135–1144. (Introduces LIME; not a stability study.)

[5] Z. Zhou, G. Hooker, and F. Wang. S-LIME: Stabilized-LIME for model explanation. *KDD*, 2021. arXiv:2106.07875.

[6] T. Chen and C. Guestrin. XGBoost: A scalable tree boosting system. *KDD*, 2016, 785–794.

[7] J. N. Pinky and R. Akula. Enhancing cryptocurrency market forecasting: Advanced machine learning techniques and industrial engineering contributions. arXiv:2410.14475, 2024.

[8] A. Ghorbani, A. Abid, and J. Zou. Interpretation of neural networks is fragile. *AAAI*, 2019, 3681–3688.

[9] C. Molnar. *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*, 2nd ed., 2022. https://christophm.github.io/interpretable-ml-book/

[10] S. Krishna, T. Han, A. Gu, S. Wu, S. Jabbari, and H. Lakkaraju. The disagreement problem in explainable machine learning: A practitioner's perspective. arXiv:2202.01602, 2022.

[11] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2, 2020, 56–67.

[12] Y. Benjamini and Y. Hochberg. Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B* 57(1), 1995, 289–300.

[13] H. R. Künsch. The jackknife and the bootstrap for general stationary observations. *Annals of Statistics* 17(3), 1989, 1217–1241.

[14] Coin Metrics. Coin Metrics Community Data. https://github.com/coinmetrics/data (accessed 2026).

[15] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu. LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*, 2017.

[16] L. Breiman. Random forests. *Machine Learning* 45, 2001, 5–32.

*Note on the proposal's reference list.* Krishna et al. [10], the core reading for the
disagreement analysis (proposal Week 6), was missing and has been added. In the proposal's
timeline, [5] (S-LIME, a stability method) and [7] (the crypto forecasting paper) appear
swapped between Weeks 4 and 5: [7] belongs with model training (Week 4) and [5] with the
explanation and stability weeks. Ribeiro et al. [4] is cited for the LIME method itself, not
as evidence about stability. The moving-block bootstrap follows Künsch [13].
