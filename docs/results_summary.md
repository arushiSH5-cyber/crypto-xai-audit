# Results summary (one page)

**Project:** Can SHAP/LIME explanations of crypto next-day direction models be trusted?
Arushi Nirmal, San José State University, CS 180, Fall 2026.

**Setup in one sentence.** XGBoost, LightGBM and a small MLP (plus an XGBoost trained on shuffled
labels as a no-skill control) predict next-day up/down for BTC, ETH, ADA, LINK and LTC, under
walk-forward validation (55 folds, 9835 test days per model). On 550 explained test days
(310 calm, 240 volatile) we measure explainer agreement, retraining stability and perturbation
stability, and compare each one with the explainer's own seed-to-seed noise floor.

## Findings

1. **The models barely beat a coin flip.** Accuracy is 0.512 (xgb), 0.512 (lgbm) and
   0.507 (mlp), against 0.502 for the shuffled-label control and 0.504 for "always up". The
   small edge appears only in volatile periods: xgb gets 0.538 there vs 0.502 when calm.
2. **The explainers disagree.** TreeSHAP and KernelSHAP share 0.686 of their top-3 features
   (xgb, Jaccard). LIME shares only 0.324 with TreeSHAP, and only 0.638 with *itself* across
   seeds. Two random rankings would share 0.164.
3. **Explanations change when the model is retrained.** TreeSHAP has zero explainer noise, yet
   its top-3 Jaccard across 6 seeds is 0.624 (xgb), and across block bootstraps 0.312. The
   no-skill control is about as stable (0.593 across seeds), so **stability does not show that
   a model learned anything.**
4. **Small input nudges change the top driver.** With noise of 5% of a standard deviation, the
   TreeSHAP top-1 feature changes for 0.280 of points (xgb), while the predicted class flips
   for only 0.136.
5. **Key question, a null result: instability is *not* worse in volatile regimes.** No
   explanation-instability test is significantly higher in volatile periods after
   Benjamini–Hochberg correction (52 tests). Only 9 of the 52 differences are even positive,
   and the two significant ones (MLP) go the other way. The one robust regime effect is in
   **accuracy**, not in explanation stability.

## Key table: volatile − calm, pooled over xgb, lgbm and mlp

| Measure (higher = less stable) | Calm | Volatile | Difference [95% CI] | p (perm.) | q (BH) | p (placebo) |
|---|---|---|---|---|---|---|
| Retraining, seeds: 1 − Jaccard | 0.497 | 0.490 | −0.007 [−0.022, 0.009] | 0.492 | 0.758 | 0.507 |
| Retraining, bootstraps: 1 − Jaccard | 0.648 | 0.631 | −0.017 [−0.033, −0.001] | 0.064 | 0.258 | 0.093 |
| Perturbation ε = 0.05: P(top-1 changes) | 0.257 | 0.228 | −0.029 [−0.060, 0.003] | 0.029 | 0.150 | 0.133 |
| Method agreement: 1 − Jaccard | 0.569 | 0.549 | −0.020 [−0.041, 0.002] | 0.102 | 0.336 | 0.130 |
| *For contrast:* accuracy (xgb) | 0.502 | 0.538 | +0.036 [0.018, 0.057] | 0.0005 | 0.028 | 0.0005 |

Source: `results/tables/regime_tests.csv`. CIs are cluster bootstraps over (coin, fold);
p-values come from within-fold label permutations (0.0005 = 1/2001 is the smallest p possible
with 2000 permutations: none was as extreme); q is Benjamini–Hochberg over all 56 tests.

## Key figure

![Volatile − calm for all 56 tests. Lines: 95% CI. Blue ticks: placebo 95% range. Red: q < 0.05. Lower panel: positive = less stable when volatile.](../results/figures/fig5_volatile_minus_calm_forest.png)

## Bottom line

For next-day crypto direction models, a per-prediction SHAP or LIME "top driver" is not a reliable
statement about the market. It depends on the explainer, the random seed, the training sample and
tiny input changes, and a model with no skill produces equally "stable" explanations. Volatility
does not make this measurably worse. The main caveats are close-only data, 550 explained points,
correlated features (the MACD trio), and models with almost no skill. Full details are in
`report/report.pdf`.
