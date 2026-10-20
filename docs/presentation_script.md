# Presentation script: recorded code review (about 10 minutes)

**Before recording.** Open the repo in your editor. Open a terminal at the repo root with the venv
active (`source .venv/bin/activate`). Open `results/figures/` in an image viewer and
`report/report.pdf` in a PDF viewer. Do a dry run of `python run_all.py --quick` once, so caches
are warm. Times are cumulative targets.

## On-screen checklist

| Time | Segment | Show on screen |
|---|---|---|
| 0:00 | 1. Motivation | `README.md` (top) |
| 0:45 | 2. Data and features | `data/raw/btc.csv` (first lines), `src/xaiaudit/data.py`, `src/xaiaudit/features.py` |
| 1:45 | 3. No-leakage design | `src/xaiaudit/validation.py`, `tests/test_features.py`, `tests/test_validation.py` |
| 2:45 | 4. Models | `src/xaiaudit/models.py`, `results/figures/fig1_accuracy_by_regime.png` |
| 3:30 | 5. Explainers | `src/xaiaudit/explainers.py`, `src/xaiaudit/metrics.py` |
| 4:15 | 6. The four audits | `src/xaiaudit/experiments.py` (`audit_model`), `fig2`, `fig3`, `fig4` |
| 6:30 | 7. Key regime result | `src/xaiaudit/analysis.py` (docstring), `fig5_volatile_minus_calm_forest.png`, `fig7_placebo_null.png` |
| 8:00 | 8. Limitations | `report/report.pdf`, Section 8 |
| 8:45 | 9. Live demo: quick run | terminal: `python run_all.py --quick`, then `results/quick/tables/regime_tests.md` |
| 9:45 | Close | `README.md` (Reproduce section) |

---

## 1. Motivation (0:00 – 0:45) · show `README.md`

"Crypto forecasting papers often end with a SHAP plot that says something like 'RSI and lagged
returns drive the price'. My question is: can we trust that plot? I train standard next-day
direction models for five coins and audit their explanations four ways. Do the methods agree?
Do the drivers survive retraining? Do they survive a tiny input nudge? And the key question: is
it worse in volatile markets? Two baselines run through everything. One is the explainer's own
noise floor. The other is a model trained on shuffled labels, which by construction has learned
nothing."

## 2. Data and features (0:45 – 1:45) · show `data/raw/btc.csv`, `data.py`, `features.py`

"The data are Coin Metrics Community daily CSVs for BTC, ETH, ADA, LINK and LTC. I use only the
price and volume columns, so it's close-only, with no open/high/low. LTC replaces SOL because
SOL's volume history is missing. The loader refuses gaps instead of forward-filling them."

"Twelve features: three lagged returns, 5- and 10-day returns, RSI, the MACD trio, 7- and 30-day
volatility, and a volume ratio. The MACD histogram is exactly MACD minus its signal. I kept that
on purpose, because correlated features are part of the story. The target is whether
*tomorrow's* return is positive."

## 3. No-leakage design (1:45 – 2:45) · show `validation.py`, then the two test files

"Validation is an expanding walk-forward: six-month test blocks starting in 2021, training on
everything before them with a one-day embargo, never shuffled. That's 11 folds per coin, 55 in
total."

"A day is 'volatile' if its 30-day volatility is above the median of the *training* window, so
the label is known on the day you predict."

"Tests back this up. `test_no_lookahead_truncation` cuts the series at five points and checks
that every past feature is identical. `test_future_shock_does_not_leak` triples future prices and
checks nothing earlier moves. The validation tests check that folds are ordered and disjoint, and
that corrupting future data can't move the regime threshold."

## 4. Models (2:45 – 3:30) · show `models.py`, then `fig1_accuracy_by_regime.png`

"Three models: XGBoost, LightGBM and a small scikit-learn MLP, plus the control: the same
XGBoost trained on shuffled labels. Here is the honest context. Accuracy is 0.512 for XGBoost and
LightGBM and 0.507 for the MLP. The control gets 0.502, and always predicting 'up' gets 0.504. In
calm markets every model is a coin flip. The only edge is in volatile markets, where XGBoost gets
0.538. So most of what we explain is noise, and I report accuracy next to every stability number."

## 5. Explainers (3:30 – 4:15) · show `explainers.py`, `metrics.py`

"TreeSHAP is exact and deterministic. KernelSHAP and LIME are sampling-based, so they're random.
Permutation importance is global. They output different units, so I compare *rankings*: the
main metric is top-3 Jaccard, the overlap of the three most important features. I also use
Spearman, top-1 agreement, and how often the top-1 feature changes. For scale: two random
rankings would get a top-3 Jaccard of 0.164."

## 6. The four audits (4:15 – 6:30) · show `experiments.py` → `audit_model`, then figures 2, 3, 4

"For each coin, fold and model I pick 10 test days, half calm and half volatile when possible.
That's 550 points, and all four audits use the same points."

*Show `fig2_method_agreement_heatmap.png`.*
"**Agreement.** The two SHAP methods share about two-thirds of their top three (0.686 for
XGBoost). LIME shares about a third (0.324). But LIME doesn't even agree with *itself*: run it
twice on the same point and the overlap is 0.638."

*Show the noise-floor block in `audit_model`.*
"**Noise floor.** KernelSHAP and LIME are re-run with new seeds on the exact same input. TreeSHAP
is re-run and *asserted* identical. The run would crash if it weren't."

*Show `fig3_retrain_stability_by_regime.png`.*
"**Retraining.** TreeSHAP has zero noise, yet changing only the XGBoost seed leaves a top-3
overlap of 0.624. Block-bootstrapping the training data drops it to 0.312. The shuffled-label
model is just as stable, at 0.593. So stability tells you nothing about whether the model learned
anything."

*Show `fig4_perturbation_vs_noise_floor.png`.*
"**Perturbation.** Adding noise of 5% of a standard deviation changes the TreeSHAP top feature
for 0.280 of points, while the prediction flips for only 0.136. For LIME, the dashed noise floor
shows that most of the change is LIME's own randomness."

## 7. Key regime result (6:30 – 8:00) · show `analysis.py` docstring, then `fig5`, then `fig7`

"Statistics: points in one fold share a model, so I use a cluster bootstrap over the 55
coin-folds. p-values come from shuffling regime labels *inside* each fold, and Benjamini–Hochberg
corrects across 56 tests. The placebo gives each fold the regime labels of a different half-year,
which shows how big a fake regime effect can look."

*Show `fig5_volatile_minus_calm_forest.png`.*
"Positive means less stable when volatile. Of 52 explanation tests, none is significantly
positive. Only 9 are positive at all. The pooled headline differences are all slightly negative,
for example −0.007 for seed retraining with a CI from −0.022 to 0.009. The two significant
explanation results, both MLP, go the *other* way. The red rows at the top are accuracy: the
models really are better in volatile markets."

*Show `fig7_placebo_null.png`.*
"And every headline difference sits inside its placebo range. My answer to the key question is a
clear null: volatility does not make these explanations less stable."

## 8. Limitations (8:00 – 8:45) · show `report.pdf`, Section 8

"Limitations: next-day direction is nearly random, so we're explaining near-noise models.
Close-only data from one source. LTC instead of SOL. Only 550 explained points, so effects
smaller than a CI half-width of 0.016 to 0.072 can't be ruled out. And correlated features like
the MACD trio let SHAP split credit arbitrarily."

## 9. Live demo (8:45 – 9:45) · terminal

```bash
python run_all.py --quick
```

"This is the smoke test: two coins from 2024, smaller settings. Both coins run in parallel. It
takes about 16 seconds (16.2 seconds in my clean-checkout check) and writes `results/quick/`."

*Open `results/quick/tables/regime_tests.md`.* "Same tables, same tests, on the small run. The
full run is the same command without `--quick` and takes about five minutes (310.3 seconds).
Everything is seeded, and a fresh clone reproduced every committed result file byte-for-byte."

## Close (9:45 – 10:00) · show README "Reproduce" section

"Takeaway: for next-day crypto models, per-prediction SHAP or LIME stories shouldn't be read as
facts about the market, in any regime. If you publish one, also report seed stability, the
explainer's noise floor and a no-skill control. Thanks."
