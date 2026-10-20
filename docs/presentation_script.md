# Presentation script: code and results walkthrough (about 10 minutes)

Setup before recording: open the repo in your editor, a terminal at the repo root with the venv
active, and `report/report.pdf` (or the PNGs in `results/figures/`) in a viewer. Times are
cumulative targets. **[SHOW]** lines say what is on screen. Everything else is spoken.

---

## 0:00 – 1:00 · The question

**[SHOW]** README.md, top section.

"Crypto forecasting papers often end with a SHAP plot that says something like 'RSI and
lagged returns drive the price'. My project asks whether we can trust that plot. I train
standard next-day direction models for five coins: Bitcoin, Ethereum, Cardano, Chainlink and
Litecoin. Then I audit their explanations four ways. One, do the explanation methods agree?
Two, do the drivers stay the same when I retrain the model? Three, do they stay the same
when I nudge the inputs slightly? And four, the key question: is all of this worse in
volatile markets than in calm ones?"

"Two controls run through everything. One is the explainer's own noise floor: the same
model and input, explained again with a new random seed. The other is a model trained on
shuffled labels, which by construction has learned nothing."

## 1:00 – 2:30 · Data, features, no look-ahead

**[SHOW]** `src/xaiaudit/data.py`, then `features.py`.

"The data are Coin Metrics Community daily CSVs, price and volume only, so there's no
open/high/low. The loader refuses gaps rather than silently forward-filling them."

"There are twelve features, all computed from information available at the end of day *t*:
lagged returns, RSI, the MACD trio, two realised volatilities and a volume ratio. The target
is whether *tomorrow's* return is positive. `next_ret` is the only forward-looking column."

**[SHOW]** `tests/test_features.py::test_no_lookahead_truncation`.

"This test proves causality. I build features on the full series and on the series cut at
five different points. Every past row has to come out identical. A second test triples all
prices from day *t* onward and checks that no earlier feature changes."

## 2:30 – 3:30 · Walk-forward validation and regimes

**[SHOW]** `src/xaiaudit/validation.py`.

"Validation is an expanding walk-forward. Each fold tests on six months, starting in January
2021. It trains on everything before that, minus a one-day embargo. That gives 11 folds per
coin, 55 in total, with no shuffling anywhere."

"A day is 'volatile' if its 30-day volatility is above the median of the *training* window.
So the regime is known on the day you predict. Tests check that the folds are ordered and
disjoint, and that corrupting future data can't move the threshold."

## 3:30 – 5:00 · The four audits in code

**[SHOW]** `src/xaiaudit/experiments.py`: scroll through `audit_model`.

"For every coin, fold and model (XGBoost, LightGBM, an MLP, and the shuffled-label control)
I fit at seed 0 and save every prediction. Then I pick 10 test days per fold, half calm and
half volatile."

"Block 1 is agreement: TreeSHAP, KernelSHAP and LIME on the same points, plus permutation
importance globally."

"Block 2 is the noise floor. KernelSHAP and LIME are re-run with new seeds. TreeSHAP is re-run
and *asserted* bit-identical, and the run would crash if it weren't."

"Block 3 is retraining: five new seeds and five block bootstraps of the training data,
re-explained each time."

"Block 4 is perturbation: Gaussian noise at 1% and 5% of each feature's standard deviation.
One detail: perturbation rep *r* reuses the explainer seed of noise-floor rep *r*, so any
difference between them comes from the input change alone."

**[SHOW]** terminal: `python run_all.py --quick` (about 20 s).

"Coins run in parallel. Seeds come from a CRC of (seed, coin, fold, task), so two runs give
bit-identical CSVs, and there's a test for that. The full run takes about five minutes."

## 5:00 – 5:45 · Statistics

**[SHOW]** `src/xaiaudit/analysis.py`, top docstring.

"Points in the same fold share a model and a time period, so I treat each coin-fold as a
cluster. Confidence intervals are cluster bootstraps. Regime p-values permute the labels
*within* each fold. Benjamini–Hochberg controls false discoveries over all 56 tests. And there
is a placebo: each fold borrows the regime labels of a *different* half-year. That shows how
big a 'regime effect' appears by pure chance."

## 5:45 – 6:30 · Result 1: the models barely work

**[SHOW]** `fig1_accuracy_by_regime.png`.

"First, context. Pooled accuracy is 0.512 for XGBoost and LightGBM and 0.507 for the MLP. The
shuffled control gets 0.502. In calm markets every model is a coin flip. The only edge is in
volatile markets, where XGBoost gets 0.538. So most of what we explain is noise. I report that
next to every stability number."

## 6:30 – 7:30 · Result 2: methods disagree; LIME disagrees with itself

**[SHOW]** `fig2_method_agreement_heatmap.png`.

"The two SHAP variants share about two-thirds of their top three features. LIME shares about
a third with either one, where random rankings would give 0.16. But LIME doesn't even agree
with *itself*: run it twice on the same point and the top-3 overlap is only about 0.64.
Permutation importance agrees with SHAP even less, because with no skill, no feature matters
for accuracy."

## 7:30 – 8:30 · Result 3: retraining and perturbation

**[SHOW]** `fig3_retrain_stability_by_regime.png`, then `fig4_perturbation_vs_noise_floor.png`.

"TreeSHAP has zero noise, yet changing only the XGBoost seed leaves a top-3 overlap of 0.62.
Bootstrapping the training years drops it to 0.31. That is real model sensitivity. And the
shuffled-label model is just as stable, so stability tells you nothing about whether a model
learned anything."

"A 5% nudge to the inputs changes the TreeSHAP top driver for 28% of points. That's about twice
as often as the prediction itself flips. For LIME, most changes sit inside its own noise floor,
the dashed line."

## 8:30 – 9:30 · Result 4: the key question

**[SHOW]** `fig5_volatile_minus_calm_forest.png`, then `fig7_placebo_null.png`.

"Is instability worse in volatile regimes? Positive means less stable when volatile. Of 52
explanation tests, none is significantly positive after correction. The four pooled headline
differences are all slightly *negative* and sit within the placebo range. The two significant
explanation results, both for the MLP, go the *other* way. The only robust regime effect is
accuracy. So my answer to the key question is a clear null: no evidence that volatility makes
explanations less stable."

## 9:30 – 10:00 · Limitations and takeaway

**[SHOW]** report, Limitations section.

"Limitations: close-only data from one source, LTC instead of SOL, 550 explained points, and
correlated features like the MACD trio that let SHAP split credit arbitrarily."

"Takeaway: for next-day crypto models, per-prediction SHAP or LIME stories shouldn't be read as
facts about the market, in any regime. If you publish one, also report seed stability, the
explainer's noise floor, and a no-skill control. Everything reproduces with one command:
`python run_all.py`. Thanks."
