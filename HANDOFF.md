# Handoff: what's done and what's left

Project: CS 180 research — auditing stability and agreement of SHAP/LIME explanations
for crypto next-day direction models (see the proposal PDF). Deadline: today.

## Done
- `data/raw/` — Coin Metrics Community Data daily CSVs (github.com/coinmetrics/data, `csv/<asset>.csv`)
  for btc, eth, ada, link, ltc. Columns used: `PriceUSD`, `volume_reported_spot_usd_1d`.
  Close-only (no OHLC). SOL was dropped because its volume history is missing in this dataset.
- `src/xaiaudit/config.py` — every setting, plus `QUICK` overrides for smoke tests.
- `src/xaiaudit/data.py` — loader; refuses gaps/duplicates.
- `src/xaiaudit/features.py` — 12 causal features + `target` = 1[next-day log return > 0].
  MACD trio is exactly collinear by design (credit-splitting is part of the story).
- `src/xaiaudit/validation.py` — expanding walk-forward folds (refit every 182 days, test from
  2021-01-01, 1-day embargo); regime = `vol_30` above the fold's TRAINING median → "volatile".
- `src/xaiaudit/models.py` — xgb, lgbm, mlp (sklearn MLPClassifier as the small neural net).
- `src/xaiaudit/explainers.py` — TreeSHAP (path-dependent, log-odds), KernelSHAP (logit link,
  k-means background), LIME (default discretisation), permutation importance (ROC-AUC, global).
- `src/xaiaudit/metrics.py` — top-k Jaccard, Spearman on |attr|, top-1 agreement, set stability.

## Install note
`pip install lime` fails to build on Python 3.13. Workaround used: download the sdist
(`pip download --no-deps --no-binary :all: lime`), extract, copy the `lime/` folder into
site-packages. Everything else: `pip install -r requirements.txt`.

## Left to build (planned design)
1. `src/xaiaudit/experiments.py` — `run_coin(coin, cfg)`, parallel over coins (joblib, n_jobs=2):
   - Per fold: fit each model at `base_seed`, plus a control `xgb_shuffled` (training labels
     permuted) so we can show what explanations of a skill-less model look like.
   - Save all test predictions (coin, model, fold, date, regime, y, p).
   - Sample `points_per_fold` test days, half per regime.
   - Agreement: per point, pairwise metrics among TreeSHAP / KernelSHAP / LIME; globally,
     Spearman of permutation importance vs mean|SHAP| and mean|LIME|.
   - Noise floor: rerun KernelSHAP and LIME `noise_floor_reps` times with different seeds on the
     identical model + input; TreeSHAP should be exactly reproducible (assert it).
   - Retraining stability: primary method (TreeSHAP for trees, KernelSHAP for mlp) across
     `retrain_seeds` and `bootstrap_reps` moving-block bootstraps (block 30) of training rows.
     Also record prediction spread across retrains.
   - Perturbation: for each eps in `perturb_eps`, add eps × training-std Gaussian noise;
     compare to the unperturbed explanation (top-1 changed rate, Jaccard, Spearman) and record
     prediction flip rate and |Δp|.
   - Write tidy CSVs to `results/cache/` and raw attributions to a pickle.
2. `src/xaiaudit/analysis.py` — aggregate by regime with bootstrap CIs; volatile − calm
   difference with permutation p-values and Benjamini–Hochberg correction; placebo test
   (random regime relabelling, `placebo_reps`) as the measurement noise floor; compare every
   instability number against the explainer's own noise floor and against the shuffled-label
   control; accuracy/AUC per regime vs 50%.
   Figures: accuracy by regime, method-agreement heatmap, retrain stability by regime,
   perturbation vs noise floor, volatile−calm forest plot (key figure), top-1 feature
   frequency across seeds, placebo null distribution.
3. `run_all.py` with `--quick`.
4. `tests/`: no look-ahead in features (truncate series, past rows unchanged), folds are
   time-ordered and non-overlapping, regime threshold uses train only, metric unit tests,
   TreeSHAP determinism, quick end-to-end run reproduces identical numbers.
5. Docs: `docs/01_problem_statement.md` + lit survey, `docs/design.md` (UML class + sequence,
   Mermaid), `docs/test_cases.md`, `notebooks/run_pipeline.ipynb` (Colab), `report/report.md`
   → PDF via pandoc, `docs/presentation_script.md`.

## Key honesty points for the report
- Next-day direction accuracy will likely be ~50–55%; always report skill next to stability,
  and contrast with the shuffled-label control. Unstable explanations of a no-skill model are
  explanations of noise.
- Separate real instability from explainer noise using the noise floor.
- Fix proposal references: add Krishna et al. (2022), "The Disagreement Problem in Explainable
  Machine Learning: A Practitioner's Perspective", arXiv:2202.01602 (the core Week 6 paper).
  S-LIME [5] and the crypto forecasting paper [7] look swapped between Weeks 4 and 5;
  Ribeiro et al. [4] introduces LIME but is not itself a stability study.
