# Test cases

All tests are automated with pytest (`tests/`). Run them from the repo root:

```bash
python -m pytest -q             # all 25 tests
python -m pytest -q -m "not slow"   # skip the end-to-end reproducibility test (TC-20)
```

## Actual run

Output of `python -m pytest -v --durations=5` on the final code (Python 3.12.15). The full log is saved in `results/pytest_output.txt`; the fresh-clone check below is in `results/reproduction_check.log`.

```
test_explainers.py::test_treeshap_deterministic_and_additive[xgb] PASSED
test_explainers.py::test_treeshap_deterministic_and_additive[lgbm] PASSED
test_explainers.py::test_stochastic_explainers_reproducible_with_same_seed PASSED
test_features.py::test_no_lookahead_truncation[400] PASSED
test_features.py::test_no_lookahead_truncation[900] PASSED
test_features.py::test_no_lookahead_truncation[1500] PASSED
test_features.py::test_no_lookahead_truncation[2500] PASSED
test_features.py::test_no_lookahead_truncation[3000] PASSED
test_features.py::test_future_shock_does_not_leak PASSED
test_features.py::test_target_is_next_day_direction PASSED
test_features.py::test_macd_trio_is_collinear PASSED
test_metrics.py::test_top_k_uses_absolute_values_and_stable_ties PASSED
test_metrics.py::test_jaccard PASSED
test_metrics.py::test_rank_corr PASSED
test_metrics.py::test_pair_and_set_stability PASSED
test_metrics.py::test_benjamini_hochberg_matches_reference PASSED
test_metrics.py::test_cluster_bootstrap_ci_contains_mean PASSED
test_metrics.py::test_regime_test_detects_real_gap_and_not_a_null PASSED
test_pipeline.py::test_sample_points_balanced_and_deterministic PASSED
test_pipeline.py::test_block_bootstrap_shape PASSED
test_pipeline.py::test_perturb_keeps_feature_identities PASSED
test_pipeline.py::test_two_quick_runs_identical PASSED
test_validation.py::test_folds_time_ordered_and_disjoint PASSED
test_validation.py::test_regime_threshold_uses_training_data_only PASSED
test_validation.py::test_fold_context_threshold_ignores_future PASSED
============================= slowest 5 durations ==============================
27.06s call     tests/test_pipeline.py::test_two_quick_runs_identical
0.25s call     tests/test_metrics.py::test_regime_test_detects_real_gap_and_not_a_null
0.08s call     tests/test_explainers.py::test_stochastic_explainers_reproducible_with_same_seed
0.07s call     tests/test_explainers.py::test_treeshap_deterministic_and_additive[lgbm]
0.06s call     tests/test_explainers.py::test_treeshap_deterministic_and_additive[xgb]
============================= 25 passed in 30.69s ==============================
```

## Test-case table

| ID | Test (`tests/…::function`) | What it checks | Why it matters | Expected result | Actual |
|---|---|---|---|---|---|
| TC-01 | `test_features.py::test_no_lookahead_truncation` (5 cases: cut at row 400, 900, 1500, 2500, 3000) | Features built on a truncated BTC series equal the full-series features on every shared date | If removing future data changed a past feature, that feature used the future (look-ahead leakage) | All 12 feature columns identical (tolerance 1e-12) | PASSED (5/5) |
| TC-02 | `test_features.py::test_future_shock_does_not_leak` | Triples every price from day *t* on | A direct check that future prices can't reach past features | Features before *t* unchanged. The label of day *t−1* (return *t−1→t*) becomes 1 | PASSED |
| TC-03 | `test_features.py::test_target_is_next_day_direction` | Recomputes 1[r(t+1) > 0] independently | The label must be *tomorrow's* direction, not today's | Equal on every row; no NaNs | PASSED |
| TC-04 | `test_features.py::test_macd_trio_is_collinear` | `macd_hist = macd − macd_signal` | Documents the deliberate collinearity discussed as a limitation | Exact equality | PASSED |
| TC-05 | `test_validation.py::test_folds_time_ordered_and_disjoint` | Real ETH walk-forward folds | Training on the future or overlapping test sets would invalidate every accuracy number | ≥ 5 folds. Train ends before test starts with a gap ≥ embargo + 1 day. Test blocks disjoint and moving forward. Expanding training window. First test day ≥ 2021-01-01 | PASSED |
| TC-06 | `test_validation.py::test_regime_threshold_uses_training_data_only` | Labels test values against a training median of 3, then adds an extreme test value | The regime must be knowable on the prediction day | `["calm", "volatile"]` both times | PASSED |
| TC-07 | `test_validation.py::test_fold_context_threshold_ignores_future` | Builds a real `FoldContext`, then corrupts all `vol_30` values after the test block | Same as TC-06, on the code path the experiments actually use | Threshold = training median; threshold and labels unchanged | PASSED |
| TC-08 | `test_metrics.py::test_top_k_uses_absolute_values_and_stable_ties` | Ranks `[0.1, −0.9, 0.5, 0.5]` | Rankings must use \|attribution\| and be deterministic on ties | Top-1 = index 1; tie goes to the lower index | PASSED |
| TC-09 | `test_metrics.py::test_jaccard` | Hand-computed top-3 sets | Jaccard is the primary metric | 0.5 for 2 shared of 4, 1.0 identical, 0.0 disjoint | PASSED |
| TC-10 | `test_metrics.py::test_rank_corr` | Spearman on \|attr\| | Secondary metric; must ignore sign and handle constant vectors | +1 identical, +1 sign-flipped, −1 reversed, NaN constant | PASSED |
| TC-11 | `test_metrics.py::test_pair_and_set_stability` | Three vectors, two identical | Checks the multi-explanation stability and top-1 change rate | 2 distinct top-1s, modal share 2/3, pairwise top-1 agreement 1/3, change rate 0.5 | PASSED |
| TC-12 | `test_metrics.py::test_benjamini_hochberg_matches_reference` | p = (0.01, 0.04, 0.03, 0.005) | BH decides which regime effects are "significant" | q = (0.02, 0.04, 0.04, 0.02); all q ≤ 1 | PASSED |
| TC-13 | `test_metrics.py::test_cluster_bootstrap_ci_contains_mean` | 20 clusters × 10 values | CIs throughout the report use this function | lo < mean < hi; mean = sample mean | PASSED |
| TC-14 | `test_metrics.py::test_regime_test_detects_real_gap_and_not_a_null` | 40 null simulations, then one +1 SD shift | The permutation test must not cry wolf, and must catch a real effect | False-positive rate ≤ 15% at α = 0.05; real shift p < 0.01, CI above 0 | PASSED |
| TC-15 | `test_explainers.py::test_treeshap_deterministic_and_additive` (xgb, lgbm) | Explains 20 points twice, and with a refitted same-seed model | TreeSHAP's noise floor is claimed to be exactly 0 | Bit-for-bit identical; margin − ΣSHAP constant across rows | PASSED (2/2) |
| TC-16 | `test_explainers.py::test_stochastic_explainers_reproducible_with_same_seed` | KernelSHAP and LIME with seed 5 twice, LIME with seed 6 | Seeds must control the explainers; a new seed must change them (that's the noise floor) | Same seed identical; new seed differs | PASSED |
| TC-17 | `test_pipeline.py::test_sample_points_balanced_and_deterministic` | 150 calm + 30 volatile days, then 50 + 2 | Point sampling decides the calm/volatile comparison | 5 + 5; same on rerun; shortfall filled from the other regime | PASSED |
| TC-18 | `test_pipeline.py::test_block_bootstrap_shape` | n = 1000, block = 30 | Retraining resamples must be the right size and contiguous | 1000 in-range indices; first block contiguous | PASSED |
| TC-19 | `test_pipeline.py::test_perturb_keeps_feature_identities` | Large-ε noise | Perturbed inputs must stay valid feature vectors | `macd_hist` identity holds; volatilities ≥ 0 | PASSED |
| TC-20 | `test_pipeline.py::test_two_quick_runs_identical` (`slow`) | Runs `run_all.py --quick` twice into separate folders (2 parallel workers) | Results must not depend on luck, process order or the machine's hash seed | Every cache and table CSV exactly equal; all 4 models and both quick coins present | PASSED (27.1 s) |

Runtime guard (not a pytest test): `experiments.audit_model` raises `AssertionError` if TreeSHAP
gives a different answer when re-run on the same model and input during the real experiment.
It never fired in the full run (550 points × 3 tree models).

## Clean-checkout reproduction check

The README steps were run on a **fresh `git clone`** of the repository into an empty folder,
with a new Python 3.12 virtual environment:

| Step | Command | Result |
|---|---|---|
| Install | `pip install -r requirements.txt` (via `uv pip`) | Installed without errors; lime built normally on Python 3.12 |
| Tests | `python -m pytest -q` | `25 passed in 57.03s` (slower than above because another job was running at the same time) |
| Quick run | `python run_all.py --quick` | `experiments done in 13.7s`, `analysis done in 2.5s; total 16.2s` |
| Full run | `python run_all.py` (after deleting the committed `results/`) | `experiments done in 262.0s`, `analysis done in 35.9s; total 297.9s`, `regime tests: 4/56 significant at BH q<0.05` |
| Same numbers? | `git status results/` after the full run | **Yes.** Every regenerated file in `results/cache/`, `results/tables/` and `results/figures/` was byte-identical to the committed version. The only change was `results/run_log.json`, which records the run's own timings |
