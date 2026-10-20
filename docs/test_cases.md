# Test cases

Run all of them with `pytest` from the repo root (about 40 s; `pytest -m "not slow"` skips the
two end-to-end runs). Every test case below is automated, and the file/function is given.

| ID | Area | Test (file::function) | What it does | Expected result |
|---|---|---|---|---|
| TC-01 | Features: no look-ahead | `test_features.py::test_no_lookahead_truncation` (5 cut points) | Builds features on the full BTC series and on the series truncated at row 400/900/1500/2500/3000; compares every shared row. | All 12 feature columns are identical on the shared dates (tolerance 1e-12). |
| TC-02 | Features: no look-ahead | `test_features.py::test_future_shock_does_not_leak` | Triples every price from day *t* onward. | Features before *t* are unchanged. The label of day *t−1* (return *t−1→t*) becomes 1, because the label is allowed to see *t* and the features are not. |
| TC-03 | Target definition | `test_features.py::test_target_is_next_day_direction` | Recomputes 1[log r(t+1) > 0] independently. | Equals `target` on every row, with no NaNs left. |
| TC-04 | Features: collinearity | `test_features.py::test_macd_trio_is_collinear` | Checks `macd_hist = macd − macd_signal`. | Exact (by design, discussed in the report). |
| TC-05 | Walk-forward folds | `test_validation.py::test_folds_time_ordered_and_disjoint` | Builds the real ETH folds. | ≥ 5 folds. Max train date < min test date, gap ≥ embargo + 1 day, test blocks disjoint and moving forward, training window expanding, first test day ≥ `test_start`. |
| TC-06 | Regime threshold | `test_validation.py::test_regime_threshold_uses_training_data_only` | Labels two test values against a train median of 3. Then adds an extreme test value. | `["calm", "volatile"]` both times (test values never move the threshold). |
| TC-07 | Regime threshold | `test_validation.py::test_fold_context_threshold_ignores_future` | Builds a `FoldContext`, then corrupts all `vol_30` values after the test block. | Threshold = median of the training `vol_30`, unchanged, and test-day labels unchanged. |
| TC-08 | Metrics | `test_metrics.py::test_top_k_uses_absolute_values_and_stable_ties` | Ranks `[0.1, −0.9, 0.5, 0.5]`. | Top-1 = index 1 (absolute value). The tie goes to the lower index. |
| TC-09 | Metrics | `test_metrics.py::test_jaccard` | Hand-computed top-3 sets. | 0.5 for overlap {0,2}, 1.0 for identical sets, 0.0 for disjoint sets. |
| TC-10 | Metrics | `test_metrics.py::test_rank_corr` | Spearman on \|attr\|. | +1 for identical, +1 for sign-flipped, −1 for reversed, NaN for a constant vector. |
| TC-11 | Metrics | `test_metrics.py::test_pair_and_set_stability` | Three vectors, two identical. | `n_distinct_top1 = 2`, `modal_top1_share = 2/3`, pairwise `top1_same = 1/3`, `top1_changed` vs reference = 0.5. |
| TC-12 | Statistics | `test_metrics.py::test_benjamini_hochberg_matches_reference` | p = (0.01, 0.04, 0.03, 0.005). | q = (0.02, 0.04, 0.04, 0.02), all q ≤ 1. |
| TC-13 | Statistics | `test_metrics.py::test_cluster_bootstrap_ci_contains_mean` | 20 clusters × 10 normal values. | lo < mean < hi, and the mean equals the sample mean. |
| TC-14 | Statistics | `test_metrics.py::test_regime_test_detects_real_gap_and_not_a_null` | 40 null simulations, then one with a +1 SD shift. | Null false-positive rate ≤ 15% at α = 0.05. The shifted case gives p < 0.01 and CI above 0. |
| TC-15 | TreeSHAP determinism | `test_explainers.py::test_treeshap_deterministic_and_additive[xgb, lgbm]` | Explains 20 points twice, and once with a re-fitted same-seed model. | Bit-for-bit identical arrays. Margin − ΣSHAP is the same constant for every row (local accuracy). |
| TC-16 | Stochastic explainers | `test_explainers.py::test_stochastic_explainers_reproducible_with_same_seed` | KernelSHAP and LIME twice with seed 5, then LIME with seed 6. | Same seed gives identical output. A new seed gives different output (which is why we measure a noise floor). |
| TC-17 | Point sampling | `test_pipeline.py::test_sample_points_balanced_and_deterministic` | 150 calm + 30 volatile days, then 50 + 2. | 5 + 5 points, the same on rerun. When volatile days run short, all 2 are taken and the rest filled with calm days. |
| TC-18 | Block bootstrap | `test_pipeline.py::test_block_bootstrap_shape` | n = 1000, block = 30. | 1000 in-range indices, first block contiguous. |
| TC-19 | Perturbation | `test_pipeline.py::test_perturb_keeps_feature_identities` | Large ε noise. | `macd_hist` still equals `macd − macd_signal`, and volatilities stay ≥ 0. |
| TC-20 | Reproducibility | `test_pipeline.py::test_two_quick_runs_identical` (`slow`) | Runs `run_all.py --quick` twice into separate folders (2 parallel workers). | Every cache and table CSV is exactly equal. All four models (incl. `xgb_shuffled`) and both quick coins present. |

Runtime guard (not a pytest test): `experiments.audit_model` raises `AssertionError` if
TreeSHAP gives a different answer when re-run on the same model and input during the real
experiment. The full run completed without it firing.
