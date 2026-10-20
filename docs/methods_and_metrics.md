# Methods and metrics

Every definition below matches the code. The file and function are given in brackets, so you
can point to them when defending the project. Notation: $P_t$ is the close price on day $t$,
$V_t$ the reported spot volume, and $r_t = \log P_t - \log P_{t-1}$ the daily log return.

---

## 1. Data

Coin Metrics Community Data, columns `PriceUSD` (reference rate at 00:00 UTC) and
`volume_reported_spot_usd_1d`, 2017-12-01 to 2026-05-22, for BTC, ETH, ADA, LINK and LTC
[`data.load_coin`]. Rows with missing or non-positive price or volume are dropped. If that
leaves any missing calendar day, the loader **raises an error** instead of filling the gap.

## 2. Features (all known at the end of day *t*) [`features.build_features`]

| Feature | Plain English | Formula |
|---|---|---|
| `ret_lag1` | today's return | $r_t$ |
| `ret_lag2` | yesterday's return | $r_{t-1}$ |
| `ret_lag3` | return two days ago | $r_{t-2}$ |
| `ret_5d` | return over the last 5 days | $\sum_{i=0}^{4} r_{t-i}$ |
| `ret_10d` | return over the last 10 days | $\sum_{i=0}^{9} r_{t-i}$ |
| `rsi_14` | Wilder's RSI, scaled to [0, 1]: are recent moves mostly up? | $1 - \frac{1}{1+RS_t}$, with $RS_t = \frac{\text{EWM}_{1/14}(\max(\Delta P,0))}{\text{EWM}_{1/14}(\max(-\Delta P,0))}$; set to 1 if there were no losses |
| `macd` | short-term vs long-term trend, as a fraction of price | $(\text{EMA}_{12}(P)_t - \text{EMA}_{26}(P)_t) / P_t$ |
| `macd_signal` | smoothed MACD | $\text{EMA}_9(\text{macd})_t$ |
| `macd_hist` | MACD minus its signal | $\text{macd}_t - \text{macd\_signal}_t$ (**exactly collinear** with the two above) |
| `vol_7` | 7-day realised volatility | sample std of $r_{t-6..t}$ |
| `vol_30` | 30-day realised volatility (also the regime variable) | sample std of $r_{t-29..t}$ |
| `volume_ratio` | is today's volume unusually high? | $\log\big(V_t / \text{mean}(V_{t-29..t})\big)$ |

$\text{EMA}_s$ uses smoothing factor $\alpha = 2/(s+1)$ (pandas `ewm(span=s, adjust=False)`).
The first 60 rows are dropped so the EMAs have warmed up, leaving 3034 rows per coin.

**Target.** $y_t = \mathbb{1}[r_{t+1} > 0]$: does the price go **up tomorrow**? `next_ret` is
the only column that looks forward, and it is never used as a feature.

## 3. Walk-forward validation [`validation.walk_forward_splits`]

Fold $i$ tests on the 182-day block $[s_i, s_i + 182)$, with $s_0 =$ 2021-01-01 and
$s_{i+1} = s_i + 182$. It trains on **every row dated before $s_i - 1$ day** (a 1-day embargo),
which gives an expanding window. A fold is kept only if it has at least 730 training rows. The
result is 11 folds per coin and 55 in total. Training sets grow from 1066 to 2886 rows, and
test blocks hold 147–182 days (the last block is shorter because the data end on 2026-05-21).
Nothing is shuffled, and no test row is ever used for training.

## 4. Volatility regime [`validation.regime_labels`]

For fold $f$ the threshold is $\tau_f = \text{median}\{\text{vol\_30}_s : s \in \text{train}_f\}$.
Test day $t$ is **volatile** if $\text{vol\_30}_t > \tau_f$, otherwise **calm**. Both
$\text{vol\_30}_t$ and $\tau_f$ are known on day $t$, so the label never peeks at the future
(tests TC-06 and TC-07). Overall, 0.293 of test days are volatile, and 7 of the 55 folds have
no volatile test days at all.

## 5. Models [`models.make_model`]

| Name | What it is | Settings | What `seed` changes |
|---|---|---|---|
| `xgb` | gradient-boosted trees (XGBoost) | 200 trees, depth 3, learning rate 0.05, subsample 0.8, colsample 0.8, min_child_weight 5 | which rows/columns each tree sees |
| `lgbm` | gradient-boosted trees (LightGBM) | 200 trees, 8 leaves, learning rate 0.05, subsample 0.8, colsample 0.8 | which rows/columns each tree sees |
| `mlp` | small neural net (scikit-learn) | standardise → 32 → 16 ReLU units, L2 1e-3, early stopping on 15% of training rows | initial weights, batch order, validation split |
| `xgb_shuffled` | **control**: `xgb` trained on randomly permuted training labels | same as `xgb` | same as `xgb` |

The control cannot learn anything real about tomorrow's direction, so its explanations
show what "explaining noise" looks like. Predicted class: up if $p > 0.5$.

## 6. Explainers [`explainers.py`]

| Explainer | Plain English | Output space | Random? |
|---|---|---|---|
| **TreeSHAP** | Exact Shapley values for tree models, computed from the tree structure (path-dependent) | log-odds | No. Bit-for-bit deterministic (TC-15, plus a runtime assert) |
| **KernelSHAP** | Estimates Shapley values by switching groups of features "off" (replacing them with a 20-centre k-means summary of the training data) and fitting a weighted regression on 200 such samples. shap's default keeps 10 of 12 features non-zero (`l1_reg = num_features(10)`) | log-odds (logit link) | Yes, through the sampled coalitions |
| **LIME** | Samples 1000 nearby points (features binned into training quartiles), weights them by closeness, and fits a linear model. The coefficients are the explanation | probability | Yes, through the sampled points |
| **Permutation importance** | Shuffle one feature in the test fold and measure how much ROC-AUC drops (10 repeats). Global only | ΔAUC | Yes (seeded) |

Because the output spaces differ, we compare **rankings of absolute attributions**, never raw values.

## 7. Agreement and stability metrics [`metrics.py`]

Let $a, b \in \mathbb{R}^{12}$ be two attribution vectors for the same prediction, and let
$T_k(a)$ be the indices of the $k$ largest $|a_j|$, with ties broken by the lower index. We use $k = 3$.

| Metric | Plain English | Formula |
|---|---|---|
| **Top-k Jaccard** | How much do the two top-3 lists overlap? 1 = same three features, 0 = none shared | $J_k(a,b) = \frac{\lvert T_k(a) \cap T_k(b)\rvert}{\lvert T_k(a) \cup T_k(b)\rvert}$ |
| **Spearman** | Do the two explanations rank *all 12* features in the same order? | Spearman rank correlation of $\lvert a\rvert$ and $\lvert b\rvert$ (undefined if either is constant) |
| **Top-1 agreement** | Do they name the same single most important feature? | $\mathbb{1}[T_1(a) = T_1(b)]$ |
| **Set stability** (retraining, noise floor) | For $m$ explanations of one point, the average of the above over all $m(m-1)/2$ pairs. Also the number of distinct top-1 features, and the share held by the most common one | $\frac{2}{m(m-1)}\sum_{i<j} J_k(a_i,a_j)$, etc. |
| **Top-1 change rate** (perturbation, noise floor) | How often does the top-1 feature differ from the unperturbed (reference) explanation? | $1 - \frac{1}{R}\sum_{r} \mathbb{1}[T_1(a_r) = T_1(a_{\text{ref}})]$ |
| **Prediction flip rate** | How often does the *predicted class* change? | $\frac{1}{R}\sum_r \mathbb{1}[(p_r > 0.5) \ne (p_0 > 0.5)]$; we also report the mean $\lvert p_r - p_0\rvert$ |
| **Class flip rate** (retraining) | The same, across refitted models instead of perturbed inputs | as above, $p_r$ from refit $r$ |
| **Excess over noise floor** | Instability beyond what the explainer's own randomness causes | (perturbation top-1 change rate) − (noise-floor top-1 change rate), paired per point |
| **Instability** (used in regime tests) | Every explanation metric oriented so that "higher = less stable" | $1 - J_3$, or the top-1 change rate |

**Random baseline.** Two independent random rankings of $p = 12$ features have expected top-3
Jaccard $\sum_{i=0}^{k} \frac{\binom{k}{i}\binom{p-k}{k-i}}{\binom{p}{k}} \cdot \frac{i}{2k-i} = 0.164$
and top-1 agreement $1/12 = 0.083$ [`analysis.random_jaccard`].

## 8. The four audits [`experiments.audit_model`]

In each (coin, fold), 10 test days are sampled: 5 calm and 5 volatile when possible, with any
shortfall filled from the other regime. That gives **550 explained points** (310 calm,
240 volatile). Every audit uses these same points.

1. **Agreement.** Per point: Jaccard, Spearman and top-1 agreement for every pair of explainers
   (TreeSHAP~KernelSHAP, TreeSHAP~LIME, KernelSHAP~LIME; only the last for the MLP). Per fold:
   the same metrics between permutation importance and mean |attribution|.
2. **Noise floor.** Same model, same input, the explainer re-run 3 times with new seeds
   ($\text{seed}_r = \text{base} + 100003\,(r+1)$). This measures how much the explainer
   disagrees with itself. For TreeSHAP the noise floor is exactly zero, and that is asserted.
3. **Retraining.** Refit the model (a) with seeds 1–5, and (b) at seed 0 on 5 **moving-block
   bootstraps**: training rows are resampled in contiguous 30-day blocks, which keeps the
   short-range time dependence. Each refit is explained with the *primary* method (TreeSHAP for
   trees, KernelSHAP for the MLP). Set stability is computed over the base model plus 5 refits
   (6 explanations per point).
4. **Perturbation.** $x' = x + \varepsilon\,\sigma_{\text{train}} \odot z$ with
   $z \sim N(0, I)$ and $\varepsilon \in \{0.01, 0.05\}$. Then `macd_hist` is recomputed as
   `macd − macd_signal` and the volatilities are clipped at ≥ 0, so $x'$ is still a valid
   feature vector. There are 10 repeats for TreeSHAP and 3 for KernelSHAP/LIME. Repeat $r$ uses
   the same explainer seed as noise-floor repeat $r$ ("common random numbers"), so the
   perturbation-minus-noise-floor gap is caused only by the input change.

## 9. Statistics [`analysis.py`]

The 10 points in a (coin, fold) share one model and one market period, so they are **not
independent**. The 55 (coin, fold) groups are therefore treated as **clusters**.

| Procedure | Plain English | Details |
|---|---|---|
| **Cluster bootstrap CI** | Resample whole folds with replacement 2000 times. The middle 95% of the recomputed means is the CI | Pooled mean $= \sum_c \text{sum}_c / \sum_c n_c$ over resampled clusters; percentile interval [2.5%, 97.5%] |
| **Volatile − calm difference** | $\bar{x}_{\text{vol}} - \bar{x}_{\text{calm}}$ for one audit cell | CI from the same cluster bootstrap |
| **Within-cluster permutation test** | If regime didn't matter, shuffling the calm/volatile labels *inside each fold* shouldn't change the difference much | 2000 shuffles; $p = \frac{1 + \#\{\lvert\Delta^*\rvert \ge \lvert\Delta\rvert\}}{1 + 2000}$ (two-sided) |
| **Benjamini–Hochberg** | With 56 tests, some small p-values appear by luck. BH controls the expected share of false discoveries | Sort $p_{(1)} \le \dots \le p_{(m)}$; $q_{(i)} = \min_{j \ge i} \min(1, m\,p_{(j)}/j)$; "significant" if $q < 0.05$ |
| **Placebo test** | Give each fold the calm/volatile sequence of a *different, randomly chosen* fold (another half-year), 2000 times. These labels look realistic but have nothing to do with the actual volatility on those days | Reported as a placebo p-value and the placebo 95% range. A real regime effect should lie outside that range |
| **Accuracy** | Share of test days where the predicted class equals the true direction | Cluster-bootstrap CI; exact binomial test vs 0.5; "always up" baseline $=$ share of up days |
| **ROC-AUC** | Probability that a random up-day gets a higher $p$ than a random down-day (0.5 = no skill) | Cluster-bootstrap CI |

The family of 56 tests: 4 accuracy, 10 agreement, 8 noise floor, 8 retraining,
22 perturbation, and 4 "headline" tests that pool the three real models.

## 10. Reproducibility

Every random draw is seeded from $\text{CRC32}(\text{base\_seed}, \text{coin}, \text{fold}, \text{task})$
[`experiments._seed`]. CRC is used instead of Python's `hash()`, which changes between processes.
Each worker runs single-threaded. Two complete runs give identical CSVs (TC-20).
