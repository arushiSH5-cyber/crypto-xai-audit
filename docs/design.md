# System design

The pipeline is a set of small, mostly functional modules under `src/xaiaudit/`, plus
four small classes that hold state (`Fold`, `FoldContext`, `LimeWrapper`, `PlaceboLabeller`). Every setting lives in `config.py`, and the entry point is
`run_all.py`. Formulas for everything below are in `docs/methods_and_metrics.md`.

## Modules

| Module | What it does |
|---|---|
| `config.py` | One dictionary with every setting (coins, dates, fold length, seeds, sample sizes, repetitions), plus `QUICK` overrides for the smoke test. `get_config(quick, overrides)` returns a copy. |
| `data.py` | `load_coin` reads one Coin Metrics CSV, keeps price and volume, and **raises an error** on duplicate or missing days instead of filling them. |
| `features.py` | `build_features` makes the 12 causal features and the next-day target. `FEATURE_GROUPS` lists related features. |
| `validation.py` | `walk_forward_splits` builds the expanding-window folds (`Fold` dataclass). `regime_labels` labels test days calm/volatile against the training median of `vol_30`. |
| `models.py` | `make_model` / `fit_model` / `predict_p` for XGBoost, LightGBM and the MLP. The seed controls all model randomness. |
| `explainers.py` | Thin wrappers that return an `(n_points, 12)` array: `tree_shap`, `kernel_shap` (with a k-means background), `LimeWrapper.explain`, and `perm_importance` (global). |
| `metrics.py` | Top-k sets, Jaccard, Spearman on \|attr\|, top-1 agreement, `set_stability` (many explanations of one point) and `against_reference` (perturbed vs original). |
| `experiments.py` | `FoldContext` holds one fold's data, regime labels, explained points, KernelSHAP background and LIME explainer. `audit_model` runs the four audits for one model. `run_coin` loops over folds and models. `run_all_coins` runs coins in parallel (joblib) and writes `results/cache/`. |
| `analysis.py` | Reads the cache. Computes cluster-bootstrap CIs, the 56 volatile-vs-calm tests (within-fold permutation, placebo, BH), accuracy/AUC per regime, and writes `results/tables/` and `results/figures/`. |
| `run_all.py` | Command-line entry point (`--quick`, `--analysis-only`, `--results-dir`). Times each stage and writes `run_log.json`. |

## Data-flow diagram

```mermaid
flowchart LR
    A[(data/raw/*.csv<br/>Coin Metrics)] -->|load_coin| B[price, volume<br/>gap-checked]
    B -->|build_features| C[12 causal features<br/>+ next-day target]
    C -->|walk_forward_splits| D[55 folds<br/>train / test]
    D -->|regime_labels<br/>train median of vol_30| E[calm / volatile<br/>label per test day]
    D --> F[fit xgb, lgbm, mlp,<br/>xgb_shuffled]
    F --> G[predictions<br/>every test day]
    E --> H[10 explained points<br/>per fold, 5 + 5]
    F --> I{four audits}
    H --> I
    I --> I1[agreement]
    I --> I2[noise floor]
    I --> I3[retraining:<br/>seeds + block bootstrap]
    I --> I4[perturbation:<br/>eps = 0.01, 0.05]
    G --> J[(results/cache/*.csv<br/>attributions.pkl)]
    I1 --> J
    I2 --> J
    I3 --> J
    I4 --> J
    J -->|run_analysis| K[cluster bootstrap CIs<br/>permutation + placebo tests<br/>BH correction]
    K --> L[(results/tables/*.csv, *.md<br/>key_numbers.json)]
    K --> M[(results/figures/fig1-fig7)]
    L --> N[report/report.md -> report.pdf]
    M --> N
```

## Module / class diagram (UML)

```mermaid
classDiagram
    direction LR
    class config {
        +CONFIG: dict
        +QUICK: dict
        +get_config(quick, overrides) dict
    }
    class data {
        +load_coin(coin, raw_dir, start, end) DataFrame
    }
    class features {
        +FEATURES: list~str~
        +FEATURE_GROUPS: dict
        +build_features(df) DataFrame
    }
    class validation {
        +walk_forward_splits(dates, test_start, refit_every_days, min_train_days, embargo_days) list~Fold~
        +regime_labels(train_vol, test_vol) ndarray
    }
    class Fold {
        <<dataclass>>
        +fold_id: int
        +train_idx: ndarray
        +test_idx: ndarray
        +train_end: Timestamp
        +test_start: Timestamp
        +test_end: Timestamp
    }
    class models {
        +TREE_MODELS: set
        +make_model(name, seed)
        +fit_model(name, X, y, seed)
        +predict_p(model, X) ndarray
    }
    class explainers {
        +tree_shap(model, X) ndarray
        +make_background(X_train, k, seed) ndarray
        +kernel_shap(model, X, background, nsamples, seed) ndarray
        +perm_importance(model, X, y, repeats, seed) ndarray
    }
    class LimeWrapper {
        +X_train: ndarray
        +names: list~str~
        +explain(model, X, num_samples, seed) ndarray
    }
    class metrics {
        +top_k(a, k) set
        +jaccard_topk(a, b, k) float
        +rank_corr(a, b) float
        +pair_metrics(a, b, k) dict
        +set_stability(vectors, k) dict
        +against_reference(ref, others, k) dict
    }
    class FoldContext {
        +coin: str
        +fold: Fold
        +Xtr, ytr, Xte, yte: ndarray
        +threshold: float
        +regime_te: ndarray
        +pts: ndarray
        +background: ndarray
        +lime: LimeWrapper
        +y_shuffled: ndarray
        +fit(model, seed, rows)
        +explain(method, model, X, seed) ndarray
        +point_keys() list~dict~
    }
    class experiments {
        +CONTROL = "xgb_shuffled"
        +run_coin(coin, cfg) (dict, dict)
        +audit_model(ctx, model, out, attrs)
        +run_all_coins(cfg) dict~DataFrame~
        +sample_points() / block_bootstrap() / perturb()
    }
    class PlaceboLabeller {
        +offsets: ndarray
        +labels(df) list~ndarray~
    }
    class analysis {
        +bh(p) ndarray
        +cluster_boot_mean(values, clusters, reps, rng)
        +regime_test(values, is_vol, clusters, reps, rng, placebo)
        +build_tests(T, cfg, placebo, rng)
        +run_analysis(cfg) dict
        +fig_*()
    }
    class run_all {
        +main(argv)
    }

    validation ..> Fold : creates
    FoldContext o-- Fold
    FoldContext o-- LimeWrapper
    FoldContext ..> models : fit
    FoldContext ..> explainers : explain
    experiments ..> FoldContext : one per (coin, fold)
    experiments ..> data
    experiments ..> features
    experiments ..> validation
    experiments ..> metrics
    analysis ..> PlaceboLabeller
    run_all ..> config
    run_all ..> experiments
    run_all ..> analysis
```

## Sequence diagram: one full run

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant R as run_all.py
    participant X as experiments
    participant W as joblib worker (per coin)
    participant C as FoldContext
    participant M as models / explainers
    participant FS as results/
    participant A as analysis

    U->>R: python run_all.py [--quick]
    R->>R: get_config(quick)
    R->>X: run_all_coins(cfg)
    X->>W: Parallel(n_jobs=2): run_coin(coin) for each coin
    W->>W: load_coin -> build_features -> walk_forward_splits
    loop each fold
        W->>C: FoldContext(coin, fold) (regime threshold from TRAIN vol_30 median, sample points, k-means background)
        loop each model in xgb, lgbm, mlp, xgb_shuffled
            C->>M: fit(model, base_seed)
            M-->>C: test predictions
            C->>M: TreeSHAP / KernelSHAP / LIME on explained points
            Note over C,M: 1 agreement (per point + permutation importance globally)
            C->>M: re-explain with new explainer seeds
            Note over C,M: 2 noise floor (TreeSHAP must be bit-identical)
            loop retrain seeds + block bootstraps
                C->>M: refit, predict, explain with primary method
            end
            Note over C,M: 3 retraining stability
            loop eps in perturb_eps
                C->>M: explain eps x train-std noisy inputs (shared seeds with noise floor)
            end
            Note over C,M: 4 perturbation stability
        end
    end
    W-->>X: tidy rows + raw attributions
    X->>FS: cache/*.csv, cache/attributions.pkl
    R->>A: run_analysis(cfg)
    A->>FS: read cache/*.csv
    A->>A: cluster-bootstrap CIs, within-fold permutation p, placebo (donor-fold labels), BH-FDR
    A->>FS: tables/*.csv|md, tables/key_numbers.json, figures/fig1..fig7.png
    R->>FS: run_log.json (timings, versions, config)
```

## Key design decisions

| Decision | Why |
|---|---|
| **Expanding walk-forward validation**: refit every 182 days, test from 2021-01-01, 1-day embargo, no shuffling | Shuffled k-fold on time series lets the model train on days *after* the ones it is tested on, which inflates accuracy. That leakage is common in this literature. Walk-forward only ever trains on the past, like a real forecaster, and refitting every 6 months lets the model use new data. |
| **Causal regime threshold**: a test day is volatile if its `vol_30` exceeds the median `vol_30` of the fold's *training* window | A threshold computed on the whole sample (or the test period) would use future volatility to label past days. The training median is known on the prediction day, so the regime is something a forecaster could actually observe. Tested by TC-06 and TC-07. |
| **Shuffled-label control** (`xgb_shuffled`) | Accuracy near 0.5 raises the question "is there anything to explain?". A model trained on permuted labels has no skill by construction. If its explanations are as agreeable and as stable as the real models', then agreement and stability can't be read as evidence of real signal. |
| **Explainer noise floor**: KernelSHAP and LIME re-run with new seeds on the same model and input; TreeSHAP asserted bit-identical | Stochastic explainers disagree with themselves. Without a noise floor, explainer randomness would be counted as model instability. Every retraining and perturbation number is compared with this floor. |
| **Moving-block bootstrap** (30-day blocks) for retraining | Daily returns and especially the features (rolling windows, EMAs) are autocorrelated. An ordinary bootstrap would break that structure. Resampling contiguous 30-day blocks keeps short-range dependence, so each resample looks like a plausible alternative history (Künsch 1989). |
| Explained points sampled half calm / half volatile per fold | Both regimes appear in every fold where possible, so regime comparisons are made within a fold, not across different years. |
| Explainer seeds shared between noise-floor rep *r* and perturbation rep *r* | Common random numbers: the perturbation-minus-noise-floor gap isolates the input change. |
| Perturbation recomputes `macd_hist = macd − macd_signal` and clips volatilities at 0 | Perturbed inputs stay valid feature vectors (on the feature manifold). |
| Cluster (coin × fold) bootstrap and within-fold permutation | Points in a fold share a model and a market period, so they are not independent. |
| Placebo = regime sequence borrowed from a different fold (another half-year) | Realistic run lengths and proportions, but no link to the volatility on the actual dates. A circular shift was rejected because regimes persist for months, so shifted labels stayed close to the real ones. |
| CRC-based seeds (`_seed(base_seed, coin, fold, tag)`) | Python's `hash()` is salted per process. CRC gives identical draws in every worker and on every run. |
