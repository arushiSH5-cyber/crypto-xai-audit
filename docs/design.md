# System design

The pipeline is a set of small, mostly functional modules under `src/xaiaudit/`, plus
two classes that hold per-fold state. Every setting lives in `config.py`, and the
entry point is `run_all.py`.

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
| Expanding walk-forward, refit every 182 days, 1-day embargo | No shuffling, no future rows in training, which avoids the leakage common in this literature. |
| Regime = trailing `vol_30` above the *training-window* median | The label is known on the prediction day (no look-ahead in the regime definition). |
| Explained points sampled half calm / half volatile per fold | Both regimes are represented in every fold, so regime comparisons are within-fold. |
| Explainer seeds shared between noise-floor rep *r* and perturbation rep *r* | Common random numbers: the perturbation-minus-noise-floor gap isolates the input change. |
| Perturbation recomputes `macd_hist = macd − macd_signal` and clips volatilities at 0 | Perturbed inputs stay on the feature manifold. |
| Shuffled-label `xgb_shuffled` control | Shows how stable or unstable explanations of a skill-less model are. |
| Cluster (coin × fold) bootstrap and within-fold permutation | Points in a fold share a model and a market period, so they are not independent. |
| Placebo = regime sequence borrowed from a different fold (another half-year) | Realistic run lengths and proportions, but no link to the volatility on the actual dates. A circular shift was rejected because regimes persist for months, so the shifted labels stayed close to the real ones. |
| CRC-based seeds (`_seed(base_seed, coin, fold, tag)`) | Python's `hash()` is salted per process. CRC gives identical draws in every worker and on every run. |
