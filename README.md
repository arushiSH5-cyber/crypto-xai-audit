# Can you trust the explanation? A crypto XAI audit

CS 180 research project (SJSU, Fall 2026). This project audits whether SHAP and LIME
feature-importance explanations of **next-day direction models** for five cryptocurrencies
(BTC, ETH, ADA, LINK, LTC) can be trusted. It asks four questions:

* **(a) Agreement:** do TreeSHAP, KernelSHAP, LIME and permutation importance name the same top drivers?
* **(b) Retraining stability:** do the drivers survive a new random seed or a block-bootstrap resample of the training data?
* **(c) Perturbation stability:** do they survive a 1% / 5%-of-a-standard-deviation input nudge?
* **(d) Regimes (the key question):** is the instability worse in high-volatility regimes than in calm ones?

Every instability number is compared with the **explainer's own noise floor** (the same model and
input, re-explained with a new seed) and with a **shuffled-label control model**
(`xgb_shuffled`, which has no skill by construction). Model accuracy is reported next to every
stability result.

**Report:** [`report/report.pdf`](report/report.pdf) (source [`report/report.md`](report/report.md)).

## Main findings

The numbers below come from `results/tables/` after the full run.

* The models have **almost no skill**. Pooled accuracy is 0.512 (XGBoost), 0.512 (LightGBM) and
  0.507 (MLP), against 0.502 for the shuffled-label control and 0.504 for "always up". All of the
  small edge is in volatile regimes: XGBoost reaches 0.538 there versus 0.502 when calm.
* **The methods disagree.** TreeSHAP and KernelSHAP share about two-thirds of their top-3 features.
  LIME shares about one-third with either of them. LIME also disagrees with *itself* across seeds
  (top-3 Jaccard about 0.64).
* **Explanations move a lot under retraining.** The TreeSHAP top-3 Jaccard is about 0.6 across
  seeds and about 0.3 across block bootstraps. TreeSHAP's noise floor is exactly zero, so this is
  real model sensitivity, not explainer noise. The skill-less control is about as stable as the
  real models, so a stable explanation is no evidence that the model learned anything.
* **No evidence that instability is worse in volatile regimes.** No explanation-instability test
  is significantly *higher* in volatile periods after Benjamini–Hochberg correction. The two
  significant explanation differences (both MLP) go the other way: explanations were *more*
  stable when volatile. The only robust regime effect is accuracy.

## Reproduce from a clean checkout

```bash
git clone <this repo> && cd crypto-xai-audit
python3.12 -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12 recommended
pip install -r requirements.txt                              # lime fails on 3.13+: see HANDOFF.md
python -m pytest -q                                          # 25 tests, about 40 s
python run_all.py --quick                                    # smoke test, about 20 s, writes results/quick/
python run_all.py                                            # full run, about 5 min on 2 cores, writes results/
```

`python run_all.py --analysis-only` rebuilds tables and figures from `results/cache/` without
re-running the experiments. Every setting (coins, dates, seeds, sample sizes, repetitions) is in
[`src/xaiaudit/config.py`](src/xaiaudit/config.py). Everything is seeded (`base_seed = 0` plus
CRC-derived per-task seeds), and two runs give bit-identical CSVs (test TC-20). On Colab, open
[`notebooks/run_pipeline.ipynb`](notebooks/run_pipeline.ipynb), set the repo URL and choose *Run all*.

Tested with Python 3.12.15, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1,
lightgbm 4.7.0, shap 0.54.1, lime 0.2.0.1. Exact versions for each run are written to
`results/run_log.json`.

## Repository layout

```
crypto-xai-audit/
├── run_all.py                 entry point: experiments -> analysis (--quick, --analysis-only)
├── requirements.txt
├── HANDOFF.md                 original design notes
├── data/raw/                  Coin Metrics Community daily CSVs (btc, eth, ada, link, ltc)
├── src/xaiaudit/
│   ├── config.py              every setting + QUICK overrides
│   ├── data.py                loader (refuses gaps/duplicates)
│   ├── features.py            12 causal features + next-day target
│   ├── validation.py          expanding walk-forward folds, train-median volatility regime
│   ├── models.py              xgb, lgbm, mlp
│   ├── explainers.py          TreeSHAP, KernelSHAP, LIME, permutation importance
│   ├── metrics.py             top-k Jaccard, Spearman, top-1 agreement, set stability
│   ├── experiments.py         the four audits + shuffled-label control, parallel over coins
│   └── analysis.py            cluster-bootstrap CIs, permutation/placebo tests, BH-FDR, figures
├── tests/                     pytest suite (see docs/test_cases.md)
├── results/
│   ├── cache/                 tidy per-point CSVs + attributions.pkl (raw attribution arrays)
│   ├── tables/                summary tables (.csv + .md) and key_numbers.json
│   ├── figures/               fig1 … fig7 (PNG)
│   └── run_log.json           timings, package versions, the exact config used
├── notebooks/run_pipeline.ipynb   Colab notebook
├── docs/
│   ├── 01_problem_statement.md    problem statement + literature-survey outline
│   ├── design.md                  UML class + sequence diagrams (Mermaid)
│   ├── test_cases.md              test-case table with expected results
│   ├── presentation_script.md     10-minute code-and-results walkthrough
│   └── CS180-Proposal_Crypto_Explainability_Audit.pdf
└── report/
    ├── report.md, report.pdf      workshop-style paper
    └── figures -> ../results/figures (copied at build time)
```

## Data

Coin Metrics Community Data (`github.com/coinmetrics/data`, `csv/<asset>.csv`), columns
`PriceUSD` and `volume_reported_spot_usd_1d`, from 2017-12-01 to 2026-05-22. The data are
close-only: there is no open/high/low, so there are no range-based features. SOL was replaced by
LTC because SOL's volume history is missing in this dataset.
