# Problem statement

**Project:** Can you trust the explanation? Auditing the stability and agreement of
feature-importance methods for cryptocurrency prediction models (CS 180, SJSU, Fall 2026).

## The problem

Papers that forecast cryptocurrency prices with machine learning often add a post-hoc
explanation: a SHAP summary plot or a LIME bar chart that says, for example, "RSI and
lagged returns drive the prediction". These feature-importance claims usually appear as
facts about the market. They are only as reliable as the explanation method, and the
explanation method is only as reliable as the model it explains.

The general ML literature has shown that explanations can be fragile. LIME and KernelSHAP
are stochastic and depend on their sampling seed. Explanations can change under small input
perturbations (Alvarez-Melis & Jaakkola 2018; Ghorbani et al. 2019). They can be
manipulated (Slack et al. 2020). Different methods often disagree about the top features of
the same prediction, the "disagreement problem" (Krishna et al. 2022). Crypto markets make
all of this harder. Daily returns are close to unpredictable, standard technical features
are strongly correlated (the MACD line, signal and histogram are an exact linear
combination), and volatility moves between calm and turbulent regimes.

## Research questions

For next-day direction models (XGBoost, LightGBM, a small MLP) on five coins (BTC, ETH,
ADA, LINK, LTC):

* **RQ-a (agreement):** Do TreeSHAP, KernelSHAP, LIME and permutation importance agree on
  the top drivers of the same prediction?
* **RQ-b (retraining):** Do the top drivers stay the same when the model is retrained with a
  new random seed or on a block-bootstrap resample of the training data?
* **RQ-c (perturbation):** Do they stay the same when the inputs are nudged slightly
  (ε = 1% and 5% of each feature's training standard deviation)?
* **RQ-d (key question):** Is any of this instability *worse in high-volatility regimes*
  than in calm ones?

Two controls run through every question. First, the **explainer's own noise floor**: the
same model and the same input, explained again with a different seed. Instability at or
below this floor is measurement noise, not a property of the model. Second, a
**shuffled-label control model** trained on permuted labels, which by construction has no
skill. Its explanations show what "explaining noise" looks like.

## Why it matters

If the stated top driver of a crypto model changes with the random seed, a 5% input nudge,
or the market regime, then feature-importance claims built on it should not be read as
statements about the market. A quantified answer tells readers and reviewers how much
weight such claims can carry, and when.

## Scope

Public daily data (Coin Metrics Community, close and volume only), strictly time-ordered
walk-forward validation, a fixed seed everywhere, and all results reproducible from a
clean checkout with one command (`python run_all.py`).

---

# Literature survey outline

1. **Post-hoc attribution methods (what is being audited)**
   * Ribeiro, Singh & Guestrin (2016): LIME, local surrogate models. Introduces the method;
     it is *not* itself a stability study.
   * Lundberg & Lee (2017): SHAP, KernelSHAP, Shapley-value axioms. Lundberg et al. (2020)
     add TreeSHAP for tree ensembles.
   * Molnar (2022): a textbook overview of permutation importance, SHAP and LIME, including
     their caveats with correlated features.
2. **Known fragility of explanations**
   * Alvarez-Melis & Jaakkola (2018): robustness of interpretability methods. Local
     Lipschitz estimates show that LIME and SHAP explanations can change sharply under
     small input changes.
   * Ghorbani, Abid & Zou (2019): interpretations of neural networks are fragile under
     imperceptible perturbations.
   * Slack et al. (2020): adversarial models can fool LIME and SHAP.
   * Zhou, Hooker & Wang (2021): S-LIME. LIME's sampling noise makes explanations unstable;
     a hypothesis-testing fix.
3. **Disagreement between methods**
   * Krishna et al. (2022): the disagreement problem. Practitioners routinely see methods
     disagree on top-k features. The paper defines top-k feature, rank and sign agreement
     metrics (our Jaccard and Spearman metrics follow it).
4. **Crypto forecasting with ML (the application)**
   * Chen & Guestrin (2016): XGBoost.
   * Pinky & Akula (2024): ML for cryptocurrency forecasting. Representative of applied
     papers that report feature importances alongside accuracy.
5. **Methodological hygiene for financial time series**
   * Walk-forward validation and the leakage risk of shuffled k-fold on time series
     (Molnar 2022, ch. on evaluation). Block bootstrap for dependent data.
   * Multiple testing: Benjamini & Hochberg (1995), false discovery rate control.
6. **Gap this project fills.** No prior study we found measures agreement, retraining
   stability and perturbation stability *together*, against an explicit noise floor and a
   no-skill control, and *split by volatility regime*, for crypto direction models.

> Note on the proposal's reading list: [5] (S-LIME, a stability method) and [7] (the crypto
> forecasting paper) appear swapped between Weeks 4 and 5. S-LIME belongs with the
> explanation-stability weeks, and the forecasting paper belongs with model training.
> Krishna et al. (2022), the core reading for Week 6, was missing and is added above.
