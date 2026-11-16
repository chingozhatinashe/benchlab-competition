# Bayes model: the linear model with the latent Bayes stream's regressors (an option, not submitted)

**The decision (user, 2026-11-16).** The [Bayes features](bayes_features.md) study found one set that helps: from the latent-arrival [Bayes stream](bayes_stream.md) fit, P(arrival by lead) and the mean path as two extra regressors of the submitted linear model. On the six folds it was 0.41% better than the submitted model and won 4 of 6, short of the 1% rule. The user made an exception: build it, but don't make it the submitted model.

**Status (2026-11-16): built as an option.**
- **Selected with** `SWP_MODEL=bayes` (`inference.run_bayes_forecast`). It falls back to the linear model, then the usual chain. The default is still the linear model.
- **Code:** `models.bayes_model` (runtime, numpy only); `training.bayes_model train|evaluate` (dev only: jax, numpyro).
- **Asset:** `src/assets/bayes_model.json` (144 kB): 32 posterior draws, the stream model's input statistics and the refitted linear model.
- **Tests:** `tests/test_bayes_model.py`; the features are tested in `tests/test_bayes_features.py`.

## The model

Per run, from the latent fit's 32 posterior draws (numpy, the same code as the study's `training.bayes_features`):

| regressor | per lead k |
|---|---|
| `structural_bayes_parr_lead_0h` | P(N ≥ 1) × P(first arrival ≤ k): when a stream arrives, without its size |
| `structural_bayes_mean_lead_0h` | the additive model's mean path: the slow wind a + ρ^k (v0 − a) plus the expected stream excess |

Both are lead-aligned (lead k reads the column for its 6-hour block), like the other structural regressors. They are computed from the speed history and the submitted linear model's own regressors, so the option fetches nothing more and takes milliseconds.

The linear part is the submitted config (`training/linear_model_config.json` with `with_fixes`) with the two regressors added, refitted on every training run (2016-10-20 → 2025-10-20, 13,148 runs). The coronal-hole submodel and the PFSS imputation work as in the submitted model.

## The final fit (2026-11-16)

`training.bayes_model train`: the latent Bayes stream fit (additive rule, one arrival per window at most, 300 + 200 draws × 2 chains, the study's NUTS settings) on 859 training windows of 84 h (CME windows dropped, as in the study). It took 102 min. The user ran it in a terminal, since the laptop's low memory stopped background runs.

**The fit did not converge: R-hat 4.45, ESS 1, no divergences.** All six fold fits had converged (R-hat at most 1.015). With all the data, the two chains settle in different modes of the stream shape:

| | chain 1 | chain 2 |
|---|---|---|
| rise width exp(μ_R) | 0.69 h | 0.96 h |
| decay inputs `w_D` (largest gaps) | 1.75, −0.88, 2.07 | 0.70, 0.23, 0.98 |
| expected arrivals per 84 h, Λ base | 0.39 | 0.47 |
| ρ_up, ρ_dn, σ_η | 0.979, 0.977, 11.2 | 0.979, 0.976, 11.3 |

**The regressors barely depend on the mode.** On 2,000 training runs, the two chains' features correlate at 0.98 (`parr`) and 0.997 (`mean`). The mean absolute difference is 0.03 in probability and 3.2 km/s, against standard deviations of 0.17 and 63 km/s. The 32 kept draws mix both modes; the linear model is refitted on the features they give.

**If this needs fixing:** refit with more warmup or a different seed (`--refit`, ~1.7 h), or keep one chain. It is not needed for the option to work.

## Eval year (descriptive, once)

`training.bayes_model evaluate`: 1,448 runs, 2025-10-20 → 2026-10-16, through the live imputation, official metric. Results: `src/experiments/bayes_model_eval_year.csv`.

| model | S_final | vs reference | CME | stream arrival | fast stream | intermediate | slow wind |
|---|---|---|---|---|---|---|---|
| bayes model | 117.55 | **−0.37%** | −1.1% | +1.4% | +3.2% | −0.2% | −0.9% |
| reference (same table) | 117.98 | — | — | — | — | — | — |
| shipped linear model | 117.98 | 0.00% | | | | | |
| persistence | 157.28 | +33% | | | | | |

- **The gain holds on the held-out year:** −0.37%, in line with −0.41% on the folds.
- **But it comes from different places:** better in CME and slow-wind runs, worse at stream arrivals (+1.4%) and in fast streams (+3.2%). On the folds the gain was mostly in fast streams (−2.5%).
- **The setup checks out:** the reference refitted on the same table equals the shipped linear model exactly.

The eval year doesn't change the choice: the model was chosen on the folds, and it stays an option.
