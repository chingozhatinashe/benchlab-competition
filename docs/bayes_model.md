# Bayes model: the linear model with the latent Bayes stream's regressors (an option, not submitted)

**The decision (user, 2026-11-16).** The [Bayes features](bayes_features.md) study found one set that helps: from the latent-arrival [Bayes stream](bayes_stream.md) fit, P(arrival by lead) and the mean path as two extra regressors of the submitted linear model. On the six folds it was 0.41% better than the submitted model and won 4 of 6, short of the 1% rule. The user made an exception: build it, but don't make it the submitted model.

**Status (2026-11-16): built as an option.**
- **Selected with** `SWP_MODEL=bayes` (`inference.run_bayes_forecast`). It falls back to the linear model, then the usual chain. The default is still the linear model.
- **Code:** `models.bayes_model` (runtime, numpy only); `training.bayes_model train|evaluate` (dev only: jax, numpyro).
- **Asset:** `src/assets/bayes_model.json` (143 kB, the constrained fit): 32 posterior draws, the stream model's input statistics and the refitted linear model.
- **Tests:** `tests/test_bayes_model.py`; the features are tested in `tests/test_bayes_features.py`.

## The model

Per run, from the latent fit's 32 posterior draws (numpy, the same code as the study's `training.bayes_features`):

| regressor | per lead k |
|---|---|
| `structural_bayes_parr_lead_0h` | P(N ≥ 1) × P(first arrival ≤ k): when a stream arrives, without its size |
| `structural_bayes_mean_lead_0h` | the additive model's mean path: the slow wind a + ρ^k (v0 − a) plus the expected stream excess |

Both are lead-aligned (lead k reads the column for its 6-hour block), like the other structural regressors. They are computed from the speed history and the submitted linear model's own regressors, so the option fetches nothing more and takes milliseconds.

The linear part is the submitted config (`training/linear_model_config.json` with `with_fixes`) with the two regressors added, refitted on every training run (2016-10-20 → 2025-10-20, 13,148 runs). The coronal-hole submodel and the PFSS imputation work as in the submitted model.

## The final fit (2026-11-17): constrained, converged

`training.bayes_model train` (default `--fit latent_constrained`): the latent Bayes stream fit on 859 training windows of 84 h (CME windows dropped, as in the study). It uses the additive rule, at most one arrival per window, 300 + 200 draws × 2 chains and the study's NUTS settings. Two changes to the plain latent fit:
- **Rise width fixed at 1 h** (`Config.rise_h`), instead of fitting μ_R.
- **A tighter prior on the decay inputs:** `w_D` sd 0.15 instead of 0.3 (`Config.w_D_sd`).

**It converges:** R-hat 1.028, ESS 214, no divergences, in 45 min. Both chains agree: μ_D 4.364 / 4.355, `w_D` −0.30, −0.03, 0.34 / −0.32, −0.02, 0.35. The user ran it in a terminal, since the laptop's low memory stops background runs.

### Why the constraint: the plain latent fit had two modes (2026-11-16)

On every training run, the plain latent fit (`--fit latent`) did not converge: R-hat 4.45, ESS 1, no divergences. All six fold fits had converged (R-hat at most 1.015). With all the data, the two chains settle in different modes of the stream shape:

| | chain 1 | chain 2 |
|---|---|---|
| rise width exp(μ_R) | 0.69 h | 0.96 h |
| decay inputs `w_D` (largest gaps) | 1.75, −0.88, 2.07 | 0.70, 0.23, 0.98 |
| expected arrivals per 84 h, Λ base | 0.39 | 0.47 |
| ρ_up, ρ_dn, σ_η | 0.979, 0.977, 11.2 | 0.979, 0.976, 11.3 |

- **More warmup doesn't merge them.** A refit with 1000 warmup steps (`--warmup 1000`, 134 min) reproduced the same two modes, with R-hat 4.68.
- **They are separate optima.** MAP runs started from each chain's mean stay in their own mode (loss 215,724.5 and 215,763.8). The default MAP start reaches the worse one, and one chain finds the better one while sampling.
- **The rise width is below the data's resolution.** Fitted rise widths of 0.7 h and 1 h are below what hourly data, with arrivals in 3 h bins, can resolve. It trades off against the decay inputs, which run up to 7 prior sds from zero.
- **The constrained model has a single optimum here.** MAP runs from both old modes and from the default start reach the same point (loss 215,772.7). The loss is higher than either mode's, partly through the tighter prior's penalty.

**The regressors barely depend on the mode.** On 2,000 training runs, the two chains' features correlate at 0.98 (`parr`) and 0.997 (`mean`).

## Eval year (descriptive, once)

`training.bayes_model evaluate`: 1,448 runs, 2025-10-20 → 2026-10-16, through the live imputation, official metric. Results: `src/experiments/bayes_model_eval_year.csv` (the constrained fit).

| model | S_final | vs reference | CME | stream arrival | fast stream | intermediate | slow wind |
|---|---|---|---|---|---|---|---|
| **bayes model (constrained fit)** | 117.80 | **−0.15%** | −1.2% | +1.9% | +3.7% | −0.5% | −1.0% |
| bayes model (plain latent fit, two modes) | 117.55 | −0.37% | −1.1% | +1.4% | +3.2% | −0.2% | −0.9% |
| reference (same table) | 117.98 | — | — | — | — | — | — |
| shipped linear model | 117.98 | 0.00% | | | | | |
| persistence | 157.28 | +33% | | | | | |

- **The gain is smaller than on the folds:** −0.15% with the converged fit, against −0.41% on the folds. The unconverged fit's −0.37% is not a reason to keep it: the eval year doesn't choose between fits, and the folds' features came from converged fold fits.
- **It comes from different places than on the folds:** better in CME, slow-wind and intermediate runs, worse at stream arrivals (+1.9%) and in fast streams (+3.7%). On the folds the gain was mostly in fast streams (−2.5%).
- **The setup checks out:** the reference refitted on the same table equals the shipped linear model exactly.

The eval year doesn't change the choice: the model was chosen on the folds, and it stays an option. The constrained fit has not been scored on the six folds.
