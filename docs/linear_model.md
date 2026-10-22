# Linear model (GLM on speed history and regressors)

The first model to use the regressors from `docs/features.md`. It's a per-lead
generalised linear model (`src/models/linear_model.py`), selected and trained by
`src/training/linear_model.py`.

On the eval year it beats the final baseline on pooled TIMSE (−15%), per-run mean
TIMSE (−6%) and per-run median TIMSE (−15%). It is worse on the p90 of per-run TIMSE
(+25%). The submission stays the final baseline; the linear model runs live only
when `SWP_MODEL=linear` is set.

## Pipelines

| | Steps |
|---|---|
| **Training** | input features → feature selection → regression. A run with any selected regressor missing is dropped (441 of 5,844 training runs). |
| **Live** | input features → imputation (training medians) → feature selection → regression. The selected regressors are built for t0 by the same code as in training (`feature_engineering.windows`); a missing one takes its training median. |

For each lead k = 1…72 h:

    eta_k  = offset + a_k + Σ_j b_kj · s_j(k) + Σ_i c_ki · x_i
    E[y_k] = g⁻¹(eta_k)

- `s_j(k)` are the baseline's speed terms, always used: the last observed speed, and the mean speed within ±3 h of t0 + k − 27 d and of t0 + k − 13.5 d.
- `x_i` are the selected regressors, standardised.
- With "relative to persistence", the Gaussian target is BC(y) − BC(last). The ridge penalty then shrinks towards persistence instead of the mean.

## Design choices

**Library: numpy for training and inference.**
- **Training.** Every variant is fitted by iteratively reweighted least squares, as a batched weighted ridge solve over the 72 leads (Gaussian needs one pass). The competition loss enters as per-run TIMSE weights that differ by run and lead, which scikit-learn estimators don't take natively.
- **Inference.** One matrix product and the inverse link, under a millisecond.
- **Asset.** `src/assets/linear_model.json` (coefficients, scaler, medians, smearing quantiles).
- **No new image dependency.** scikit-learn would add about 140 MB (with SciPy) and 3.5 s of import time for nothing at inference.
- **Parity checks.** scikit-learn is used only in tests. `LinearRegression`, `TweedieRegressor(power=2, link="log")` and `TweedieRegressor(power=3, link="log")` give the same coefficients as our Gaussian, Gamma and inverse-Gaussian fits.

**Model family and target transformation.** These were chosen on the training folds, with a shortlist of 9 regressors from the screening. The score is per-run mean TIMSE relative to persistence, as a geometric mean over the two folds (lower is better). Each row below is the best ridge for that variant.

| Family / link | Box-Cox λ | Relative to persistence | Loss weights | Score |
|---|---|---|---|---|
| **Gaussian / identity** | **1 (none)** | **yes** | **run TIMSE** | **0.4970** |
| Gaussian / identity | 0.5 | yes | run TIMSE | 0.4980 |
| Gaussian / identity | 1 (none) | no | run TIMSE | 0.4996 |
| Gaussian / identity | 0 (log) | yes | run TIMSE | 0.5077 |
| Gamma / log | — | yes (offset log last) | run TIMSE | 0.5186 |
| Inverse Gaussian / log | — | yes (offset log last) | run TIMSE | 0.5233 |
| Gaussian / identity | −0.5 | no | run TIMSE | 0.5336 |
| any | any | any | uniform | 0.687–0.723 |

What the variants show:
- **Box-Cox:** the more a transform compresses the high-speed tail, the worse it does: λ = 1 ≈ 0.5 < 0 (log) < −0.5. TIMSE weights errors at high speeds most, and compressing the tail fights that.
- **GLMs:** Gamma and inverse Gaussian (log link) model the mean directly and avoid the back-transform bias, but their deviance scores relative errors. That costs 4–5%.
- **The loss weights matter far more than the family:** uniform weights lose about 38%.
- **Relative to persistence** helps slightly.

**Feature lookback windows.** Each regressor is a (signal, statistic, window) candidate: 117 in all, for example Bt as a mean over 1, 3, 6, 12, 24 or 48 h, and the coronal-hole strip every 6 h from 0 to 5 days before t0. Selection keeps one window per signal and statistic. Scores when added first to the speed terms (lower is better; speed terms alone 0.5326):

| Signal | Score by window |
|---|---|
| coronal-hole strip | 0 h 0.5030 · 6 h 0.4986 · **12 h 0.4965** · 18 h 0.4968 · 24 h 0.4990 · 36 h 0.5082 · 48 h 0.5191 · 72 h 0.5354 · 120 h 0.5347 |
| density mean | 1 h 0.5245 · 6 h 0.5188 · 12 h 0.5182 · **24 h 0.5181** · 48 h 0.5253 |
| Bt mean | 1 h 0.5181 · 3 h 0.5176 · **6 h 0.5171** · 12 h 0.5188 · 24 h 0.5211 · 48 h 0.5253 |
| field variability mean | 1 h 0.5377 · 6 h 0.5348 · 24 h 0.5313 · **48 h 0.5276** |
| peak flare flux | 24 h 0.5329 · 48 h 0.5317 · 72 h 0.5312 · **120 h 0.5302** |
| F10.7 mean | 1 d 0.5383 · **7 d 0.5340** · 27 d 0.5374 |

- **Coronal holes:** images from 6–24 h before t0 carry the most information. Older images lose it steadily, as holes rotate past the central meridian.
- **In-situ signals:** 6–24 h means work best, longer for the slowly varying field variability.

**Feature selection.** Greedy forward selection on the training folds. It starts from the speed terms, adds the candidate that improves the score most at each step, and stops below a 0.5% gain. The ridge is then re-tuned (0.1).

| Step | Added | Score | Gain |
|---|---|---|---|
| 0 | speed terms only | 0.5326 | |
| 1 | `ch_cm_at_12h`: coronal-hole strip 12 h before t0 | 0.4965 | 6.8% |
| 2 | `density_mean_24h` | 0.4828 | 2.8% |
| 3 | `xray_peak_max_48h`: peak flare flux | 0.4789 | 0.8% |
| 4 | `b_var_mean_48h`: field variability | 0.4765 | 0.5% |
| 5 | `f107_mean_168h`: F10.7, 7 days | 0.4740 | 0.5% |
| (6) | `bt_mean_6h` | 0.4731 | 0.2%, below the threshold |

- **Score against the baseline:** 0.474 on the folds, against 0.518 for the final baseline on the same folds.
- **Bt:** it was the strongest regressor in the screening, but adds little once density and the coronal-hole strip are in.

**Pilot.** As agreed, the selection was first run on 500 random training runs (seed 0, `*_pilot500_seed0` files):
- **Measured:** 42 s and 16 MB.
- **Extrapolated for 5,844 runs:** 10–25 min.
- **Full run:** 8 min and 126 MB.
- **Pilot selection:** the coronal-hole strip first as well, at 6 h instead of 12 h, then a different and shorter tail. 130 validation runs per fold are too few to rank small gains.

## Eval year (2025-10-20 → 2026-10-20, 1,448 runs; scored once)

| | Linear model | Final baseline | Persistence |
|---|---|---|---|
| Pooled TIMSE | **4.75e7** | 5.58e7 | 8.13e7 |
| Per-run mean | **2.44e6** | 2.59e6 | 5.57e6 |
| Per-run median | **7.21e5** | 8.45e5 | 7.76e5 |
| Per-run p90 | 4.00e6 | **3.21e6** | 1.38e7 |
| RMSE (km/s) | **105.5** | 109.7 | 110.0 |

By regime (median per-run TIMSE ratio; regimes as in `final_baseline_analysis.ipynb`):

| Regime | Runs | Linear / baseline | Linear beats baseline | Linear / persistence | Baseline / persistence |
|---|---|---|---|---|---|
| CME | 295 | 0.93 | 59% | 0.54 | 0.53 |
| possible CME | 81 | 1.02 | 48% | 0.95 | 0.95 |
| stream arrival | 304 | 1.13 | 39% | 0.29 | 0.23 |
| fast stream | 105 | 1.07 | 49% | 0.74 | 0.65 |
| slow wind | 186 | **0.75** | **75%** | 5.4 | 6.6 |
| intermediate | 477 | **0.88** | 59% | 1.26 | 1.64 |

- **Quiet wind improves most,** as the screening suggested. In slow wind the linear model beats the baseline in 75% of runs. It is also the first model to beat persistence on the per-run median.
- **Stream arrivals and fast streams get somewhat worse,** hence the worse p90.
- **The over-forecast remains:** the mean residual is −49 km/s at 24 h and −103 km/s at 72 h (baseline: −59 and −105). Five regressors reduce it at short leads only.
- **Coronal-hole gap:** `ch_cm_at_12h` was imputed in 20% of eval runs because of the SDO archive gap (about 2025-09-12 to 2025-12-16). The other regressors were imputed in under 1% of runs.

**Full analysis:** `src/experiments/linear_model.ipynb`. It covers:
- the regimes, the bias and the stream arrivals;
- the coefficients, an ablation of each regressor, and the coronal-hole gap.

Main additional findings:
- **One event dominates the gain:** 69% of the per-run-mean gain comes from the January 2026 CME.
- **Large coronal holes:** the model over-forecasts when the hole at the central meridian is large.
- **A glitch in the training data** (1,087 km/s at 2022-10-05 02:00) makes the coefficients jump every sixth lead.

## Live

- `SWP_MODEL=linear` makes `model.py` use `inference.run_linear_forecast`.
  - The speed history and the regressors are fetched in parallel. Only the AIA image lags the model uses are fetched.
  - Missing regressors take their medians.
  - Any failure falls back to the final baseline, then persistence.
- **In the image** (4 vCPU / 16 GiB, t0 2026-10-22 06Z): 5 of 5 regressors observed, 21 s, a valid forecast, image unchanged at 90 MiB.
- Without the variable, `model.py` runs the final baseline as before.

## Reproduce

```bash
python -m training.linear_model select --sample 500 --seed 0   # pilot (timed)
python -m training.linear_model select --max-seconds 480       # full; re-run until it finishes
python -m training.linear_model train
python -m training.linear_model evaluate                       # eval year, once
```
