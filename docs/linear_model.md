# Linear model (GLM on speed history and regressors)

> **Submitted model since 2026-10-26**, with loss weights re-tuned for the official metric (`weights="official"`, `blend=0.25`; [official_retune.md](official_retune.md)). The selection below was made on the old per-run TIMSE; re-run on the official metric (2026-10-29) it picks the same variant and regressor families, so the config is unchanged ([official_reselection.md](official_reselection.md)).

The first model to use the regressors from `docs/features.md`. It's a per-lead
generalised linear model (`src/models/linear_model.py`), selected and trained by
`src/training/linear_model.py`.

**The shipped model** was re-selected and refitted on 9 years with six yearly folds
([Nine years of history](#nine-years-of-history)). It has four regressors:
- Bt, mean of the last hour;
- the coronal-hole strip 24 h before t0;
- peak flare flux over 72 h;
- density, 24 h mean.

It keeps the [data fixes](#data-fixes): glitches removed from the training speeds, and a submodel
for runs without a coronal-hole image.

On the eval year it beats the final baseline on:
- pooled TIMSE (−14%);
- per-run mean TIMSE (−10%);
- per-run median TIMSE (−21%);
- 66% of runs won.

It is worse on the p90 of per-run TIMSE (+7%).

Since 2026-10-24 the submission is the XGBoost hybrid, which uses this model as its anchor and
first fallback; `SWP_MODEL=linear` submits the linear model alone. The sections before "Nine years of history" describe the 4-year
fits, and the analysis is `src/experiments/linear_model.ipynb`.

## Pipelines

| | Steps |
|---|---|
| **Training** | input features → feature selection → regression. A run with any selected regressor missing is dropped (441 of 5,844 training runs). |
| **Live** | input features → imputation (training medians) → feature selection → regression. The selected regressors are built for t0 by the same code as in training (`feature_engineering.windows`); a missing one takes its training median, except the coronal-hole strip: a run without it goes to a submodel fitted without it (see [Data fixes](#data-fixes)). |

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

These are the results for the first fit, before the [data fixes](#data-fixes).

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

The eval analysis of this first fit found:
- **One event dominated the gain:** 69% of the per-run-mean gain came from the January 2026 CME.
- **Large coronal holes:** the model over-forecast when the hole at the central meridian was large.
- **A glitch in the training data** (1,087 km/s at 2022-10-05 02:00) made the coefficients jump every sixth lead.

These led to the data fixes below. **The notebook `src/experiments/linear_model.ipynb` now analyses the shipped model** (after the fixes). It covers:
- the regimes, the bias and the stream arrivals;
- the coefficients, contributions and ablation of each regressor;
- the coronal-hole gap, the despiking, and forecasts along the year.

## Data fixes

The eval analysis found three data problems. All are fixed for every config scored on the
folds and for the shipped fit (`training.linear_model.with_fixes`). The third, the regressor
cap, came later (2026-10-24) and is described after the re-selection below.

**1. Glitches in the training speeds (`Config.despike`).**
- **The glitch:** one hour read 1,087 km/s at 2022-10-05 02:00, between hours of about 500 km/s. Runs start every 6 hours, so it was the target of leads 2, 8, 14, … h. Because TIMSE weights favour a run's fastest hours, it took up to 2% of the fitting weight at each of those leads, and the coefficients jumped every sixth lead.
- **The rule:** `feature_engineering.recurrence.despike` drops hour h when the medians of the 3 hours before and the 3 hours after agree within 75 km/s and h differs from both by more than 150 km/s in the same direction. A shock or a stream onset is a step, so its two sides disagree and it is kept. The threshold was set on the training years: it flags 4 hours in 2021–2025.
- **Where it applies:** the cleaned series is used for the fitting targets and the recurrence terms. Their windows end at least 246 h before t0, so cleaning them uses only past hours. The last observed speed stays raw: its hour can't be checked against later hours.
- **Scoring is unchanged:** it always uses the raw series, as the platform does.
- **Result:** the lead-to-lead jumps in the coefficients are 3–7× smaller.

**2. The coronal-hole gaps (`windows.CH_FALLBACK_HOURS`, `Config.optional`).**
- **Fallback image:** if the image at the lag is missing, the strip comes from the image 6 h, then 12 h older. This covers short outages live. It recovers only 7 of the 332 missing training runs and 6 of the 285 missing eval runs: almost all missing values come from long archive gaps (40 days in Nov 2024 – Jan 2025, 96 days in Sep – Dec 2025, 9 days in Mar 2026).
- **Submodel:** the coronal-hole regressor is "optional". The model also fits a submodel without it, on every run the other regressors cover (5,735). A run still missing the strip is forecast by the submodel instead of with the median strip. Both models are in the asset (format 2).

**Fold scores** for the first fit's five regressors (lower is better; eval year not used):

| | Score | Fold 1 | Fold 2 |
|---|---|---|---|
| First fit (no fixes) | 0.4740 | 0.5562 | 0.4039 |
| despike only | 0.4735 | 0.5453 | 0.4111 |
| image fallback only | 0.4732 | 0.5564 | 0.4024 |
| submodel only | 0.4664 | 0.5562 | 0.3911 |
| fallback + submodel | 0.4659 | 0.5564 | 0.3901 |
| **all fixes (shipped)** | **0.4646** | 0.5455 | 0.3956 |

- **The submodel brings most of the gain,** in fold 2, which contains the Nov 2024 – Jan 2025 gap.
- **Despiking is about neutral on the score.** Its value is the stable fit.
- **The regressors were selected before the fixes,** so the selection was re-run with them.

### Re-selection with the fixes (the shipped model)

The full variant search and forward selection were run again with both fixes:
- **Timing:** the 500-run pilot took 52 s; the full run took 8.5 min.
- **Variant:** the same winner, Gaussian with no transform, relative to persistence.

| Step | Added | Score | Gain |
|---|---|---|---|
| 0 | speed terms only | 0.5347 | |
| 1 | `ch_cm_at_12h` | 0.4861 | 9.1% |
| 2 | `density_mean_24h` | 0.4701 | 3.3% |
| 3 | `xray_peak_max_48h` | 0.4660 | 0.9% |
| 4 | `b_var_mean_48h` | 0.4633 | 0.6% |
| (5) | `bt_max_6h` | 0.4624 | 0.2%, below the threshold |

- **Same regressors, minus F10.7:** the first four regressors and their windows are unchanged. F10.7 no longer clears the 0.5% gain.
- **Ridge:** it re-tunes to 0 (0.4614, against 0.4633 at 0.1).
- **Fold score:** 0.4614, against 0.4740 for the first fit and 0.518 for the final baseline.

**Eval year, scored once after the re-selection:**

| | Shipped (fixes) | First fit | Final baseline | Persistence |
|---|---|---|---|---|
| Pooled TIMSE | **4.74e7** | 4.75e7 | 5.58e7 | 8.13e7 |
| Per-run mean | **2.37e6** | 2.44e6 | 2.59e6 | 5.57e6 |
| Per-run median | **6.76e5** | 7.21e5 | 8.45e5 | 7.76e5 |
| Per-run p90 | 3.76e6 | 4.00e6 | **3.21e6** | 1.38e7 |
| RMSE (km/s) | **102.2** | 105.5 | 109.7 | 110.0 |

| Regime | Runs | Shipped / baseline (median) | Beats baseline | First fit / baseline (median) |
|---|---|---|---|---|
| CME | 295 | **0.89** | 66% | 0.93 |
| possible CME | 81 | **0.82** | 70% | 1.02 |
| stream arrival | 304 | 1.12 | 41% | 1.13 |
| fast stream | 105 | **0.98** | 51% | 1.07 |
| slow wind | 186 | **0.62** | 83% | 0.75 |
| intermediate | 477 | **0.76** | 73% | 0.88 |

- **Every summary score improves on the first fit.** The gains are largest on the typical run: the median is 20% below the baseline's, and the model wins 83% of slow-wind and 73% of intermediate runs.
- **Stream arrivals remain the weak spot** (median ratio 1.12), and so does the p90.
- **The 72 h over-forecast is unchanged** (−105 km/s, as for the baseline). At 24 h it is −49 km/s, against −59 for the baseline.
- **Submodel use:** 19% of eval runs had no coronal-hole image and went to the submodel. On those runs the model now beats the baseline (median ratio 0.88, wins 61%), where the first fit only tied it.
- **The gain is broader:** the January 2026 CME accounts for 49% of the per-run-mean gain (first fit 69%), and the model wins more runs than it loses in 11 of 13 months.
- **A smaller six-lead pattern remains.** It is structural: with 6-hourly runs, each lead is fitted only on target hours at one time of day modulo 6. Training on hourly forecast times would remove it.

### 3. The regressor cap (`Config.cap_quantile`, 2026-10-24)

- **The problem:** the eval-year run of 2026-01-20 00Z had a last-hour Bt of 61 nT, far above anything in training, and the linear Bt term extrapolated it into a large over-forecast (15× the baseline's TIMSE).
- **The fix:** each regressor is clipped to its training [q, 1 − q] quantiles, in fitting and in forecasting. The bounds are stored in the asset; the coronal-hole submodel has its own.
- **Chosen on the six yearly folds** (`python -m training.regressor_cap`, `src/experiments/regressor_cap.ipynb`), refitting the linear model and the hybrid's trees on top for each q:

| q | Linear | Hybrid | Linear vs no cap | Hybrid vs no cap | Folds won (linear / hybrid) |
|---|---|---|---|---|---|
| 0 (before) | 0.4387 | 0.4258 | | | |
| 0.001 | 0.4377 | 0.4254 | −0.24% | −0.10% | 6 / 3 |
| **0.005 (shipped)** | **0.4375** | **0.4253** | −0.27% | −0.11% | 5 / 4 |
| 0.01 | 0.4375 | 0.4263 | −0.28% | +0.12% | 5 / 3 |
| 0.025 | 0.4379 | 0.4254 | −0.18% | −0.08% | 3 / 2 |

- **The score gain is small,** near fold-to-fold noise; the point is robustness. The shipped nine-year bounds are Bt 1.47–19.2 nT, X-ray peak 10⁻⁷–10⁻³·²⁴ W/m², density 0.22–12.9 cm⁻³, coronal-hole strip 0–0.284.
- **Where it acts:** Bt in up to 2% of fold runs; the X-ray peak in 7–11% of runs in the 2021 and 2023 folds, whose training years were near solar minimum.
- **Both assets were retrained** with the cap (`training.linear_model.CAP_QUANTILE`). The eval year was not re-scored.

### 4. Lead-aligned coronal holes (`ch_cm_lead_72h`, 2026-10-24)

- **The change:** the coronal-hole regressor `ch_cm_at_24h` (one image, 24 h before t0, for every lead) is replaced by `ch_cm_lead_72h`. For lead k it takes the strip of the image at lag L_k = 72 − k before t0, rounded up to the 6-hour image grid and at least 0: the image about 72 h before the hour being forecast.
- **Chosen on the six yearly folds** by `training.lead_aligned_ch` ([lead_aligned_ch.md](lead_aligned_ch.md)): delays of 60–120 h were tried, and 72 h was best.
- **How the model takes it:** `Config.input_features` expands it into the 13 lagged columns `ch_cm_at_0h` … `ch_cm_at_72h`, and each lead's design uses its own column (one fit, per-column scaling, cap and medians). A missing lagged image is filled from the nearest one of the same run within 24 h (`LinearModel.fill_leads`); if the strip is still missing, the no-coronal-hole submodel forecasts the run, as before.

| Six folds | Before (`ch_cm_at_24h`) | Lead-aligned | Change |
|---|---|---|---|
| Linear | 0.4375 | 0.4256 | −2.7% |
| Hybrid (trees refitted on it) | 0.4253 | 0.4190 | −1.5% |

- **Fit:** 11,300 runs (11,301 before).
- **Live:** 15 AIA images per run instead of 4; the full live run at 2026-10-24 00Z took 14 s, with 18.6 MB of inputs.
- **The eval year was not re-scored.**

## Live

- `SWP_MODEL=linear` makes `model.py` use `inference.run_linear_forecast`.
  - The speed history and the regressors are fetched in parallel. Only the AIA image lags the model uses are fetched.
  - Missing regressors take their medians. A missing coronal-hole strip (after the 12 h → 18 h → 24 h image fallback) switches to the submodel without it.
  - Any failure falls back to the final baseline, then persistence.
- **In the image** (4 vCPU / 16 GiB, t0 2026-10-22 06Z): all regressors observed (5 of 5 for the first fit; 4 of 4 for the shipped model, re-checked locally), 21 s, a valid forecast, image unchanged at 90 MiB. The image fallback fetches three AIA images (12, 18, 24 h) instead of one.
- Without the variable, `model.py` runs the XGBoost hybrid (since 2026-10-24), with this model as its first fallback.

## Reproduce

```bash
python -m training.linear_model select --sample 500 --seed 0   # pilot (timed)
python -m training.linear_model select --max-seconds 480       # full; re-run until it finishes
python -m training.linear_model train
python -m training.linear_model evaluate                       # eval year, once
```

## Nine years of history

**Data and folds.** The training history was extended to 2016-10 → 2025-10 (`docs/evaluation.md`),
and the selection now uses six yearly walk-forward folds, 2019-10 → 2025-10
(`training.folds.SELECTION`).

**Selection:**
- **Process:** a 1,000-run pilot (4 min), then the full run (about 1 h, in resumable slices; stage 1 now caches each variant).
- **Variant:** the same winner, Gaussian, no transform, relative to persistence (0.4487 on the shortlist).

| Step | Added | Fold score |
|---|---|---|
| 0 | speed terms only | 0.4949 |
| 1 | `bt_mean_1h` | 0.4759 |
| 2 | `ch_cm_at_24h` | 0.4563 |
| 3 | `xray_peak_max_72h` | 0.4423 |
| 4 | `density_mean_24h` | 0.4387 |
| (5) | `polarity_mean_48h` | 0.4376, a 0.26% gain, below the threshold |

- **Score:** 0.4387 on the six folds, against 0.4523 for the 4-year settings refitted on the same 9 years (−3.0%). The ridge stays at 0.1.
- **Fit:** 11,301 runs, 6 glitch hours despiked, and 11,632 runs for the no-coronal-hole submodel.

**Eval year** (`src/experiments/linear_model.ipynb`):

| | 9-year linear | 4-year linear | Final baseline |
|---|---|---|---|
| Pooled TIMSE | 4.82e7 | **4.74e7** | 5.58e7 |
| Per-run mean | **2.34e6** | 2.37e6 | 2.59e6 |
| Per-run median | **6.67e5** | 6.76e5 | 8.45e5 |
| Per-run p90 | 3.43e6 | 3.76e6 | **3.21e6** |

- **Stream arrivals are now level with the baseline** (median ratio 0.98, against 1.12).
- **The model wins more runs than the baseline in 12 of 13 months.**
- **New failure:**
  - **The run:** 2026-01-20 00Z, at 15× the baseline's TIMSE.
  - **The cause:** the last hour's Bt was 61 nT, far outside the training range, and the linear Bt term extrapolated it into a large over-forecast.
  - **The fix:** cap the regressors at a high training quantile, chosen on the folds.
