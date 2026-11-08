# Post-hoc corrections of the linear forecast (not built)

**The question.** Can a correction applied after the forecast fix the biases the performance study found ([model_diagnostics.md](model_diagnostics.md))?
- **Slow wind:** slow hours are 41 km/s too high.
- **High forecasts:** forecasts of 650–700 km/s are 90 km/s too high.

This is items (i b) and (i d) of the follow-up plan: a persistence blend and a calibration model. Quantile mapping, a distribution map, was tried before and failed ([quantile_mapping.md](quantile_mapping.md)). These are regressions on the residual instead, with the official-metric loss weights.
- **Study:** `python -m training.post_calibration` (`--pilot --folds 2024`).
- **Analysis:** `src/experiments/post_calibration.ipynb`; results in `src/experiments/post_calibration.csv`.

**Result: not built.** On the pilot fold (2024), every correction makes the score worse, by +0.2% to +23%. Most also make the slow-wind over-forecast larger.
- **Scope:** run on the 2024 fold only. The pilot showed no variant near the rule, so the other five folds were not run (user decision, 2026-11-08).
- **Why 2024 is a fair test:** it is the fold with the largest over-forecast (−52 km/s overall, −71 km/s on slow runs). If a correction was going to help anywhere, it was there.

## Method

- **Corrections,** with residual r = observed − forecast, fitted with `official_weights` at blend 0.25 against the official truth:
  - **bias:** one offset per lead;
  - **persistence blend:** f + w(k)·(persistence − f), with w a quadratic in the lead, optionally a separate w(k) when the last speed is below 450 km/s;
  - **calibration:** per lead, a ridge regression of r on the forecast and the last speed, plus hinges of both (forecast at 450, 550 and 650 km/s; last speed below 450 and above 650). Ridge 0.1 or 1. Also a version fitted on and applied to slow runs (last speed < 450) only.
- **Where each correction is fitted:**
  - **Nested (the design):** the submitted config fitted on the runs before the fold minus one year forecasts that last year. The correction is fitted on those forecasts, then applied to the fold forecasts of the config fitted on all runs before the fold. The correction never sees the fold.
  - **In-sample (added after the first pilot):** the fold's own model on all its fit runs. That is more data and the same model, but the residuals are in-sample.
- **Reference:** the submitted config refitted on the runs before the fold, 0.7849 of persistence on the 2024 fold (as in the other studies).
- **Decision rule, set before running:** as in [slow_wind.md](slow_wind.md): more than 1% better on at least 4 of 6 folds, with slow runs not worse.

## Results (2024 fold)

| Correction | Fitted on | S_final vs reference | Slow runs vs reference | Slow runs' mean residual (km/s) | All runs' mean residual |
|---|---|---|---|---|---|
| reference | — | — | — | −71 | −52 |
| blend, slow apart | all fit runs (in-sample) | +0.16% | +10.1% | −78 | −57 |
| calibration, ridge 1 | all fit runs (in-sample) | +0.28% | +3.0% | −73 | −52 |
| bias | the year before | +1.5% | +8.7% | −81 | −63 |
| blend, slow apart | the year before | +2.2% | +18.4% | −84 | −62 |
| persistence blend | the year before | +2.5% | +18.5% | −84 | −63 |
| calibration, slow only | the year before | +3.7% | +25.3% | −90 | −63 |
| calibration, ridge 1 | the year before | +9.8% | +12.3% | −83 | −61 |
| calibration, ridge 0.1 | the year before | +23.1% | +19.5% | −86 | −60 |

## Why it fails

- **Under the official weights, the correction pushes forecasts up.** On the year before the 2024 fold, the plain mean residual is −23 km/s, but the official-weighted mean is **+10 km/s**: the weights stress the fast hours, which are under-forecast. A correction fitted with the metric's own weights therefore raises the forecast, and the fold's mean residual goes from −52 to −63 km/s. Fitted with plain weights instead, it would lower forecasts and lose the TIMSE half of the score (the trade-off in [slow_wind.md](slow_wind.md)).
- **The bias also drifts.** The mean residual by fold is −10, −20, −20, −27, −23 and −51 km/s ([model_diagnostics.md](model_diagnostics.md)), so even the direction learned from one year doesn't carry to the next.
- **The persistence blend's weight turns negative after the first hours.** Fitted on the year before, w(k) is +0.21 at 1 h, −0.14 at 24 h, −0.29 at 48 h and −0.21 at 72 h. Beyond the first day, the blend pushes forecasts *away* from persistence, so slow runs get 18% worse.
- **In-sample, there is little left to correct.** The model's own fit already absorbs the average error per lead (its intercepts) and the linear dependence on the last speed. The hinge terms find only a small nonlinear pattern, and it doesn't transfer to the fold (+0.3%).
- **The calibration on the forecast level overfits** (ridge 0.1: +23%). Its hinges extrapolate badly at the highest forecasts, where the fit year had few runs.

## Implications

- A post-step can't fix a bias that drifts with the solar cycle when it learns from the past year or from the training fit.
- The over-forecast in the latest fold (−51 km/s) is mostly a level shift that arrived with the solar maximum ([slow_wind.md](slow_wind.md) found that lowering the blend helps only that fold).
- A correction would need a signal of the cycle phase that is available at t0. The recency weights ([recency_weights.md](recency_weights.md)) and the F10.7 level ([new_features.md](new_features.md)) were tried for that and didn't pass.
